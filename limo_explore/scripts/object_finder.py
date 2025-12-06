#!/usr/bin/env python3

import os, math, time
import numpy as np
import rospy
import torch
import message_filters
import tf2_ros, tf2_geometry_msgs
from cv_bridge import CvBridge
import cv2
from PIL import Image as PILImage

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, PointStamped, Quaternion, Point, Twist
from std_msgs.msg import String, Bool
from tf.transformations import quaternion_from_euler, euler_from_quaternion

from pymycobot import MyCobot280

from nanoowl.owl_predictor import OwlPredictor


class ObjectFinder:
    """
    Object detection, navigation, and grasping node for a mobile manipulator.

    The node runs a finite state machine with the following states:

    States:
        - S_SEARCHING:
            Actively searches for the target object in synchronized RGB-D data.

        - S_APPROACHING:
            A reliable object detection was achieved and a navigation goal
            has been sent. The robot waits until the navigation stack reports
            that the goal has been reached.

        - S_ARRIVED:
            The robot reached the navigation goal and performs fine yaw alignment
            to center the object in the camera image.

        - S_ALIGNED:
            The object is centered in the image and the robot is properly aligned.
            The system decides whether to perform a short forward close-in motion
            or proceed directly to grasp computation.

        - S_CLOSING_IN:
            The robot moves forward using velocity control until a minimum
            standoff distance to the object is reached.
    """
    S_SEARCHING = 0
    S_APPROACHING = 1
    S_ARRIVED = 2
    S_ALIGNED = 3
    S_CLOSING_IN = 4

    def __init__(self):
        """
        Initialize the object finder node.
        Loads parameters and state, sets up detector, TF, robot, ROS communication, and trajectory logging.
        """
        rospy.init_node("object_finder")

        self._load_params()
        self._init_state()
        self._init_detector()
        self._init_arm()
        self._init_ros_comm()
        self._init_traj_logging()

        # signal readiness for other nodes
        self.pub_ready.publish(Bool(data=True))
        rospy.loginfo(
            "object_finder ready | prompt=%s thr=%.2f engine=%s",
            self.prompt, self.threshold, self.engine_path
        )

    def _load_params(self):
        """
        Load all ROS parameters for detection, navigation, depth filtering and grasping.
        """
        # Detector
        self.prompt       = rospy.get_param("~prompt", "a water bottle")
        self.threshold    = rospy.get_param("~threshold", 0.15)
        self.engine_path  = rospy.get_param(
            "~image_encoder_engine",
            os.path.expanduser("~/nanoowl/data/owl_image_encoder_patch32.engine")
        )
        self.min_hits     = rospy.get_param("~min_hits", 3)
        self.max_age      = rospy.get_param("~max_age_sec", 2.0)

        # Navigation
        self.target_frame = rospy.get_param("~target_frame", "map")
        self.base_frame   = rospy.get_param("~base_frame",   "base_link")

        # Camera / depth topics & frame
        self.camera_frame = rospy.get_param("~camera_frame", "camera_depth_optical_frame")
        self.rgb_topic    = rospy.get_param("~rgb_topic",   "/camera/color/image_raw")
        self.depth_topic  = rospy.get_param("~depth_topic", "/camera/depth/image_raw")
        self.info_topic   = rospy.get_param("~info_topic",  "/camera/color/camera_info")

        # Depth robustness
        self.depth_min        = rospy.get_param("~depth_min", 0.10)
        self.depth_max        = rospy.get_param("~depth_max", 4.50)
        self.near_percentile  = rospy.get_param("~near_percentile", 15.0)
        self.band_width_m     = rospy.get_param("~band_width_m", 0.02)
        self.bbox_shrink_px   = rospy.get_param("~bbox_shrink_px", 6)
        self.median_patch_px  = rospy.get_param("~median_patch_px", 7)

        # Navigation close-in behaviour
        self.min_standoff   = rospy.get_param("~min_standoff", 0.30)    # meters
        self.close_in_speed = rospy.get_param("~close_in_speed", 0.10)  # m/s

        # Top grasp config
        self.top_percentile   = rospy.get_param("~top_percentile", 90.0)
        self.grasp_z_lift     = rospy.get_param("~grasp_lift_m", 0.02)
        self.publish_debug    = rospy.get_param("~publish_debug", True)

        # Trajectory logging
        self.traj_max_duration_sec = rospy.get_param("~traj_max_duration_sec", 6 * 60.0)
        self.traj_log_dir = rospy.get_param(
            "~traj_log_dir", os.path.expanduser("~/traj_logs")
        )

        # MyCobot connection params (used in _init_robot)
        self.mc_port = rospy.get_param("~port", "/dev/ttyACM0")
        self.mc_baud = rospy.get_param("~baud", 115200)

    def _init_state(self):
        """
        Initialize internal state variables and finite state machine.
        """
        # FSM & detection state
        self.state = self.S_SEARCHING
        self.query = self.prompt
        self.hits_in_row = 0
        self.last_seen_ts = 0.0
        self.sent_goal_for_this_track = False
        self.approached = False
        self.last_detection = None  # for debug / grasp

        # Close-in state
        self._close_in_done = False

        # Trajectory logging (arrays will be filled in _init_traj_logging)
        self.traj_logging_active = True
        self.traj_start_time = rospy.Time.now()
        self.traj_last_xy = None
        self.traj_total_dist = 0.0
        self.traj_log = []  # list of (t_sec, x, y, yaw, cum_dist)

    def _init_detector(self):
        """
        Initialize the OWL-ViT predictor and initial text encoding.
        """
        self.predictor = OwlPredictor(
            "google/owlvit-base-patch32",
            image_encoder_engine=self.engine_path
        )
        self.text_encoding = self.predictor.encode_text(self.query)

    def _init_arm(self):
        """
        Initialize MyCobot arm and move it to a safe default configuration.
        """
        self.mc = MyCobot280(self.mc_port, self.mc_baud)
        # initial "safe" pose for the arm
        self.mc.send_angles([-90.0, 0.0, -10.0, -90.0, 0.0, 57], 50)

    def _init_ros_comm(self):
        """
        Set up publishers, subscribers and synchronized RGB+depth+info callback.
        """
        self.bridge = CvBridge()
        self.tfbuf = tf2_ros.Buffer(rospy.Duration(10.0))
        self.tfl   = tf2_ros.TransformListener(self.tfbuf)

        # Publishers
        self.pub_found  = rospy.Publisher("/object_found", Bool, queue_size=1, latch=True)
        self.pub_ready  = rospy.Publisher("/object_detection_ready", Bool, queue_size=1, latch=True)
        self.pub_pose   = rospy.Publisher("/object_pose",  PoseStamped, queue_size=1, latch=True)
        self.pub_goal   = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1)
        self.pub_cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.pub_dbg    = (rospy.Publisher("/object_debug", Image, queue_size=1) if self.publish_debug else None)

        # Image + depth + camera info sync
        s_rgb   = message_filters.Subscriber(self.rgb_topic, Image)
        s_depth = message_filters.Subscriber(self.depth_topic, Image)
        s_info  = message_filters.Subscriber(self.info_topic, CameraInfo)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [s_rgb, s_depth, s_info],
            queue_size=10,
            slop=0.12,
        )
        self.sync.registerCallback(self.tick)

        # Other subscribers
        rospy.Subscriber("/object_approached", Bool, self.approached_cb, queue_size=1)
        rospy.Subscriber("/object_query", String, self.query_cb, queue_size=1)

    def _init_traj_logging(self):
        """
        Prepare trajectory logging (path, directory) and start sampling timer.
        """
        os.makedirs(self.traj_log_dir, exist_ok=True)
        run_id = time.strftime("%Y%m%d_%H%M%S")
        self.traj_log_path = os.path.join(self.traj_log_dir, f"traj_{run_id}.txt")

        self.traj_start_time = rospy.Time.now()
        self.traj_last_xy = None
        self.traj_total_dist = 0.0
        self.traj_log = []

        # timer to sample pose
        self.traj_timer = rospy.Timer(
            rospy.Duration(0.1),
            self._traj_timer_cb,
        )

    # ---------------- Callbacks ----------------

    def query_cb(self, msg: String):
        """
        Update the object search query at runtime.
        Resets detection history and re-encodes the text prompt for the detector.
        """
        self.query = msg.data.strip()
        self.hits_in_row = 0
        self.sent_goal_for_this_track = False
        self.text_encoding = self.predictor.encode_text(self.query)
        rospy.loginfo("Updated prompt: %s", self.query)

    def approached_cb(self, msg: Bool):
        """
        Callback indicating whether the navigation stack has reached the object.
        """
        self.approached = bool(msg.data)

    def _traj_timer_cb(self, _evt):
        """
        Sample robot pose and accumulate distance while logging is active.
        """
        if not self.traj_logging_active:
            return
        
        elapsed = (rospy.Time.now() - self.traj_start_time).to_sec()
        if elapsed >= self.traj_max_duration_sec:
            rospy.loginfo("[object_finder] Trajectory max duration reached (%.1f s) -> saving log",
                          elapsed)
            self._stop_and_save_traj()
            return
        
        try:
            # pose of base in target frame (e.g. map)
            T = self.tfbuf.lookup_transform(self.target_frame, self.base_frame,
                                            rospy.Time(0), rospy.Duration(0.05))
        except Exception:
            return

        x = T.transform.translation.x
        y = T.transform.translation.y
        q = T.transform.rotation
        yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])[2]

        t_rel = (rospy.Time.now() - self.traj_start_time).to_sec()

        if self.traj_last_xy is not None:
            dx = x - self.traj_last_xy[0]
            dy = y - self.traj_last_xy[1]
            step = math.hypot(dx, dy)
            self.traj_total_dist += step

        self.traj_last_xy = (x, y)
        self.traj_log.append((t_rel, x, y, yaw, self.traj_total_dist))

    def tick(self, rgb_msg: Image, depth_msg: Image, info_msg: CameraInfo):
        """
        Main callback for synchronized RGB, depth, and camera info.
        Handles state machine, detection, 3D reconstruction, TF transforms,
        and transitions to alignment / close-in / grasp.

        Args:
            rgb_msg (Image): RGB image message.
            depth_msg (Image): Depth image message.
            info_msg (CameraInfo): Camera intrinsic parameters.
        """
        # State handling before running detection
        if self._handle_pre_detection_states():
            return

        # Convert messages to usable arrays and camera params
        frame = self._prepare_frame(rgb_msg, depth_msg, info_msg)
        if frame is None:
            return
        rgb, depth, H, W, fx, fy, cx, cy, rgb_pil = frame

        # Run detector and get best bounding box
        det = self._run_detector(rgb_pil, W, H)
        if det is None:
            self._no_detection()
            if self.publish_debug:
                self._publish_debug(rgb, None, None)
            return
        x1, y1, x2, y2, score = det

        # Foreground segmentation and pick pixel (u,v) + foreground depths
        fg = self._select_foreground_pixel(depth, x1, y1, x2, y2, W, H)
        if fg is None:
            self._no_detection()
            if self.publish_debug:
                self._publish_debug(rgb, (x1, y1, x2, y2), None)
            return
        u, v, roi, comp, dsel = fg

        # If robot is already arrived: only refine yaw alignment
        if self.state == self.S_ARRIVED:
            if self._align_on_pixel(u, cx):
                return

        # Refine depth at (u,v)
        Z = self._refine_depth_at_pick(depth, u, v, H, W, roi, comp, dsel)
        if Z is None or not (self.depth_min <= Z <= self.depth_max):
            self._no_detection()
            if self.publish_debug:
                self._publish_debug(rgb, (x1, y1, x2, y2), None)
            return

        # 2D -> 3D in camera, then to base_link and map
        tf_res = self._project_to_base_and_map(
            rgb_msg, info_msg, u, v, Z, fx, fy, cx, cy
        )
        if tf_res is None:
            self._no_detection()
            return
        p_base, p_map = tf_res

        # Debug image
        if self.publish_debug:
            self._publish_debug(rgb, (x1, y1, x2, y2), score, pick=(u, v))

        # Pose message for navigation (object pose in target frame)
        self._publish_object_pose(rgb_msg, p_map)

        # Update detection state, FSM transitions and possibly trigger close-in / grasp
        self._update_detection_state(rgb_msg, depth_msg, info_msg, x1, y1, x2, y2, 
                                     u, v, Z, p_base, p_map)
        
    def _handle_pre_detection_states(self) -> bool:
        """
        Handle state transitions that do not require a new detection.
        Returns True if no further processing should happen in this tick.
        """
        if self.state == self.S_CLOSING_IN:
            self._move_forward()
            return True

        # Waiting only for approach completion
        if self.state == self.S_APPROACHING and self.approached:
            self.state = self.S_ARRIVED
        return False

    def _prepare_frame(self, rgb_msg, depth_msg, info_msg):
        """
        Convert ROS messages to OpenCV arrays and extract camera intrinsics.

        Args:
            rgb_msg (Image): RGB image ROS message.
            depth_msg (Image): Depth image ROS message.
            info_msg (CameraInfo): Camera intrinsic parameters.

        Returns:
            tuple | None:
                (rgb, depth, H, W, fx, fy, cx, cy, rgb_pil),
                or None if conversion fails.
        """
        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            if depth.dtype == np.uint16:
                depth = depth.astype(np.float32) / 1000.0
            else:
                depth = depth.astype(np.float32)
        except Exception as e:
            rospy.logwarn_throttle(5.0, "cv_bridge err: %s", e)
            return None

        H, W = rgb.shape[:2]
        fx, fy, cx, cy = info_msg.K[0], info_msg.K[4], info_msg.K[2], info_msg.K[5]
        rgb_pil = PILImage.fromarray(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        return rgb, depth, H, W, fx, fy, cx, cy, rgb_pil
    
    def _run_detector(self, rgb_pil: PILImage.Image, W: int, H: int):
        """
        Run OWL-ViT predictor and return best bounding box.
        
        Args:
            rgb_pil (PIL.Image): RGB image in PIL format.
            W (int): Image width.
            H (int): Image height.

        Returns:
            tuple | None:
                (x1, y1, x2, y2, score) of the best detection,
                or None if no object is detected.
        """
        try:
            preds = self.predictor.predict(
                rgb_pil, self.query, self.text_encoding, threshold=self.threshold
            )
        except Exception as e:
            rospy.logwarn_throttle(5.0, "predict failed: %s", e)
            return None

        if preds.scores.numel() == 0:
            return None

        i = int(torch.argmax(preds.scores).item())
        score = float(preds.scores[i].detach().cpu().item())
        x1, y1, x2, y2 = [
            int(round(v)) for v in preds.boxes[i].detach().float().cpu().numpy()
        ]
        # clamp to image bounds
        x1 = max(0, min(x1, W - 1))
        x2 = max(0, min(x2, W - 1))
        y1 = max(0, min(y1, H - 1))
        y2 = max(0, min(y2, H - 1))

        if x2 <= x1 or y2 <= y1:
            return None

        return x1, y1, x2, y2, score

    def _select_foreground_pixel(self, depth, x1, y1, x2, y2, W, H):
        """
        Select a robust foreground pixel inside the detection bounding box.

        Args:
            depth (np.ndarray): Depth image in meters.
            x1, y1, x2, y2 (int): Bounding box pixel coordinates.
            W (int): Image width.
            H (int): Image height.

        Returns:
            tuple | None:
                (u, v, roi, comp, dsel) where:
                    u, v   - selected pixel,
                    roi    - depth sub-image,
                    comp   - foreground component mask,
                    dsel   - depth values of selected points,
                or None if selection fails.
        """
        # shrink inner border
        s = int(self.bbox_shrink_px)
        x1s = max(x1 + s, 0)
        y1s = max(y1 + s, 0)
        x2s = max(min(x2 - s, W - 1), x1s + 1)
        y2s = max(min(y2 - s, H - 1), y1s + 1)
        if x2s <= x1s or y2s <= y1s:
            x1s, y1s, x2s, y2s = x1, y1, x2, y2

        roi = depth[y1s:y2s, x1s:x2s]
        valid = np.isfinite(roi) & (roi > self.depth_min) & (roi < self.depth_max)
        if not np.any(valid):
            return None

        vals = roi[valid].ravel()
        if vals.size < 30:
            return None

        # Histogram peak near camera
        bin_width = 0.01
        bins = np.arange(
            self.depth_min,
            min(self.depth_max, np.percentile(vals, 98.0)) + bin_width,
            bin_width,
            dtype=np.float32,
        )
        hist, edges = np.histogram(vals, bins=bins)
        peak = None
        thr = max(25, int(0.01 * vals.size))
        for i in range(len(hist)):
            if hist[i] >= thr and (i == 0 or hist[i] >= hist[i - 1]) and (i == len(hist) - 1 or hist[i] >= hist[i + 1]):
                peak = i
                break

        if peak is None:
            center = float(np.percentile(vals, 5.0))
        else:
            center = 0.5 * (edges[peak] + edges[peak + 1])

        band = (roi >= center - self.band_width_m) & (roi <= center + self.band_width_m) & valid
        if np.count_nonzero(band) < 30:
            cutoff = np.percentile(vals, 10.0)
            band = (roi <= cutoff) & valid
            if np.count_nonzero(band) < 10:
                band = valid

        # connected components to select main foreground
        band_u8 = (band.astype(np.uint8) * 255)
        num_labels, labels = cv2.connectedComponents(band_u8, connectivity=4)
        if num_labels > 1:
            seed_c = int(round((x1s + x2s) * 0.5)) - x1s
            seed_r = int(round((y1s + y2s) * 0.5)) - y1s
            seed_r = np.clip(seed_r, 0, roi.shape[0] - 1)
            seed_c = np.clip(seed_c, 0, roi.shape[1] - 1)
            seed_label = labels[seed_r, seed_c]

            if seed_label != 0 and np.any(labels == seed_label):
                comp = (labels == seed_label)
            else:
                best_label, best_med = 0, None
                for lb in range(1, num_labels):
                    m = np.median(roi[labels == lb])
                    if not np.isfinite(m):
                        continue
                    if (best_med is None) or (m < best_med):
                        best_med, best_label = m, lb
                comp = (labels == best_label) if best_label != 0 else (labels != 0)
        else:
            comp = band

        ys, xs = np.where(comp)
        if xs.size == 0:
            return None

        # weighted center inside foreground component
        u0, v0 = (x1s + x2s) * 0.5, (y1s + y2s) * 0.5
        dx, dy = (x1s + xs) - u0, (y1s + ys) - v0
        w_c = 1.0 / (1.0 + (dx * dx + dy * dy))
        dsel = roi[ys, xs]
        w_d = 1.0 / (1.0 + np.maximum(0.0, dsel - np.min(dsel)))
        w = w_c * w_d

        u = int(round(np.average(x1s + xs, weights=w)))
        v = int(round(np.average(y1s + ys, weights=w)))

        return u, v, roi, comp, dsel
    
    def _align_on_pixel(self, u: int, cx: float) -> bool:
        """
        Align robot yaw so that the detected pixel is centered in the image.

        Args:
            u (int): Pixel x-coordinate of the detected object.
            cx (float): Camera principal point x-coordinate.

        Returns:
            bool:
                True if alignment was handled this tick and further processing
                should stop, False otherwise.
        """
        err_px = u - cx
        if abs(err_px) <= 8:
            self.pub_cmd_vel.publish(Twist())
            self.state = self.S_ALIGNED
        else:
            tw = Twist()
            tw.angular.z = -0.05 if err_px > 0 else 0.05
            self.pub_cmd_vel.publish(tw)
        return True
    
    def _refine_depth_at_pick(self, depth, u, v, H, W, roi, comp, dsel):
        """
        Refine depth estimation around a selected foreground pixel.

        Args:
            depth (np.ndarray): Full depth image in meters.
            u (int): Pixel x-coordinate of selected point.
            v (int): Pixel y-coordinate of selected point.
            H (int): Image height.
            W (int): Image width.
            roi (np.ndarray): Depth ROI corresponding to the bounding box.
            comp (np.ndarray): Foreground component mask.
            dsel (np.ndarray): Depth values of selected foreground pixels.

        Returns:
            float | None:
                Refined depth value in meters, or None if refinement fails.
        """
        win = int(self.median_patch_px) if self.median_patch_px % 2 == 1 else int(self.median_patch_px) + 1
        r = win // 2
        x0, x3 = max(0, u - r), min(W, u + r + 1)
        y0, y3 = max(0, v - r), min(H, v + r + 1)
        patch = depth[y0:y3, x0:x3].copy()

        # rebuild full comp mask to crop to patch
        comp_full = np.zeros_like(depth, dtype=bool)
        comp_full[y0:y3, x0:x3] = True 
        patch_mask = comp_full[y0:y3, x0:x3]

        patch = patch[
            np.isfinite(patch)
            & (patch > self.depth_min)
            & (patch < self.depth_max)
            & patch_mask
        ]
        if patch.size > 20:
            Z = float(np.median(patch))
        else:
            Z = float(np.percentile(dsel, 5.0))

        return Z if np.isfinite(Z) else None
    
    def _project_to_base_and_map(self, rgb_msg, info_msg, u, v, Z, fx, fy, cx, cy):
        """
        Project a pixel with depth into base_link and target/map frame.

        Args:
            rgb_msg (Image): RGB image message (timestamp source).
            info_msg (CameraInfo): Camera intrinsic parameters.
            u (int): Pixel x-coordinate.
            v (int): Pixel y-coordinate.
            Z (float): Depth value at (u,v) in meters.
            fx (float): Focal length in x-direction.
            fy (float): Focal length in y-direction.
            cx (float): Principal point x-coordinate.
            cy (float): Principal point y-coordinate.

        Returns:
            tuple | None:
                (p_base, p_map) as PointStamped, or None if TF transform fails.
        """
        # 2D -> 3D in camera frame
        Xc = (u - cx) * Z / fx
        Yc = (v - cy) * Z / fy

        # camera -> base_link
        try:
            p_cam = PointStamped()
            p_cam.header = rgb_msg.header
            p_cam.header.frame_id = info_msg.header.frame_id or self.camera_frame
            p_cam.point.x, p_cam.point.y, p_cam.point.z = Xc, Yc, Z

            T_cb = self._lookup_transform_tolerant(
                self.base_frame, p_cam.header.frame_id, rgb_msg.header.stamp
            )
            p_cam.header.stamp = T_cb.header.stamp
            p_base = tf2_geometry_msgs.do_transform_point(p_cam, T_cb)
        except Exception as e:
            rospy.logwarn_throttle(2.0, "TF to %s failed: %s", self.base_frame, e)
            return None

        # base_link -> target_frame (map)
        try:
            T_bt = self._lookup_transform_tolerant(
                self.target_frame, self.base_frame, rgb_msg.header.stamp
            )
            p_base.header.stamp = T_bt.header.stamp
            p_map = tf2_geometry_msgs.do_transform_point(p_base, T_bt)
        except Exception:
            p_map = None

        return p_base, p_map
    
    def _publish_object_pose(self, rgb_msg, p_map):
        """
        Publish the detected object pose for navigation purposes.

        Args:
            rgb_msg (Image): RGB image message (used for timestamp).
            p_map (PointStamped): Object position in target frame (e.g. map).
        """
        if p_map is None:
            return

        pose = PoseStamped()
        pose.header.stamp = rgb_msg.header.stamp
        pose.header.frame_id = self.target_frame
        pose.pose.position.x = p_map.point.x
        pose.pose.position.y = p_map.point.y
        pose.pose.position.z = 0.0

        try:
            T_bl = self.tfbuf.lookup_transform(
                self.target_frame, self.base_frame,
                rgb_msg.header.stamp, rospy.Duration(0.1)
            )
            rx, ry = T_bl.transform.translation.x, T_bl.transform.translation.y
            yaw = math.atan2(p_map.point.y - ry, p_map.point.x - rx)
        except Exception:
            yaw = 0.0

        pose.pose.orientation = Quaternion(*quaternion_from_euler(0, 0, yaw))
        self.pub_pose.publish(pose)

    def _update_detection_state(self, rgb_msg, depth_msg, info_msg,
                            x1, y1, x2, y2, u, v, Z, p_base, p_map):
        """
        Update detection state, publish object status, and advance FSM transitions.

        Args:
            rgb_msg (Image): RGB image message.
            depth_msg (Image): Depth image message.
            info_msg (CameraInfo): Camera intrinsics message.
            x1, y1, x2, y2 (int): Bounding box pixel coordinates.
            u (int): Selected foreground pixel x-coordinate.
            v (int): Selected foreground pixel y-coordinate.
            Z (float): Estimated depth at (u,v) in meters.
            p_base (PointStamped): Object position in base_link frame.
            p_map (PointStamped | None): Object position in target/map frame.
        """
        self.last_detection = dict(
            rgb_msg=rgb_msg, depth_msg=depth_msg, info_msg=info_msg,
            bbox=(x1, y1, x2, y2),
            pick_uv=(u, v),
            Z=Z,
            point_base=p_base,
            point_map=p_map,
        )
        self.last_seen_ts = time.time()
        self.hits_in_row += 1
        found_now = self.hits_in_row >= self.min_hits
        self.pub_found.publish(Bool(data=found_now))

        if self.state == self.S_SEARCHING and found_now:
            self.state = self.S_APPROACHING

        elif self.state == self.S_ALIGNED:
            # either start close-in or directly grasp
            if self._start_close_in():
                return
            self._stop_and_save_traj()
            self._compute_grasp(rgb_msg, depth_msg, info_msg)

    # ---------------- Grasp computation ----------------
    
    def _compute_grasp(self, rgb_msg, depth_msg, info_msg):
        """
        Compute and execute a grasp based on a close-range RGB-D observation.

        Args:
            rgb_msg (Image): RGB image message.
            depth_msg (Image): Depth image message.
            info_msg (CameraInfo): Camera intrinsic parameters.

        Returns:
            bool:
                True if the grasp procedure was executed successfully,
                False otherwise.
        """
        # Reuse common frame prep (wie in tick)
        frame = self._prepare_frame(rgb_msg, depth_msg, info_msg)
        if frame is None:
            return False
        _, depth, H, W, fx, fy, cx, cy, rgb_pil = frame

        # Detection at grasp time
        det = self._run_detector(rgb_pil, W, H)
        if det is None:
            rospy.logwarn("No detection at grasp time")
            return False
        x1, y1, x2, y2, _score = det

        # Build depth band in ROI
        band_points = self._build_grasp_band(depth, x1, y1, x2, y2, W, H)
        if band_points is None:
            rospy.logwarn("no valid depth band @grasp")
            return False
        us, vs, Zs = band_points

        # Project all band points to base_link
        base_points = self._band_points_to_base(us, vs, Zs, info_msg, depth_msg, fx, fy, cx, cy)
        if base_points is None:
            return False
        xs_b, ys_b, zs_b = base_points

        # Select top point and map to arm coordinates
        top = self._top_point_from_band(xs_b, ys_b, zs_b)
        if top is None:
            return False
        x_top, y_top, z_top = top

        x_mm = x_top * 1000.0
        y_mm = y_top * 1000.0
        z_mm = z_top * 1000.0

        # base_link -> arm frame mapping
        X_arm =  y_mm
        Y_arm = -x_mm
        Z_arm =  z_mm

        rospy.loginfo(
            "GRASP base_link[m]=(%.3f,%.3f,%.3f) -> arm[mm]=(%.1f,%.1f,%.1f)",
            x_top, y_top, z_top, X_arm, Y_arm, Z_arm
        )
        return self.do_grasp(X_arm, Y_arm, Z_arm)
    
    def _build_grasp_band(self, depth, x1, y1, x2, y2, W, H):
        """
        Build a foreground depth band inside a detection bounding box.

        Args:
            depth (np.ndarray): Depth image in meters.
            x1 (int): Left bounding box pixel.
            y1 (int): Top bounding box pixel.
            x2 (int): Right bounding box pixel.
            y2 (int): Bottom bounding box pixel.
            W (int): Image width in pixels.
            H (int): Image height in pixels.

        Returns:
            tuple | None:
                (us, vs, Zs) pixel coordinates and depths of band points,
                or None if no valid foreground band can be formed.
        """
        s = int(self.bbox_shrink_px)
        x1s, y1s = max(x1 + s, 0), max(y1 + s, 0)
        x2s = max(min(x2 - s, W - 1), x1s + 1)
        y2s = max(min(y2 - s, H - 1), y1s + 1)
        if x2s <= x1s or y2s <= y1s:
            x1s, y1s, x2s, y2s = x1, y1, x2, y2

        roi = depth[y1s:y2s, x1s:x2s]
        valid = np.isfinite(roi) & (roi > self.depth_min) & (roi < self.depth_max)
        if not np.any(valid):
            rospy.logwarn("no depth valid @grasp")
            return None

        vals = roi[valid]
        near = float(np.percentile(vals, self.near_percentile))
        band = (roi >= near - self.band_width_m) & (roi <= near + self.band_width_m) & valid
        if np.count_nonzero(band) < 30:
            band = valid

        ys, xs = np.where(band)
        if xs.size == 0:
            rospy.logwarn("no band pixels @grasp")
            return None

        us = (x1s + xs).astype(np.float32)
        vs = (y1s + ys).astype(np.float32)
        Zs = roi[ys, xs].astype(np.float32)
        return us, vs, Zs
    
    def _band_points_to_base(self, us, vs, Zs, info_msg, depth_msg, fx, fy, cx, cy):
        """
        Project foreground pixel coordinates from the camera frame into base_link.

        Args:
            us (np.ndarray): Pixel x-coordinates of band points.
            vs (np.ndarray): Pixel y-coordinates of band points.
            Zs (np.ndarray): Depth values corresponding to (u,v) in meters.
            info_msg (CameraInfo): Camera intrinsics message.
            depth_msg (Image): Depth image ROS message (for timestamp and frame).
            fx (float): Focal length in x-direction.
            fy (float): Focal length in y-direction.
            cx (float): Principal point x-coordinate.
            cy (float): Principal point y-coordinate.

        Returns:
            tuple | None:
                (xs_b, ys_b, zs_b) NumPy arrays in base_link frame [m],
                or None if TF lookup or projection fails.
        """
        try:
            src_frame = info_msg.header.frame_id or self.camera_frame
            T_cb = self._lookup_transform_tolerant(
                self.base_frame, src_frame, depth_msg.header.stamp
            )
        except Exception as e:
            rospy.logwarn("TF for grasp failed: %s", e)
            return None

        xs_b, ys_b, zs_b = [], [], []
        for u, v, Z in zip(us, vs, Zs):
            Xc = (u - cx) * Z / fx
            Yc = (v - cy) * Z / fy

            p = PointStamped()
            p.header = depth_msg.header
            p.header.frame_id = src_frame
            p.point.x, p.point.y, p.point.z = float(Xc), float(Yc), float(Z)
            pb = tf2_geometry_msgs.do_transform_point(p, T_cb).point

            xs_b.append(pb.x)
            ys_b.append(pb.y)
            zs_b.append(pb.z)

        return (
            np.asarray(xs_b, dtype=np.float32),
            np.asarray(ys_b, dtype=np.float32),
            np.asarray(zs_b, dtype=np.float32),
        )
    
    def _top_point_from_band(self, xs_b, ys_b, zs_b):
        """
        Compute the grasp point in base_link from band points.

        Args:
            xs_b (np.ndarray): X coordinates of foreground points in base_link [m].
            ys_b (np.ndarray): Y coordinates of foreground points in base_link [m].
            zs_b (np.ndarray): Z coordinates of foreground points in base_link [m].

        Returns:
            tuple | None:
                (x_top, y_top, z_top) in meters, or None if not enough valid points exist.
        """
        if zs_b.size < 5:
            rospy.logwarn("too few points @grasp")
            return None

        z_top = np.percentile(zs_b, self.top_percentile)
        top_mask = zs_b >= z_top
        if np.count_nonzero(top_mask) < 5:
            top_mask = zs_b >= (z_top - 0.005)  # 5 mm tolerance

        x_top = float(np.median(xs_b[top_mask]))
        y_top = float(np.median(ys_b[top_mask]))
        z_top = float(np.median(zs_b[top_mask])) + float(self.grasp_z_lift)
        return x_top, y_top, z_top
    
    def do_grasp(self, X_arm_mm, Y_arm_mm, Z_arm_mm):
        """
        Execute a simple grasp with the MyCobot arm.

        The function moves the arm to a pre-grasp configuration, approaches the
        target position from the front, closes the gripper, and finally lifts the object.
        All coordinates are specified in the arm coordinate system (millimeters).

        Args:
            X_arm_mm (float): Target X position in arm frame [mm].
            Y_arm_mm (float): Target Y position in arm frame [mm].
            Z_arm_mm (float): Target Z position in arm frame [mm].

        Returns:
            bool: True if the grasp sequence was executed successfully,
                False if an error occurred during execution.
        """

        #if not (0 <= Z_arm_mm <= 400 and abs(X_arm_mm) <= 300 and abs(Y_arm_mm) <= 320):
        #    rospy.logwarn("Grasp target seems out of reach: (%.1f,%.1f,%.1f) mm", X_arm_mm, Y_arm_mm, Z_arm_mm)
        #    return False

        try:
            self.mc.send_angles([-85.37, -45.8, -104.2, 125.0, 3.0, 50.4], 50)

            self.mc.set_gripper_state(0, 80)

            rx, ry, rz = -110, 45, 165
            speed = 40
            X_arm_mm = X_arm_mm + 5
            Y_arm_mm = Y_arm_mm + 50
            Z_arm_mm = Z_arm_mm + 10

            self.mc.send_coords([X_arm_mm, Y_arm_mm + 30, Z_arm_mm + 10, rx, ry, rz], speed)
            rospy.sleep(1.5)

            self.mc.send_coords([X_arm_mm, Y_arm_mm , Z_arm_mm, rx, ry, rz], speed)
            rospy.sleep(0.8)

            self.mc.set_gripper_state(1, 80)
            
            rospy.sleep(0.5)

            self.mc.send_angles([-77.0, -50.0, -40.0, 100.0, -5.0, 52], 50)
            rospy.sleep(0.8)
            return True
        
        except Exception as e:
            rospy.logerr("Grasp failed: %s", e)
            return False
    
    def _stop_and_save_traj(self):
        """
        Stop logging and write trajectory to text file.
        """
        if not self.traj_logging_active:
            return
        self.traj_logging_active = False

        try:
            with open(self.traj_log_path, "w") as f:
                f.write("# t_sec x y yaw_rad cum_dist_m\n")
                for t, x, y, yaw, dist in self.traj_log:
                    f.write(f"{t:.3f} {x:.6f} {y:.6f} {yaw:.6f} {dist:.6f}\n")
            rospy.loginfo("[object_finder] Trajectory log saved to %s (dist=%.3f m, N=%d)",
                          self.traj_log_path, self.traj_total_dist, len(self.traj_log))
        except Exception as e:
            rospy.logerr("[object_finder] Failed to save trajectory log: %s", e)

    def _start_close_in(self) -> bool:
        """
        Compute how much to move to reach min_standoff and launch async drive.
        """
        d = self._current_object_planar_distance()
        if d is None:
            return False
        delta = max(0.0, d - self.min_standoff)
        rospy.loginfo(f"[close_in] delta={delta:.3f}")
        if delta < 0.02:
            return False
        self._close_in_done = False
        self.state = self.S_CLOSING_IN
        return True
    
    def _current_object_planar_distance(self) -> float:
        """
        Compute planar distance from robot (base_link) to the last detected object
        using the object position stored in the map frame.
        """
        if not self.last_detection:
            return None

        p_map = self.last_detection.get("point_map", None)
        if p_map is None:
            return None

        try:
            # Transform object pose from map (p_map.header.frame_id) to base_link
            T_mb = self._lookup_transform_tolerant(
                self.base_frame,
                p_map.header.frame_id,
                rospy.Time(0),
            )
            p_base_now = tf2_geometry_msgs.do_transform_point(p_map, T_mb)
            pb = p_base_now.point
            return math.hypot(pb.x, pb.y)
        except Exception as e:
            rospy.logwarn_throttle(
                2.0,
                "TF map->base_link failed in _current_object_planar_distance: %s",
                e,
            )
    
    def _move_forward(self):
        """
        Step controller for S_CLOSING_IN.
        Publishes forward /cmd_vel while the planar distance to the object
        is greater than self.min_standoff. When reached, stops and switches
        to S_ARRIVED. Call this once per tick while in S_CLOSING_IN.
        """
        if self.last_detection is None:
            self.pub_cmd_vel.publish(Twist())
            return

        d = self._current_object_planar_distance()
        rospy.loginfo(f"[close_in] d={d:.3f}")
        if d is None:
            self.pub_cmd_vel.publish(Twist())
            return

        if d > float(self.min_standoff):
            tw = Twist()
            tw.linear.x = float(self.close_in_speed)  # forward
            self.pub_cmd_vel.publish(tw)
        else:
            # Reached target distance: stop and mark done
            self.pub_cmd_vel.publish(Twist())
            self._close_in_done = True
            self.state = self.S_ARRIVED

    # ---------------- Utilities ----------------

    def _no_detection(self):
        """
        Handle missing detections.
        """
        if time.time() - self.last_seen_ts > self.max_age:
            self.hits_in_row = 0
            self.pub_found.publish(Bool(data=False))

    def _publish_debug(self, rgb, box, score, pick=None):
        """
        Publish a debug visualization image.

        Draws the detection bounding box, confidence score, and selected target
        pixel on the RGB image and publishes it on the debug image topic.

        Args:
            rgb (np.ndarray): RGB image as OpenCV array (H×W×3, BGR).
            box (tuple | None): Bounding box (x1, y1, x2, y2) in pixel coordinates,
                                or None if no box should be drawn.
            score (float | None): Detection confidence score to overlay as text.
            pick (tuple | None): Selected pixel (u, v) to be marked, or None.
        """
        if not self.publish_debug: return
        try:
            img = rgb.copy()
            if box is not None:
                x1,y1,x2,y2 = box
                cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,255), 2)
                if score is not None:
                    cv2.putText(img, f"{self.query} {score:.2f}", (x1, max(0,y1-6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
            if pick is not None:
                cv2.drawMarker(img, pick, (0,180,255), cv2.MARKER_CROSS, 20, 2)
            self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(img, encoding="bgr8"))
        except Exception:
            pass
    
    def _lookup_transform_tolerant(self, target_frame: str, source_frame: str, 
                                   stamp: rospy.Time, timeout: float = 0.3):
        """
        Try exact timestamp; on Extrapolation/Lookup fallback to latest (Time(0)).
        
        Args:
            target_frame (str): Target coordinate frame.
            source_frame (str): Source coordinate frame.
            stamp (rospy.Time): Desired transform timestamp.
            timeout (float): Maximum time in seconds to wait for TF data.

        Returns:
            geometry_msgs.msg.TransformStamped:
                The requested transform, either at the exact timestamp or the
                most recent available one.
        """
        try:
            return self.tfbuf.lookup_transform(target_frame, source_frame, stamp, rospy.Duration(timeout))
        except (tf2_ros.ExtrapolationException, tf2_ros.LookupException, tf2_ros.ConnectivityException) as e:
            rospy.logwarn_throttle(2.0, "TF exact lookup failed (%s). Falling back to latest.", type(e).__name__)
            return self.tfbuf.lookup_transform(target_frame, source_frame, rospy.Time(0), rospy.Duration(timeout))


if __name__ == "__main__":
    ObjectFinder()
    rospy.spin()