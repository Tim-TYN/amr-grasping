#!/usr/bin/env python3


import math
import numpy as np
import rospy
import tf2_ros
import tf2_geometry_msgs

from nav_msgs.msg import OccupancyGrid
from map_msgs.msg import OccupancyGridUpdate
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion, Twist
from std_msgs.msg import Bool
from nav_msgs.srv import GetPlan, GetPlanRequest

import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus


class FrontierPlanner:
    """
    Frontier-based exploration planner for a mobile robot.
    Uses a camera coverage map and a global costmap to select navigation goals,
    exploring the environment until a target object is detected and then switching to object-approach mode.
    """
    def __init__(self):
        """
        Loads parameters and internal state, sets up ROS communication, TF and move_base interfaces,
        and starts the periodic planning timer.
        """
        self._load_params()
        self._init_state()
        self._init_ros_comm()

        # TF
        self.tfbuf = tf2_ros.Buffer()
        self.tfl = tf2_ros.TransformListener(self.tfbuf)

        # move_base Action
        self.mb = actionlib.SimpleActionClient("/move_base", MoveBaseAction)
        self.mb.wait_for_server()
        rospy.loginfo("[frontier_selector] Connected to /move_base")

        # move_base plan service
        rospy.wait_for_service("/move_base/GlobalPlanner/make_plan")
        self.make_plan = rospy.ServiceProxy("/move_base/GlobalPlanner/make_plan", GetPlan)
        rospy.loginfo("[frontier_selector] /move_base/make_plan is ready")

        # Timer
        self.timer = rospy.Timer(rospy.Duration(1.0), self.tick)

    def _load_params(self):
        """
        Load all ROS parameters.
        """
        self.cov_topic = rospy.get_param("~coverage_topic", "/cam_coverage")
        self.goal_topic = rospy.get_param("~goal_topic", "/move_base_simple/goal")
        self.global_costmap_topic = rospy.get_param("~global_costmap_topic", "/move_base/global_costmap/costmap")

        self.goal_reached_radius = float(rospy.get_param("~goal_reached_radius", 0.5))
        self.min_frontier_spacing_m = float(rospy.get_param("~min_frontier_spacing_m", 0.1))
        self.min_frontier_dist_m = float(rospy.get_param("~min_frontier_dist_m", 0.3))
        self.cost_free_threshold = int(rospy.get_param("~cost_free_threshold", 10))
        self.plan_tolerance = float(rospy.get_param("~plan_tolerance", 0.3))
        self.max_plan_tries = int(rospy.get_param("~max_plan_tries", 100))
        self.strategy = str(rospy.get_param("~strategy", "clossest"))
        self.frame_robot = rospy.get_param("~frame_robot", "base_link")
        self.standoff = float(rospy.get_param("~standoff", 0.45))
        self.standoff_arc_deg = float(rospy.get_param("~standoff_arc_deg", 10))
        self.standoff_max_deg = float(rospy.get_param("~standoff_max_deg", 100))
        self.safety_radius_m = float(rospy.get_param("~safety_radius_m", 0.2))
        self.max_candidates_eval = int(rospy.get_param("~max_candidates_eval", 100))

    def _init_state(self):
        """
        Initialize internal state variables.
        """
        self.costmap_msg = None       # type: OccupancyGrid | None
        self.cov_msg = None           # type: OccupancyGrid | None
        self.current_goal = None      # type: PoseStamped | None
        self.object_pose = None       # type: PoseStamped | None
        self.object_found = False
        self.detector_ready = False
        self.initial_spin_done = False
        self.approached = False

        self.plan_fail_count = 0
        self.last_goal_distance = None
        self.last_progress_time = rospy.Time.now()

    def _init_ros_comm(self):
        """
        Initialize publishers, subscribers and action client.
        """
        self.map_sub = rospy.Subscriber(self.global_costmap_topic, OccupancyGrid, self.costmap_cb, queue_size=1)
        self.costmap_update_sub = rospy.Subscriber("/move_base/global_costmap/costmap_updates", OccupancyGridUpdate,
                                                   self.costmap_update_cb, queue_size=10)
        self.cov_sub = rospy.Subscriber(self.cov_topic, OccupancyGrid, self.cov_cb, queue_size=1)
        self.found_sub = rospy.Subscriber("/object_found", Bool, self.found_cb, queue_size=1)
        self.object_pose_sub = rospy.Subscriber("/object_pose", PoseStamped, self.object_pose_cb, queue_size=1)
        self.ready_sub = rospy.Subscriber("/object_detection_ready", Bool, self.ready_cb, queue_size=1)

        self.goal_pub = rospy.Publisher(self.goal_topic, PoseStamped, queue_size=1)
        self.pub_cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.approached_pub = rospy.Publisher("/object_approached", Bool, queue_size=1)

    # ----------------- Callbacks -----------------
    def costmap_cb(self, msg: OccupancyGrid):
        """
        Initializes the global costmap and its internal NumPy representation.
        """
        if self.costmap_msg is None:
            self.costmap_msg = msg

    def costmap_update_cb(self, upd: OccupancyGridUpdate):
        """
        Callback for receiving *incremental updates* to the global costmap.
        The update message specifies a rectangular region of changed cells.  
        This function writes the updated block directly into the stored costmap.

        - If the update is out of bounds → logs a warning and ignores it.
        - Otherwise, updates only the affected sub-region for efficiency.
        """
        if self.costmap_msg is None:
            return

        cm = self.grid_to_array(self.costmap_msg)  # shape (H, W)

        H, W = cm.shape
        ux, uy = upd.x, upd.y
        uw, uh = upd.width, upd.height

        if ux < 0 or uy < 0 or ux + uw > W or uy + uh > H:
            rospy.logwarn("[frontier_selector] costmap update out of bounds: x=%d,y=%d,w=%d,h=%d", ux, uy, uw, uh)
            return

        block = np.asarray(upd.data, dtype=np.int8).reshape((uh, uw))
        cm[uy:uy+uh, ux:ux+uw] = block

        self.costmap_msg.data = cm.flatten().astype(np.int8).tolist()
        #self.publish_cm_raw_debug()


    def cov_cb(self, msg: OccupancyGrid):
        """
        Callback for receiving the camera coverage grid (/cam_coverage).
        Stores the most recent visibility map used to identify frontier cells.
        """
        self.cov_msg = msg

    def found_cb(self, msg: Bool):
        """
        Callback indicating that the perception system has detected the target object.
        Sets 'object_found' to True the first time this occurs.
        """
        if not self.object_found:
            self.object_found = bool(msg.data)

    def object_pose_cb(self, msg: PoseStamped):
        """
        Callback for receiving the estimated 6D pose of the detected object.
        The planner uses this to compute a standoff pose for grasping.
        """
        self.object_pose = msg

    def ready_cb(self, msg: Bool):
        """
        Callback indicating whether the object detection module is ready.
        Exploration does not start until this flag becomes True.
        """
        self.detector_ready = bool(msg.data)

    def _mb_done_cb(self, status, result):
        """
        Callback triggered when move_base finishes executing a navigation goal.

        - Logs the result (SUCCEEDED, ABORTED, PREEMPTED, etc.)  
        - Sets the 'approached' flag to True if the robot successfully reached the goal  
        - Publishes the 'object_approached' status to notify downstream components
        """
        txt = {
            GoalStatus.SUCCEEDED: "SUCCEEDED",
            GoalStatus.ABORTED: "ABORTED",
            GoalStatus.PREEMPTED: "PREEMPTED",
            GoalStatus.REJECTED: "REJECTED",
            GoalStatus.LOST: "LOST",
        }.get(status, str(status))
        rospy.loginfo(f"[frontier_selector] move_base finished: {txt}")

        reached = status == GoalStatus.SUCCEEDED
        self.approached = reached
        self.approached_pub.publish(Bool(data=reached))

    # ----------------- Main loop -----------------
    def tick(self, _evt):
        """
        Periodic main loop of the frontier planner.

        - Ensures all required data (costmap, coverage, detector readiness) is available.
        - If the object is detected, switches to object-approach mode.
        - Otherwise performs a one-time 360° spin to improve initial coverage.
        - Retrieves the robot pose from TF.
        - If a goal is active, checks whether it has been reached or become infeasible.
        - If no goal is active, selects the next frontier goal and publishes it.

        This function coordinates exploration until the object is found.
        """

        if not self._is_ready_for_tick():
            return
        
        if self.object_found:
            self._handle_object_mode()
            return

        if not self.initial_spin_done:
            self.initial_spin_done = self.spin_in_place_once(angular_speed=0.3)
            return
        
        # Get robot pose in map
        try:
            T = self.tfbuf.lookup_transform(self.costmap_msg.header.frame_id, self.frame_robot,
                                            rospy.Time(0), rospy.Duration(0.2))
        except Exception:
            return
        rx = T.transform.translation.x
        ry = T.transform.translation.y

        if self.current_goal is not None:
            if self._update_current_goal_status(rx, ry):
                # goal was dropped or reached, do not continue this cycle
                return

        # If no goal, compute one
        if self.current_goal is None:
            rospy.loginfo("[frontier_selector] Start compute Goal")
            goal = self.select_next_goal((rx, ry))
            rospy.loginfo(f"[frontier_selector] Goal computed {goal}")
            if goal is not None and not self.object_found and not self.approached:
                self.current_goal = goal
                self.goal_pub.publish(goal)
                rospy.loginfo("[frontier_selector] New goal published at (%.2f, %.2f), yaw=%.1f°",
                              goal.pose.position.x, goal.pose.position.y, yaw_from_quat(goal.pose.orientation)*180.0/math.pi)

    def _is_ready_for_tick(self) -> bool:
        """
        Check whether the planner has all required data to perform one tick.

        Conditions checked:
        - Global costmap and camera coverage map are available.
        - Robot is not already in approach mode.
        - Object detector is active.
        - Costmap and coverage map share the same frame, resolution, origin, and size.

        Returns:
            True  → All inputs valid, planner may proceed.
            False → Missing data or inconsistent maps.
        """
        if self.costmap_msg is None or self.cov_msg is None or self.approached or not self.detector_ready:
            return False
        
        if self.costmap_msg.header.frame_id != self.cov_msg.header.frame_id:
            rospy.logwarn_throttle(
                5.0,
                "[frontier_selector] Frame mismatch costmap='%s' vs coverage='%s'",
                self.costmap_msg.header.frame_id, self.cov_msg.header.frame_id
            )
            return False
        
        cm_info = self.costmap_msg.info
        cov_info = self.cov_msg.info

        if abs(cm_info.resolution - cov_info.resolution) > 1e-6:
            rospy.logwarn(
                "[frontier_selector] Resolution mismatch: costmap=%.4f, coverage=%.4f",
                cm_info.resolution, cov_info.resolution
            )

        if (abs(cm_info.origin.position.x - cov_info.origin.position.x) > 1e-3 or
            abs(cm_info.origin.position.y - cov_info.origin.position.y) > 1e-3):
            rospy.logwarn(
                "[frontier_selector] Origin mismatch: costmap=(%.3f, %.3f), coverage=(%.3f, %.3f)",
                cm_info.origin.position.x, cm_info.origin.position.y,
                cov_info.origin.position.x, cov_info.origin.position.y
            )

        if (cm_info.width != cov_info.width) or (cm_info.height != cov_info.height):
            rospy.logwarn(
                "[frontier_selector] Size mismatch: costmap=(%d x %d), coverage=(%d x %d)",
                cm_info.width, cm_info.height, cov_info.width, cov_info.height
            )
        return True
    
    def _handle_object_mode(self):
        """
        Handle behavior after the object has been detected.

        - Computes a safe standoff pose around the detected object.
        - Sends this pose as a navigation goal to move_base.
        - If no safe pose exists (e.g., blocked or unreachable), logs a warning.
        
        Executed only when object detection is active and exploration should stop.
        """
        fixed_frame = self.costmap_msg.header.frame_id
        ps = self._pick_standoff_goal(self.object_pose, fixed_frame)
        rospy.loginfo(f"[frontier_selector] Standoff goal: {ps}")
        if ps is not None and not self.approached:
            goal = MoveBaseGoal()
            goal.target_pose = ps
            self.mb.send_goal(goal, done_cb=self._mb_done_cb)
        else:
            rospy.logwarn_throttle(2.0, "[frontier_selector] No safe standoff pose found around object.")
        
    def _update_current_goal_status(self, rx: float, ry: float) -> bool:
        """
        Check whether the current goal has been reached, timed out, or become infeasible.

        Args:
            rx (float): Robot x-position in the global frame [m].
            ry (float): Robot y-position in the global frame [m].

        Returns:
            bool:
                True  - if the current goal was cleared (reached or dropped),
                False - if the goal remains active
        """
        gx = self.current_goal.pose.position.x
        gy = self.current_goal.pose.position.y
        dist = math.hypot(gx - rx, gy - ry)

        # Reached?
        if dist <= self.goal_reached_radius:
            rospy.loginfo("[frontier_selector] Goal reached (within radius). Selecting next...")
            self.current_goal = None
            self.plan_fail_count = 0
            self.last_goal_distance = None
            return True

        infeasible = False
        now = rospy.Time.now()

        # No progress
        if self.last_goal_distance is None or dist < self.last_goal_distance - 0.05:
            self.last_goal_distance = dist
            self.last_progress_time = now
        if (now - self.last_progress_time).to_sec() > 8.0:
            rospy.logwarn("[frontier_selector] Goal timed out (no progress).")
            infeasible = True

        # Local region check
        if not self.is_goal_region_free(gx, gy, radius_m=0.05):
            rospy.logwarn("[frontier_selector] Goal region not free.")
            infeasible = True

        if infeasible:
            rospy.logwarn("[frontier_selector] Current goal became infeasible. Dropping and selecting next...")
            self.current_goal = None
            self.plan_fail_count = 0
            self.last_goal_distance = None
            return True

        return False
    
    def spin_in_place_once(self, angular_speed: float = 0.3, eps: float = 0.05) -> bool:
        """
        Perform a single 360° rotation of the robot in place.

        Args:
            angular_speed (float): Angular velocity in rad/s used for the rotation.
            eps (float): Tolerance on the remaining angle to consider the spin complete.

        Returns:
            bool:
                True if the full rotation was completed,
                False if the spin was aborted early due to object detection or errors.
        """
        rate = rospy.Rate(20)
        twist = Twist()
        twist.angular.z = angular_speed

        # pick fixed frame
        fixed_frame = self.costmap_msg.header.frame_id if self.costmap_msg else "map"

        completed = False
        try:
            T0 = self.tfbuf.lookup_transform(fixed_frame, self.frame_robot,
                                            rospy.Time(0), rospy.Duration(0.5))
            last_yaw = yaw_from_quat(T0.transform.rotation)
            cum_abs = 0.0

            rospy.loginfo("[frontier_selector] Initial spin: start (TF-integrated)")
            while (not rospy.is_shutdown()) and (cum_abs < (2.0 * math.pi - eps)):
                if self.object_found:
                    rospy.loginfo("[frontier_selector] Initial spin: aborted (object_found=True)")
                    break
                self.pub_cmd_vel.publish(twist)

                T = self.tfbuf.lookup_transform(fixed_frame, self.frame_robot,
                                                rospy.Time(0), rospy.Duration(0.2))
                yaw = yaw_from_quat(T.transform.rotation)
                dyaw = angle_diff(yaw, last_yaw)
                cum_abs += abs(dyaw)
                last_yaw = yaw

                rate.sleep()
            else:
                # loop ended without 'break' → completed full spin
                completed = True

        except Exception as ex:
            rospy.logwarn("[frontier_selector] Spin TF error: %s. Falling back to time-based.", str(ex))
            duration = (2.0 * math.pi) / max(abs(angular_speed), 1e-3)
            end_time = rospy.Time.now() + rospy.Duration(duration)
            while (not rospy.is_shutdown()) and (rospy.Time.now() < end_time):
                if self.object_found:
                    rospy.loginfo("[frontier_selector] Initial spin (time-based): aborted (object_found=True)")
                    break
                self.pub_cmd_vel.publish(twist)
                rate.sleep()
            else:
                completed = True

        # Clean stop
        stop = Twist()
        for _ in range(5):
            self.pub_cmd_vel.publish(stop)
            rate.sleep()

        if completed:
            rospy.loginfo("[frontier_selector] Initial spin: completed ~360°")
        return completed

        
    def select_next_goal(self, robot_xy):
        """
        Select the next exploration goal based on the camera coverage and global costmap.

        Frontier definition on /cam_coverage:
            - cov == 100  → seen by camera
            - at least one 8-neighbor with cov < 0 → unknown neighbor (frontier)
        Additional constraints (global costmap):
            - cell is known (cm >= 0)
            - cell is obstacle free (cm <= self.cost_free_threshold)

        Args:
            robot_xy (tuple[float, float]): Robot position (x, y) in the global frame [m].

        Returns:
            PoseStamped | None:
                A reachable frontier goal in the global frame, or None if no valid goal exists.
        """
        cm = self.grid_to_array(self.costmap_msg)
        cov = self.grid_to_array(self.cov_msg)
        if cm is None or cov is None:
            return None

        info = self.costmap_msg.info
        res  = info.resolution
        ox   = info.origin.position.x
        oy   = info.origin.position.y
        H, W = cm.shape

        # 1) Build basic masks
        free_cm_mask, seen_cov_mask, unknown_cov = self._build_frontier_masks(cm, cov)

        # 2) Detect frontier cells in coverage map
        cx_all, cy_all = self._detect_frontier_cells(seen_cov_mask, unknown_cov)
        if cx_all.size == 0:
            rospy.logwarn("[frontier_selector] No coverage frontiers found")
            return None

        # 3) Filter by costmap (known & free) + min distance to robot
        cx_all, cy_all, dists = self._filter_frontiers(
            cx_all, cy_all, free_cm_mask, robot_xy, ox, oy, res
        )
        if cx_all.size == 0:
            rospy.logwarn("[frontier_selector] No valid frontiers after filtering")
            return None

        # 4) Prepare integral image for clearance
        free_int, cell_r = self._build_clearance_integral(free_cm_mask, H, W, res)

        # 5) Thin frontiers and score candidates (clearance, distance)
        candidates = self._build_candidates(
            cx_all, cy_all, dists, free_int, cell_r, res
        )
        if not candidates:
            rospy.logwarn("[frontier_selector] No candidates after scoring")
            return None

        # 6) Try candidates in order and pick first reachable goal
        goal = self._pick_reachable_goal(candidates, unknown_cov, res)
        if goal is None:
            rospy.logwarn("[frontier_selector] No reachable frontier found")
        return goal
    
    def _build_frontier_masks(self, cm, cov):
        """
        Build basic masks for frontier detection.

        Args:
            cm (np.ndarray): Global costmap array (HxW).
            cov (np.ndarray): Coverage grid array (HxW) from /cam_coverage.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
                free_cm_mask  - True where costmap is known and free,
                seen_cov_mask - True where coverage is 100 (seen),
                unknown_cov   - True where coverage is unknown (< 0).
        """
        free_cm_mask  = (cm >= 0) & (cm <= self.cost_free_threshold)
        seen_cov_mask = (cov == 100)
        unknown_cov   = (cov < 0)  # typically cov == -1

        # Optional debug: distribution of coverage values
        #vals, counts = np.unique(cov, return_counts=True)
        #rospy.loginfo(f"[frontier_selector] cov unique: {dict(zip(vals.tolist(), counts.tolist()))}")

        return free_cm_mask, seen_cov_mask, unknown_cov


    def _detect_frontier_cells(self, seen_cov_mask, unknown_cov):
        """
        Detect frontier cells on the coverage grid.

        Args:
            seen_cov_mask (np.ndarray): Boolean mask of seen cells.
            unknown_cov (np.ndarray): Boolean mask of unknown cells.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                (cx_all, cy_all) arrays of x/y indices of frontier cells.
        """
        neigh_unknown_cov = count_unknown_neighbors(unknown_cov)
        frontier_cov_mask = seen_cov_mask & (neigh_unknown_cov > 0)

        cy_all, cx_all = np.where(frontier_cov_mask)
        rospy.loginfo(
            "[frontier_selector] coverage frontiers=%d (before costmap filter)",
            int(cy_all.size),
        )
        return cx_all, cy_all


    def _filter_frontiers(self, cx_all, cy_all, free_cm_mask, robot_xy, ox, oy, res):
        """
        Filter frontier cells by costmap free space and minimum distance to the robot.

        Args:
            cx_all (np.ndarray): Frontier x-indices in grid coordinates.
            cy_all (np.ndarray): Frontier y-indices in grid coordinates.
            free_cm_mask (np.ndarray): Boolean mask of free costmap cells.
            robot_xy (tuple[float, float]): Robot position (x, y) in world frame [m].
            ox (float): Map origin x-position [m].
            oy (float): Map origin y-position [m].
            res (float): Map resolution [m/cell].

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
                Filtered (cx_all, cy_all) and corresponding distances dists [m].
        """
        # 1) Only frontiers that are free in the costmap
        mask_free_cm = free_cm_mask[cy_all, cx_all]
        cx_all, cy_all = cx_all[mask_free_cm], cy_all[mask_free_cm]
        rospy.loginfo(
            "[frontier_selector] frontiers after costmap free filter=%d",
            int(cy_all.size),
        )
        if cy_all.size == 0:
            return cx_all, cy_all, np.array([])

        # 2) Compute world coordinates + distance to robot
        xs = ox + (cx_all.astype(np.float32) + 0.5) * res
        ys = oy + (cy_all.astype(np.float32) + 0.5) * res
        rx, ry = robot_xy
        dists = np.hypot(xs - rx, ys - ry)

        # 3) Enforce minimum distance
        keep = dists >= self.min_frontier_dist_m
        rospy.loginfo(
            "[frontier_selector] after min_dist: %d candidates",
            int(np.count_nonzero(keep)),
        )
        if not np.any(keep):
            rospy.logwarn("[frontier_selector] Frontiers only outside distance window")
            return np.array([]), np.array([]), np.array([])

        cx_all = cx_all[keep]
        cy_all = cy_all[keep]
        dists  = dists[keep]
        return cx_all, cy_all, dists


    def _build_clearance_integral(self, free_cm_mask, H, W, res):
        """
        Build an integral image over free cells and compute the clearance radius in cells.

        Args:
            free_cm_mask (np.ndarray): Boolean mask of free costmap cells.
            H (int): Grid height.
            W (int): Grid width.
            res (float): Map resolution [m/cell].

        Returns:
            tuple[np.ndarray, int]:
                free_int - integral image of free cells (shape (H+1, W+1)),
                cell_r   - clearance radius in cells.
        """
        free_uint = free_cm_mask.astype(np.uint8)
        free_int = np.zeros((H + 1, W + 1), dtype=np.int32)
        free_int[1:, 1:] = np.cumsum(np.cumsum(free_uint, axis=0), axis=1)

        cell_r = max(1, int(self.safety_radius_m / max(res, 1e-6)))
        return free_int, cell_r


    def _build_candidates(self, cx_all, cy_all, dists, free_int, cell_r, res):
        """
        Build and score frontier candidates based on clearance and distance.

        Applies greedy thinning in grid space and computes a clearance score
        using an integral image.

        Args:
            cx_all (np.ndarray): Frontier x-indices in grid coordinates.
            cy_all (np.ndarray): Frontier y-indices in grid coordinates.
            dists (np.ndarray): Distances of frontier cells to the robot [m].
            free_int (np.ndarray): Integral image of free cells.
            cell_r (int): Clearance radius in cells.
            res (float): Map resolution [m/cell].

        Returns:
            list[tuple]:
                Sorted list of candidates as (clearance, distance, cy, cx),
                ordered by descending clearance and ascending distance.
        """
        if cx_all.size == 0:
            return []

        H = self.costmap_msg.info.height
        W = self.costmap_msg.info.width

        # 1) Thinning in grid space (avoid too dense goals)
        min_spacing_cells = max(1, int(self.min_frontier_spacing_m / max(res, 1e-6)))
        selected = []            # list of (cy, cx, idx)
        selected_for_spacing = []  # list of (cy, cx) for distance checks

        # sort indices by distance (ascending) so closer frontiers are preferred
        order = np.argsort(dists)
        limit = min(int(self.max_candidates_eval), int(dists.size))

        for idx in order[:limit]:
            cy = int(cy_all[idx])
            cx = int(cx_all[idx])
            if not is_far_from_all((cy, cx), selected_for_spacing, min_spacing_cells):
                continue
            selected.append((cy, cx, idx))
            selected_for_spacing.append((cy, cx))

        if not selected:
            rospy.logwarn("[frontier_selector] No candidates after thinning")
            return []

        rospy.loginfo(f"[frontier_selector] selected frontier cells (pre-clearance): {selected}")

        # 2) Compute clearance for each selected candidate
        candidates = []  # list of (clearance, dist, cy, cx)
        for (cy, cx, idx) in selected:
            clearance = self._clearance_value(cx, cy, cell_r, free_int, W, H)
            dist = float(dists[idx])
            candidates.append((clearance, dist, cy, cx))

        if not candidates:
            return []

        # sort: first by clearance descending, then by distance ascending
        candidates.sort(key=lambda c: (-c[0], c[1]))
        rospy.loginfo(
            "[frontier_selector] candidates after clearance sort (clearance, dist, cy, cx): %s",
            str(candidates[:10])  # log only first few
        )
        return candidates


    def _pick_reachable_goal(self, candidates, unknown_cov, res):
        """
        Iterate over candidates and return the first reachable frontier goal.

        Args:
            candidates (list[tuple]): Candidate list (clearance, dist, cy, cx).
            unknown_cov (np.ndarray): Boolean mask of unknown coverage cells.
            res (float): Map resolution [m/cell].

        Returns:
            PoseStamped | None:
                A reachable goal PoseStamped in the costmap frame,
                or None if no candidate is reachable.
        """
        tried = 0
        for (clearance, dist, cy, cx) in candidates:
            pose, yaw = self.cell_to_goal_pose(cx, cy, res, unknown_cov)
            gx, gy = pose.position.x, pose.position.y

            # Make sure the goal is still within the costmap bounds
            mm = self.world_to_map(gx, gy, self.costmap_msg)
            if mm is None:
                rospy.loginfo(f"[frontier_selector] {cy}, {cx} maps out of costmap bounds")
                continue

            # Final expensive check: global planner must find a path
            if self.is_reachable(pose):
                rospy.loginfo(
                    "[frontier_selector] picked goal with clearance=%.3f, dist=%.2f at cell=(%d,%d)",
                    clearance, dist, cy, cx
                )
                return to_ps(pose, self.costmap_msg.header.frame_id)

            tried += 1
            if tried >= self.max_plan_tries:
                break

        rospy.logwarn("[frontier_selector] No reachable frontier among %d tried", max(0, tried))
        return None

    def publish_cm_raw_debug(self):
        """
        Publish the current global costmap (self.costmap_msg) 1:1 on a debug topic.
        """
        if self.costmap_msg is None:
            return

        msg = OccupancyGrid()
        msg.header = self.costmap_msg.header
        msg.info = self.costmap_msg.info
        msg.data = list(self.costmap_msg.data)

        if not hasattr(self, "debug_cm_pub"):
            self.debug_cm_pub = rospy.Publisher(
                "/debug/global_costmap_raw", OccupancyGrid, queue_size=1, latch=True
            )

        self.debug_cm_pub.publish(msg)

    def _pick_standoff_goal(self, object_ps: PoseStamped, fixed_frame: str):
        """
        Compute a reachable standoff navigation goal around a detected object.

        The method samples poses on a circular arc around the object at a fixed
        standoff distance and selects the first pose that is reachable according
        to the global planner.

        Args:
            object_ps (PoseStamped): Detected object pose.
            fixed_frame (str): Target frame for the standoff goal (usually map).

        Returns:
            PoseStamped | None:
                A feasible standoff goal pose in `fixed_frame`,
                or None if no safe and reachable pose is found.
        """
        obj_in_fixed = object_ps
        if object_ps is None or fixed_frame is None:
            return None
        
        # 1) Transform object pose into fixed_frame if necessary
        if object_ps.header.frame_id != fixed_frame:
            try:
                T = self.tfbuf.lookup_transform(fixed_frame, object_ps.header.frame_id,
                                                rospy.Time(0), rospy.Duration(0.3))
                obj_in_fixed = tf2_geometry_msgs.do_transform_pose(object_ps, T)
            except Exception as e:
                rospy.logwarn_throttle(2.0, "TF object pose to %s failed: %s", fixed_frame, str(e))
                return None

        ox = obj_in_fixed.pose.position.x
        oy = obj_in_fixed.pose.position.y
        base_yaw = yaw_from_quat(obj_in_fixed.pose.orientation)

        # 2) Generate arc offsets
        d_rad = math.radians(max(1e-3, self.standoff_arc_deg))
        max_steps = int(round(max(0.0, self.standoff_max_deg) / max(1e-3, self.standoff_arc_deg)))
        offsets = [0.0]
        for k in range(1, max_steps + 1):
            offsets.append(+k * d_rad)
            offsets.append(-k * d_rad)

        # 3) Evaluate standoff candidates
        for off in offsets:
            cang = base_yaw + off
            gx = ox - self.standoff * math.cos(cang)
            gy = oy - self.standoff * math.sin(cang)
            
            yaw_face = math.atan2(oy - gy, ox - gx)
            q = self._sanitize_quat(yaw_to_quat(yaw_face))

            goal_pose = Pose()
            goal_pose.position = Point(gx, gy, 0.0)
            goal_pose.orientation = q

            # 4) Check reachability through the global planner
            if not self.is_reachable(goal_pose):
                continue

            return to_ps(goal_pose, fixed_frame)
        
        return None


    # ----------------- Helpers -----------------

    def _sanitize_quat(self, q: Quaternion) -> Quaternion:
        """
        Normalize a quaternion and fall back to identity if it is near-degenerate.

        Args:
            q (Quaternion): Input quaternion (possibly unnormalized or invalid).

        Returns:
            Quaternion:
                Normalized quaternion with unit length. 
        """
        n = math.sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w)
        if not np.isfinite(n) or n < 1e-6:
            rospy.logwarn("Goal orientation invalid; using identity.")
            return Quaternion(0.0, 0.0, 0.0, 1.0)
        return Quaternion(q.x/n, q.y/n, q.z/n, q.w/n)

    def _clearance_value(self, mx: int, my: int, cell_r: int, free_int: np.ndarray, w: int, h: int) -> float:
        """
        Compute a local clearance score around a grid cell using an integral image.

        Args:
            mx (int): Cell x-index.
            my (int): Cell y-index.
            cell_r (int): Clearance radius in cells.
            free_int (np.ndarray): Integral image of free cells (H+1xW+1).
            w (int): Grid width.
            h (int): Grid height.

        Returns:
            float:
                Ratio of free cells among all known cells in the square window,
                in the range [0.0, 1.0]. 0.0 means no clearance or no known cells.
        """
        if self.costmap_msg is None:
            # If we cannot judge, treat as neutral clearance.
            return 0.0

        y0 = max(0, my - cell_r); y1 = min(h - 1, my + cell_r)
        x0 = max(0, mx - cell_r); x1 = min(w - 1, mx + cell_r)

        # free cells from integral image (integral image has shape (h+1, w+1))
        free = self._box_sum(free_int, y0, x0, y1 + 1, x1 + 1)

        data = self.costmap_msg.data
        tot_known = 0

        for iy in range(y0, y1 + 1):
            base = iy * w
            for ix in range(x0, x1 + 1):
                v = data[base + ix]
                if v >= 0:
                    tot_known += 1

        # No known cells → treat as zero clearance
        if tot_known == 0:
            return 0.0

        free_ratio = float(free) / float(tot_known)
        return free_ratio
    
    @staticmethod
    def _box_sum(ii: np.ndarray, y0: int, x0: int, y1: int, x1: int) -> int:
        """
        Compute the sum over a rectangular region using an integral image.

        Args:
            ii (np.ndarray): Integral image of shape (H+1, W+1).
            y0 (int): Top index (inclusive).
            x0 (int): Left index (inclusive).
            y1 (int): Bottom index (exclusive).
            x1 (int): Right index (exclusive).

        Returns:
            int:
                Sum of values in the rectangle [y0:y1, x0:x1].
        """
        return int(ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0])

    def grid_to_array(self, grid_msg: OccupancyGrid):
        """
        Convert an OccupancyGrid message into a 2D NumPy array.

        Args:
            grid_msg (OccupancyGrid): Grid to convert.

        Returns:
            np.ndarray:
                2D array of shape (height, width) with cell values.
        """
        data = np.asarray(grid_msg.data, dtype=np.int16)
        return data.reshape((grid_msg.info.height, grid_msg.info.width))

    def cell_to_goal_pose(self, cx, cy, res, unknown_mask):
        """
        Convert a grid cell into a world pose oriented toward unknown space.

        Args:
            cx (int): Cell x-index.
            cy (int): Cell y-index.
            res (float): Map resolution [m/cell].
            unknown_mask (np.ndarray): Boolean mask of unknown coverage cells.

        Returns:
            tuple[Pose, float]:
                pose - goal pose at the cell center in world coordinates,
                yaw  - yaw angle used for the pose orientation [rad].
        """
        ox = self.costmap_msg.info.origin.position.x
        oy = self.costmap_msg.info.origin.position.y
        x = ox + (cx + 0.5) * res
        y = oy + (cy + 0.5) * res

        yaw = face_unknown_yaw(cx, cy, unknown_mask)
        q = yaw_to_quat(yaw)
        pose = Pose(Point(x, y, 0.0), q)
        return pose, yaw
    
    def world_to_map(self, x: float, y: float, grid: OccupancyGrid):
        """
        Convert world coordinates into costmap grid indices.

        Args:
            x (float): World x-coordinate [m].
            y (float): World y-coordinate [m].

        Returns:
            tuple[int, int] | None:
                (mx, my) costmap indices if inside bounds, otherwise None.
        """
        info = grid.info
        res = info.resolution
        mx = int((x - info.origin.position.x) / res)
        my = int((y - info.origin.position.y) / res)
        if 0 <= mx < info.width and 0 <= my < info.height:
            return mx, my
        return None

    def is_goal_region_free(self, gx: float, gy: float, radius_m: float = 0.30) -> bool:
        """
        Check whether the area around a goal position is free in the global costmap.

        Args:
            gx (float): Goal x-position in the global frame [m].
            gy (float): Goal y-position in the global frame [m].
            radius_m (float): Radius around the goal to check [m].

        Returns:
            bool:
                True if the goal region is free, False otherwise.
        """
        if self.costmap_msg is None:
            return True  # can't judge → don't block
        info = self.costmap_msg.info
        res = info.resolution
        w, h = info.width, info.height
        data = self.costmap_msg.data

        mm = self.world_to_map(gx, gy, self.costmap_msg)
        if mm is None:
            return False
        mx, my = mm

        cell_r = max(1, int(radius_m / max(res, 1e-6)))
        for iy in range(max(0, my - cell_r), min(h, my + cell_r + 1)):
            for ix in range(max(0, mx - cell_r), min(w, mx + cell_r + 1)):
                if (ix - mx)**2 + (iy - my)**2 > cell_r**2:
                    continue
                v = data[iy * w + ix]
                if v < 0:  # unknown
                    return False
                if v > self.cost_free_threshold:  # occupied/costly
                    return False
        return True

    def is_reachable(self, goal_pose: Pose) -> bool:
        """
        Check whether a valid global path exists to a given goal pose.

        Args:
            goal_pose (Pose): Goal pose in the global frame.

        Returns:
            bool:
                True if a valid global plan exists,
                False if no plan is found or the service call fails.
        """
        try:
            T = self.tfbuf.lookup_transform(self.costmap_msg.header.frame_id, self.frame_robot,
                                            rospy.Time(0), rospy.Duration(0.2))
        except Exception:
            return False
        start = PoseStamped()
        start.header.frame_id = self.costmap_msg.header.frame_id
        start.pose.position.x = T.transform.translation.x
        start.pose.position.y = T.transform.translation.y
        start.pose.position.z = 0.0
        start.pose.orientation = Quaternion(*[0, 0, 0, 1])  # orientation is irrelevant for global plan

        goal = PoseStamped()
        goal.header.frame_id = self.costmap_msg.header.frame_id
        goal.pose = goal_pose

        req = GetPlanRequest()
        req.start = start
        req.goal = goal
        req.tolerance = self.plan_tolerance
        try:
            resp = self.make_plan(req)
            return len(resp.plan.poses) > 0
        except rospy.ServiceException:
            return False
        
    def _publish_frontier_markers(self, cy_all, cx_all):
        """
        Publish frontier cells as visualization markers in RViz.

        Args:
            cy_all (np.ndarray): Frontier y-indices in grid coordinates.
            cx_all (np.ndarray): Frontier x-indices in grid coordinates.
        """
        if self.cov_msg is None:
            return

        from visualization_msgs.msg import Marker, MarkerArray

        marker_array = MarkerArray()

        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        info = self.cov_msg.info 
        res = info.resolution
        ox = info.origin.position.x
        oy = info.origin.position.y
        header = self.cov_msg.header

        for i, (cy, cx) in enumerate(zip(cy_all, cx_all)):
            wx = ox + (cx + 0.5) * res
            wy = oy + (cy + 0.5) * res

            m = Marker()
            m.header = header
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = wx
            m.pose.position.y = wy
            m.pose.position.z = 0.05
            m.pose.orientation.w = 1.0

            m.scale.x = 0.12
            m.scale.y = 0.12
            m.scale.z = 0.12

            m.color.r = 0.0
            m.color.g = 0.7 
            m.color.b = 1.0 
            m.color.a = 1.0

            marker_array.markers.append(m)

        self.frontier_pub.publish(marker_array)


# ---------------- Utility functions ----------------

def count_unknown_neighbors(unknown_mask: np.ndarray) -> np.ndarray:
    """
    Count unknown cells in the 8-neighborhood for each cell in a grid.

    Args:
        unknown_mask (np.ndarray): Boolean mask of unknown cells.

    Returns:
        np.ndarray:
            Integer array of same shape giving the number of unknown neighbors
            in the 8-connected neighborhood.
    """
    H, W = unknown_mask.shape
    cnt = np.zeros_like(unknown_mask, dtype=np.int16)
    # 8 dirs (dy, dx)
    dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    for dy, dx in dirs:
        cnt_y0 = max(0, -dy)
        cnt_y1 = min(H, H - dy)
        cnt_x0 = max(0, -dx)
        cnt_x1 = min(W, W - dx)
        src = unknown_mask[cnt_y0:cnt_y1, cnt_x0:cnt_x1]
        cnt[cnt_y0+dy:cnt_y1+dy, cnt_x0+dx:cnt_x1+dx] += src.astype(np.int16)

    return cnt


def is_far_from_all(cell, chosen, min_spacing_cells: int) -> bool:
    """
    Check whether a candidate cell is sufficiently far from a set of chosen cells.

    Args:
        cell (tuple[int, int]): Candidate cell (cy, cx).
        chosen (list[tuple[int, int]]): List of already selected cells (cy, cx).
        min_spacing_cells (int): Minimum required spacing in grid cells.

    Returns:
        bool:
            True if the candidate is far enough from all chosen cells,
            False otherwise.
    """
    cy, cx = cell
    for (py, px) in chosen:
        if abs(py - cy) <= min_spacing_cells and abs(px - cx) <= min_spacing_cells:
            if (py - cy)**2 + (px - cx)**2 <= (min_spacing_cells**2):
                return False
    return True


def face_unknown_yaw(cx: int, cy: int, unknown_mask: np.ndarray) -> float:
    """
    Estimate a yaw angle pointing toward nearby unknown cells.

    Args:
        cx (int): Cell x-index.
        cy (int): Cell y-index.
        unknown_mask (np.ndarray): Boolean mask of unknown cells.

    Returns:
        float:
            Yaw angle in radians pointing roughly toward unknown space.
            If no unknown neighbors are found, returns 0.0.
    """
    H, W = unknown_mask.shape
    dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    vx = 0.0
    vy = 0.0
    for dy, dx in dirs:
        ny = cy + dy
        nx = cx + dx
        if 0 <= ny < H and 0 <= nx < W and unknown_mask[ny, nx]:
            # vector from cell to neighbor (unknown)
            vx += dx
            vy += dy
    if vx == 0.0 and vy == 0.0:
        # fallback: face along +x (arbitrary); using 0 rad
        return 0.0
    return math.atan2(vy, vx)


def angle_diff(a: float, b: float) -> float:
    """
    Compute the shortest signed angular difference between two angles.

    Args:
        a (float): First angle in radians.
        b (float): Second angle in radians.

    Returns:
        float:
            Difference a - b wrapped into the range [-pi, pi].
    """
    d = (a - b + math.pi) % (2.0 * math.pi) - math.pi
    return d


def yaw_to_quat(yaw: float) -> Quaternion:
    """
    Convert a yaw angle into a Z-axis rotation quaternion.

    Args:
        yaw (float): Yaw angle in radians.

    Returns:
        Quaternion:
            Quaternion representing a pure yaw rotation.
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    # yaw-only quaternion (z-axis)
    return Quaternion(x=0.0, y=0.0, z=sy, w=cy)


def yaw_from_quat(q: Quaternion) -> float:
    """
    Extract yaw angle from a planar quaternion.

    Assumes rotation only around the Z-axis (2D navigation).

    Args:
        q (Quaternion): Input quaternion.

    Returns:
        float:
            Yaw angle in radians.
    """
    return math.atan2(2.0*(q.w*q.z), 1 - 2.0*(q.z*q.z))


def to_ps(pose: Pose, frame_id: str) -> PoseStamped:
    """
    Wrap a Pose in a PoseStamped with current timestamp and given frame.

    Args:
        pose (Pose): Pose to wrap.
        frame_id (str): TF frame ID for the pose.

    Returns:
        PoseStamped:
            PoseStamped containing the given pose and metadata.
    """
    ps = PoseStamped()
    ps.header.stamp = rospy.Time.now()
    ps.header.frame_id = frame_id
    ps.pose = pose
    return ps


if __name__ == "__main__":
    rospy.init_node("frontier_goal_selector")
    node = FrontierPlanner()
    rospy.loginfo("[frontier_selector] Node ready")
    rospy.spin()