#!/usr/bin/env python3
import math, time
import numpy as np
import rospy
import tf2_ros
import tf2_geometry_msgs
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import PoseStamped, Quaternion, Twist, Pose, Point
from nav_msgs.msg import OccupancyGrid
from nav_msgs.srv import GetPlan, GetPlanRequest
from std_msgs.msg import Bool
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus
from nav_msgs.msg import OccupancyGrid
from map_msgs.msg import OccupancyGridUpdate

class StraightPlanner:
    """
    Drive straight until an obstacle is ahead; then pick a new heading that maximizes unseen area
    from /cam_coverage and drive straight again. Publishes /move_base_simple/goal.
    """

    def __init__(self):
        """
        Loads parameters, initializes internal state and ROS interfaces, and starts the periodic planning loop.
        """
        self._load_params()
        self._init_state()
        self._init_ros_interfaces()

        rospy.Timer(rospy.Duration(self.replan_period), self._tick)
        rospy.loginfo("StraightPlanner: ready.")

    def _load_params(self) -> None:
        """
        Load all ROS parameters.
        """
        # costmap / coverage / frames
        self.global_costmap_topic   = rospy.get_param("~global_costmap_topic", "/move_base/global_costmap/costmap")
        self.cov_topic              = rospy.get_param("~coverage_topic", "/cam_coverage")
        self.goal_topic             = rospy.get_param("~goal_topic", "/move_base_simple/goal")
        self.make_plan_srv_name     = rospy.get_param("~make_plan_srv", "/move_base/GlobalPlanner/make_plan")
        self.global_frame           = rospy.get_param("~global_frame", "map")
        self.base_frame             = rospy.get_param("~base_frame", "base_link")
        self.cost_free_threshold    = int(rospy.get_param("~cost_free_threshold", 50))

        # standoff around detected object
        self.standoff          = float(rospy.get_param("~standoff", 0.3))
        self.standoff_arc_deg  = float(rospy.get_param("~standoff_arc_deg", 10))
        self.standoff_max_deg  = float(rospy.get_param("~standoff_max_deg", 100))
        self.plan_tolerance    = float(rospy.get_param("~plan_tolerance", 0.3))

        # ray / extension geometry
        self.safe_margin       = rospy.get_param("~safe_margin", 0.40)        # stop this far before obstacle
        self.segment_max_len   = rospy.get_param("~segment_max_len", 3.0)
        self.min_forward_free  = rospy.get_param("~min_forward_free", 0.4)    # if less free -> choose new heading
        self.sample_step_m     = rospy.get_param("~sample_step", 0.05)        # ray marching step

        # heading selection
        self.heading_samples     = rospy.get_param("~heading_samples", 16)    # 360°/N
        self.eval_range          = rospy.get_param("~eval_range", 4.0)        # how far we "count" unseen
        self.min_gain_to_accept  = rospy.get_param("~min_gain_to_accept", 0)

        # main loop
        self.replan_period = rospy.get_param("~replan_period", 2.0)

    def _init_state(self) -> None:
        """
        Initialize all internal state variables.
        """
        # maps
        self.costmap_msg  = None
        self.costmap_info = None
        self.costmap_arr  = None

        self.cov          = None
        self.cov_info     = None
        self.cov_arr      = None

        # exploration state
        self.current_heading          = None
        self.last_goal                = None
        self.object_found             = False
        self.object_detection_ready   = False
        self.initial_spin_done        = False
        self.approached               = False
        self.object_pose              = None

    def _init_ros_interfaces(self) -> None:
        """
        Set up publishers, subscribers, TF, services, action client.
        """
        # publishers
        self.approached_pub = rospy.Publisher("/object_approached", Bool, queue_size=1)
        self.pub_cmd_vel    = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.goal_pub       = rospy.Publisher(self.goal_topic, PoseStamped, queue_size=1)

        # subscribers
        self.found_sub = rospy.Subscriber("/object_found", Bool, self.found_cb, queue_size=1)
        self.ready_sub = rospy.Subscriber("/object_detection_ready", Bool, self.ready_cb, queue_size=1)
        self.object_pose_sub = rospy.Subscriber("/object_pose", PoseStamped, self.object_pose_cb, queue_size=1)

        self.map_sub = rospy.Subscriber(
            self.global_costmap_topic, OccupancyGrid, self.costmap_cb, queue_size=1
        )
        self.costmap_update_sub = rospy.Subscriber(
            "/move_base/global_costmap/costmap_updates",
            OccupancyGridUpdate,
            self.costmap_update_cb,
            queue_size=10,
        )
        self.cov_sub = rospy.Subscriber(
            self.cov_topic, OccupancyGrid, self.cov_cb, queue_size=1
        )

        # TF buffer & listener
        self.tfbuf = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tflis = tf2_ros.TransformListener(self.tfbuf)

        # services
        rospy.wait_for_service(self.make_plan_srv_name)
        self.make_plan = rospy.ServiceProxy(self.make_plan_srv_name, GetPlan)

        # move_base action client
        self.mb = actionlib.SimpleActionClient("/move_base", MoveBaseAction)
        rospy.loginfo("Waiting for /move_base action server...")
        self.mb.wait_for_server()
        rospy.loginfo("Connected to /move_base")

    # ----------------- Callbacks -----------------

    def costmap_cb(self, msg: OccupancyGrid):
        """
        Initializes the global costmap and its internal NumPy representation.
        """
        if self.costmap_msg is None:
            self.costmap_msg = msg
            self.costmap_info = msg.info
            self.costmap_arr = np.asarray(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)

    def costmap_update_cb(self, upd: OccupancyGridUpdate):
        """
        Callback for receiving *incremental updates* to the global costmap.
        The update message specifies a rectangular region of changed cells.  
        This function writes the updated block directly into the stored costmap.

        - If the update is out of bounds → logs a warning and ignores it.
        - Otherwise, updates only the affected sub-region for efficiency.
        """
        if self.costmap_msg is None or self.costmap_arr is None:
            return

        cm = self.costmap_arr  # shape (H, W)

        H, W = cm.shape
        ux, uy = upd.x, upd.y
        uw, uh = upd.width, upd.height

        if ux < 0 or uy < 0 or ux + uw > W or uy + uh > H:
            rospy.logwarn("[planner] costmap update out of bounds: x=%d,y=%d,w=%d,h=%d", ux, uy, uw, uh)
            return

        block = np.asarray(upd.data, dtype=np.int8).reshape((uh, uw))
        cm[uy:uy+uh, ux:ux+uw] = block

        self.costmap_msg.data = cm.flatten().astype(np.int8).tolist()

    def cov_cb(self, msg):
        """
        Callback for receiving the camera coverage grid (/cam_coverage).
        Stores the most recent visibility map used to identify frontier cells.
        """
        self.cov = msg
        self.cov_info = msg.info
        self.cov_arr = np.asarray(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)

    def found_cb(self, msg):
        """
        Callback indicating that the perception system has detected the target object.
        Sets 'object_found' to True the first time this occurs.
        """
        if not self.object_found:
            self.object_found = bool(msg.data)

    def ready_cb(self, msg):
        """
        Callback indicating whether the object detection module is ready.
        Exploration does not start until this flag becomes True.
        """
        self.object_detection_ready = msg.data

    def object_pose_cb(self, msg: PoseStamped):
        """
        Callback for receiving the estimated 6D pose of the detected object.
        The planner uses this to compute a standoff pose for grasping.
        """
        self.object_pose = msg

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
        rospy.loginfo(f"move_base finished: {txt}")

        reached = status == GoalStatus.SUCCEEDED
        self.approached = reached
        self.approached_pub.publish(Bool(data=reached))

    # ----------------- Main loop -----------------
    def _tick(self, _evt):
        """
        Main planner loop executed periodically.
        Handles object-approach behavior, initial exploration spin, and 
        regular exploration goal generation and publishing.
        """
        if not self.object_detection_ready or self.approached:
            return

        if self.costmap_arr is None or self.cov_arr is None:
            rospy.loginfo_throttle(2.0, "waiting for global costmap & /cam_coverage ...")
            return

        # Object handling has priority
        if self.object_found:
            goal = self._pick_standoff_goal(self.object_pose, self.global_frame)
            rospy.loginfo(f"[frontier_selector] Goal: {goal}")
            if goal is not None and not self.approached:
                self._send_move_base_goal(goal)
            else:
                rospy.logwarn_throttle(2.0, "[frontier_selector] No safe standoff pose found around object.")
            return

        # Initial 360° scan
        if not self.initial_spin_done:
            self.initial_spin_done = self.spin_in_place_once(angular_speed=0.3)
            return

        # Get robot pose
        try:
            rx, ry, ryaw = self._pose()
        except Exception:
            rospy.loginfo_throttle(2.0, "pose unavailable")
            return

        goal = self._compute_exploration_goal(rx, ry, ryaw)
        if goal is None:
            return

        gx, gy, gyaw = goal
        self._publish_goal(gx, gy, gyaw)

        
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
        fixed_frame = "map"

        completed = False
        try:
            T0 = self.tfbuf.lookup_transform(self.global_frame, self.base_frame,
                                            rospy.Time(0), rospy.Duration(0.5))
            last_yaw = yaw_from_quat(T0.transform.rotation)
            cum_abs = 0.0

            rospy.loginfo("[frontier_selector] Initial spin: start (TF-integrated)")
            while (not rospy.is_shutdown()) and (cum_abs < (2.0 * math.pi - eps)):
                if self.object_found:
                    rospy.loginfo("[frontier_selector] Initial spin: aborted (object_found=True)")
                    break
                self.pub_cmd_vel.publish(twist)

                T = self.tfbuf.lookup_transform(self.global_frame, self.base_frame,
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
    
    def _compute_exploration_goal(self, rx: float, ry: float, ryaw: float):
        """
        Compute the next exploration goal based on free space and unseen coverage.

        Args:
            rx (float): Robot x-position in the global frame [m].
            ry (float): Robot y-position in the global frame [m].
            ryaw (float): Robot yaw angle in the global frame [rad].

        Returns:
            tuple | None:
                (gx, gy, yaw) as the next exploration goal in the global frame,
                or None if no suitable and reachable goal could be found.
        """
        # pick or keep heading
        heading = self.current_heading if self.current_heading is not None else ryaw

        free_dist = self._ray_free_distance(rx, ry, heading, self.segment_max_len)

        # If ahead is almost blocked, choose a new heading toward unseen
        if free_dist < self.min_forward_free:
            best = None
            for k in range(self.heading_samples):
                ang = -math.pi + (2.0 * math.pi) * k / self.heading_samples

                # Skip headings that immediately collide
                if self._ray_free_distance(rx, ry, ang, max(self.min_forward_free, 1.0)) < self.min_forward_free:
                    continue

                gain = self._coverage_gain_along(rx, ry, ang, self.eval_range)

                # Prefer roughly forward if gains equal
                forward_bias = 0.1 * abs(((ang - ryaw + math.pi) % (2.0 * math.pi)) - math.pi)
                score = gain - forward_bias

                if (best is None) or (score > best[0]):
                    best = (score, ang, gain)

            if best is None or best[2] < self.min_gain_to_accept:
                # fallback: align with current yaw and try forward again
                heading = ryaw
                free_dist = self._ray_free_distance(rx, ry, heading, self.segment_max_len)
            else:
                heading = best[1]
                free_dist = self._ray_free_distance(rx, ry, heading, self.segment_max_len)

        self.current_heading = heading

        # compute goal point along heading (before obstacle)
        goal_dist = max(self.min_forward_free, min(free_dist, self.segment_max_len))
        gx = rx + math.cos(heading) * goal_dist
        gy = ry + math.sin(heading) * goal_dist

        # only accept goal if path exists
        if self._plan_exists((rx, ry), (gx, gy), tol=self.plan_tolerance):
            # avoid spamming the same goal
            if (self.last_goal is None) or (math.hypot(self.last_goal[0] - gx, self.last_goal[1] - gy) > 0.2):
                return gx, gy, heading
            return None

        # try a shorter nudge
        rospy.loginfo_throttle(2.0, "no plan to segment endpoint; nudging shorter")
        short = max(0.1, self.min_forward_free)
        gx2 = rx + math.cos(heading) * short
        gy2 = ry + math.sin(heading) * short

        if self._plan_exists((rx, ry), (gx2, gy2), tol=self.plan_tolerance):
            return gx2, gy2, heading

        # force new heading next tick
        self.current_heading = None
        return None
    
    def _ray_free_distance(self, x0, y0, ang, max_dist):
        """
        Ray-march from a start position along a given heading in the global costmap.

        The ray is advanced in small steps until an occupied or unknown cell is
        encountered or the maximum distance is reached.

        Args:
            x0 (float): Start x-position in the global frame [m].
            y0 (float): Start y-position in the global frame [m].
            ang (float): Ray direction (heading) in radians.
            max_dist (float): Maximum ray length in meters.

        Returns:
            float:
                Distance in meters to the last free point along the ray.
        """
        if self.costmap_info is None or self.costmap_arr is None:
            return 0.0
        step = max(self.sample_step_m, self.costmap_info.resolution)
        d = 0.0
        while d < max_dist:
            x = x0 + math.cos(ang) * d
            y = y0 + math.sin(ang) * d
            if self._is_occupied(x, y):
                return max(0.0, d - self.safe_margin)
            d += step
        return max_dist

    def _coverage_gain_along(self, x0, y0, ang, rng):
        """
        Estimate exploration gain along a ray direction.

        Traverses the ray until an obstacle is encountered and counts how many
        previously unseen cells (from the coverage map) are observed.

        Args:
            x0 (float): Start x-position in the global frame [m].
            y0 (float): Start y-position in the global frame [m].
            ang (float): Ray direction (heading) in radians.
            rng (float): Maximum evaluation range in meters.

        Returns:
            int:
                Number of unseen cells encountered along the ray.
        """
        if (self.cov_info is None) or (self.costmap_info is None) or (self.costmap_arr is None):
            return 0

        res = self.costmap_info.resolution
        step = max(self.sample_step_m, res)
        unseen = 0
        d = 0.0
        while d < rng:
            x = x0 + math.cos(ang) * d
            y = y0 + math.sin(ang) * d

            # stop at obstacle from costmap
            if self._is_occupied(x, y):
                break

            # coverage in /cam_coverage (gleiche Weltkoordinaten)
            cell = self._world_to_cell(self.cov_info, x, y)
            if cell is not None:
                if self.cov_arr[cell[1], cell[0]] == -1:
                    unseen += 1
            d += step
        return unseen

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
        base_yaw = yaw_from_quat(obj_in_fixed.pose.orientation)  # Richtung Roboter->Objekt zum Zeitpunkt der Detektion

        # (2) Kandidatenwinkel: 0, +d, -d, +2d, -2d, ... bis max
        d_rad = math.radians(max(1e-3, self.standoff_arc_deg))
        max_steps = int(round(max(0.0, self.standoff_max_deg) / max(1e-3, self.standoff_arc_deg)))
        offsets = [0.0]
        for k in range(1, max_steps + 1):
            offsets.append(+k * d_rad)
            offsets.append(-k * d_rad)

        for off in offsets:
            # (3) Punkt auf dem Kreis
            cang = base_yaw + off
            gx = ox - self.standoff * math.cos(cang)
            gy = oy - self.standoff * math.sin(cang)

            # (4) Orientierung: vom Ziel aufs Objekt schauen
            yaw_face = math.atan2(oy - gy, ox - gx)
            q = self._sanitize_quat(yaw_to_quat(yaw_face))

            goal_pose = Pose()
            goal_pose.position = Point(gx, gy, 0.0)
            goal_pose.orientation = q

            # (5) Sicherheit + Erreichbarkeit
            if not self.is_goal_region_free(gx, gy, radius_m=0.05):
                continue
            if not self.is_reachable(goal_pose):
                continue

            return to_ps(goal_pose, fixed_frame)
        
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
        if self.costmap_info is None or self.costmap_arr is None:
            return True  # can't judge → don't block

        res = self.costmap_info.resolution
        h, w = self.costmap_arr.shape

        mm = self.world_to_map(gx, gy)
        if mm is None:
            return False
        mx, my = mm

        cell_r = max(1, int(radius_m / max(res, 1e-6)))
        for iy in range(max(0, my - cell_r), min(h, my + cell_r + 1)):
            for ix in range(max(0, mx - cell_r), min(w, mx + cell_r + 1)):
                if (ix - mx)**2 + (iy - my)**2 > cell_r**2:
                    continue
                v = self.costmap_arr[iy, ix]
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
            T = self.tfbuf.lookup_transform(self.global_frame, self.base_frame,
                                            rospy.Time(0), rospy.Duration(0.2))
        except Exception:
            return False
        start = PoseStamped()
        start.header.frame_id = self.global_frame
        start.pose.position.x = T.transform.translation.x
        start.pose.position.y = T.transform.translation.y
        start.pose.position.z = 0.0
        start.pose.orientation = Quaternion(*[0, 0, 0, 1])  # orientation is irrelevant for global plan

        goal = PoseStamped()
        goal.header.frame_id = self.global_frame
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
        
    def _plan_exists(self, start_xy, goal_xy, tol=0.05):
        """
        Check whether a valid global plan exists between two positions.

        Args:
            start_xy (tuple[float, float]): Start position (x, y) in the global frame [m].
            goal_xy (tuple[float, float]): Goal position (x, y) in the global frame [m].
            tol (float): Planning tolerance in meters.

        Returns:
            bool:
                True if a valid plan exists, False otherwise.
        """
        req = GetPlanRequest()
        req.start.header.frame_id = self.global_frame
        req.start.pose.orientation.w = 1.0
        req.start.pose.position.x, req.start.pose.position.y = start_xy
        req.goal.header.frame_id = self.global_frame
        req.goal.pose.orientation.w = 1.0
        req.goal.pose.position.x, req.goal.pose.position.y = goal_xy
        req.tolerance = tol
        try:
            resp = self.make_plan(req)
            return len(resp.plan.poses) > 0
        except Exception as e:
            rospy.logwarn_throttle(2.0, "make_plan failed: %s", e)
            return False
    
    # ----------------- Helpers -----------------
    def _pose(self):
        """
        Returns the current robot pose (x, y, yaw) in the global frame.
        Uses TF to transform from base frame to the configured global frame.
        """
        T = self.tfbuf.lookup_transform(self.global_frame, self.base_frame, rospy.Time(0), rospy.Duration(0.2))
        x = T.transform.translation.x
        y = T.transform.translation.y
        q = T.transform.rotation
        yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        return x, y, yaw

    def _world_to_cell(self, info, x, y):
        """
        Convert world coordinates (x, y) into grid cell indices of an OccupancyGrid.

        Args:
            info (nav_msgs.msg.MapMetaData): Meta information of the grid (resolution, origin, size).
            x (float): World x-coordinate [m].
            y (float): World y-coordinate [m].

        Returns:
            tuple[int, int] | None:
                (cx, cy) cell indices if inside map bounds, otherwise None.
        """
        res = info.resolution
        ox, oy = info.origin.position.x, info.origin.position.y
        cx = int((x - ox)/res); cy = int((y - oy)/res)
        if 0 <= cx < info.width and 0 <= cy < info.height:
            return cx, cy
        return None

    def _is_occupied(self, x, y):
        """
        Check occupancy at a world position using the global costmap.

        Args:
            x (float): World x-coordinate [m].
            y (float): World y-coordinate [m].

        Returns:
            bool:
                True if the cell is unknown or occupied (cost > threshold),
                False if it is free.
        """
        if self.costmap_info is None or self.costmap_arr is None:
            return True 

        cell = self.world_to_map(x, y)
        if cell is None:
            return True
        cy, cx = cell[1], cell[0]
        v = self.costmap_arr[cy, cx]

        if v < 0:
            return True  # unknown als Hindernis behandeln

        return v > self.cost_free_threshold
    
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
    
    def world_to_map(self, x: float, y: float):
        """
        Convert world coordinates into costmap grid indices.

        Args:
            x (float): World x-coordinate [m].
            y (float): World y-coordinate [m].

        Returns:
            tuple[int, int] | None:
                (mx, my) costmap indices if inside bounds, otherwise None.
        """
        if self.costmap_info is None:
            return None
        res = self.costmap_info.resolution
        ox = self.costmap_info.origin.position.x
        oy = self.costmap_info.origin.position.y
        mx = int((x - ox) / res)
        my = int((y - oy) / res)
        if 0 <= mx < self.costmap_info.width and 0 <= my < self.costmap_info.height:
            return mx, my
        return None

    def _publish_goal(self, x, y, yaw):
        """
        Publish a navigation goal in the global frame.

        Args:
            x (float): Goal x-position in the global frame [m].
            y (float): Goal y-position in the global frame [m].
            yaw (float): Goal yaw orientation in radians.
        """
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.global_frame
        msg.pose.position.x = x
        msg.pose.position.y = y
        q = quaternion_from_euler(0, 0, yaw)
        msg.pose.orientation = Quaternion(*q)
        self.goal_pub.publish(msg)
        self.last_goal = (x, y, yaw)
        rospy.loginfo_throttle(2.0, f"[Planner] goal -> x={x:.2f} y={y:.2f} yaw={yaw:.2f}")

    def _send_move_base_goal(self, ps: PoseStamped):
        """
        Send a PoseStamped goal to move_base via the action interface.

        Args:
            ps (PoseStamped): Target pose in a valid navigation frame.
        """
        goal = MoveBaseGoal()
        goal.target_pose = ps
        self.mb.send_goal(goal, done_cb=self._mb_done_cb)

# ---------------- Utility functions ----------------
        
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
    rospy.init_node("straight_explore_planner")
    StraightPlanner()
    rospy.spin()