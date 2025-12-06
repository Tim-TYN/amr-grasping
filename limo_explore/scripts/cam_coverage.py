#!/usr/bin/env python3
import rospy, math
import numpy as np
import tf2_ros
from nav_msgs.msg import OccupancyGrid
from tf.transformations import euler_from_quaternion
from sensor_msgs.msg import CameraInfo
from std_srvs.srv import Empty, EmptyResponse


class CamCoverage:
    """
    ROS node to compute a camera coverage map.

    Idea:
    - Takes the current SLAM occupancy grid map (/map) and the camera pose from TF.
    - Simulates rays (raycasting) within the camera’s field of view (FOV).
    - Marks all map cells that fall inside the visible area as "seen".
    - Publishes the result as an OccupancyGrid (/cam_coverage):
        - -1 = unknown
        - 100 = seen by the camera
    """
    
    def __init__(self):
        """Initialize node: subscribers, publishers, parameters, TF, and timer."""

        self.tfbuf = tf2_ros.Buffer()
        self.tfl = tf2_ros.TransformListener(self.tfbuf)
        
        self.fx = 487.5594177246094
        self.width = 640
        self.fov = 2.0 * math.atan((self.width * 0.5) / self.fx)

        self.map = None
        self.cov_accum = None
        self.cov_msg = None
        self.initial_region_marked = False

        self.map_sub = rospy.Subscriber(rospy.get_param("~map_topic","/map"),
                                        OccupancyGrid, self.map_cb, queue_size=1)
        self.pub = rospy.Publisher(rospy.get_param("~coverage_topic","/cam_coverage"),
                                   OccupancyGrid, queue_size=1, latch=True)
        
        self.range_m = rospy.get_param("~range_m", 1.5)
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/camera/color/camera_info")
        self.camera_frame = rospy.get_param("~frame_camera", "camera_link")

        try:
            msg = rospy.wait_for_message(self.camera_info_topic, CameraInfo, timeout=2.0)
            self._apply_caminfo(msg)

            if self.camera_frame in ("auto", "", None):
                self.camera_frame = msg.header.frame_id or "camera_link"

            rospy.loginfo("[cam_coverage] CameraInfo: fx=%.2f, width=%d, FOV=%.1f° (frame=%s)",
                        self.fx, self.width, math.degrees(self.fov), self.camera_frame)
        except rospy.ROSException:
            rospy.logwarn("[cam_coverage] No CameraInfo %s (timeout) – using defaults fx=%.2f, width=%d",
                        self.camera_info_topic, self.fx, self.width)

        self.timer = rospy.Timer(rospy.Duration(0.2), self.tick)  # 5 Hz

        # Reset service
        self.reset_srv_global = rospy.Service("/cam_coverage/reset", Empty, self.handle_reset)


    def map_cb(self, msg):
        """
        Map callback: store the latest map. If geometry (size/resolution/origin/frame) changes,
        reinitialize the accumulated coverage to match the new map.
        """
        reinit_needed = (
            self.map is None
            or self._map_geom_key(self.map) != self._map_geom_key(msg)
        )
        self.map = msg
        if reinit_needed:
            self._init_cov_from_map(msg)

    def _apply_caminfo(self, msg: CameraInfo):
        """
        Update the internal camera model parameters from a ROS CameraInfo message.
        """
        if msg.width:
            self.width = msg.width
        if msg.K[0] > 0:
            self.fx = msg.K[0]
            
        self.fov = 2.0 * math.atan((self.width * 0.5) / self.fx)

    def tick(self, _):
        """
        Called periodically by the timer.
        Performs coverage computation:
        - Gets the camera pose in the map frame
        - Performs raycasting in the camera’s FOV
        - Publishes a new coverage grid
        """

        if self.map is None or self.cov_accum is None:
            return

        try:
            T = self.tfbuf.lookup_transform(self.map.header.frame_id, self.camera_frame,
                                            rospy.Time(0), rospy.Duration(0.2))
        except Exception:
            return

        # Camera pose + yaw orientation
        tx = T.transform.translation
        q = T.transform.rotation
        yaw = euler_from_quaternion([q.x,q.y,q.z,q.w])[2]
        # yaw += math.radians(90) # offset
        origin = (tx.x, tx.y)

        if not self.initial_region_marked:
            self._mark_region_seen(origin, 0.5)
            self.initial_region_marked = True
            # Publish once so RViz shows it immediately
            self._publish_cov_accum()

        # Build instantaneous coverage snapshot: -1 everywhere, 100 where seen now
        H, W = self.map.info.height, self.map.info.width
        cov_inst = np.full((H, W), -1, dtype=np.int16)

        # Raycast across the camera FOV and mark visible free/unknown cells as 100
        rays  = 181
        steps = int(self.range_m / max(self.map.info.resolution, 1e-6))
        for ang in np.linspace(yaw - self.fov/2.0, yaw + self.fov/2.0, rays):
            self._cast_ray(origin, ang, steps, cov_inst)

        # Accumulate: keep anything already seen, add new 100s from this frame
        # Equivalent to cov_accum = maximum(cov_accum, cov_inst) cellwise
        seen_mask = (cov_inst == 100)
        self.cov_accum[seen_mask] = 100

        # Publish updated accumulated coverage
        self._publish_cov_accum()

    def _mark_region_seen(self, origin_xy, radius_m):
        """
        Mark a circular region around the camera origin as seen in cov_accum.

        Args:
            origin_xy (tuple[float, float]): Camera position (x, y) in map frame [m].
            radius_m (float): Radius of the region to mark as seen [m].
        """
        if self.map is None or self.cov_accum is None:
            return

        res = self.map.info.resolution
        H, W = self.map.info.height, self.map.info.width

        # Convert robot world position to cell indices
        cell_center = self._world_to_cell(origin_xy[0], origin_xy[1])
        if cell_center is None:
            rospy.logwarn("[cam_coverage] Initial region: origin outside map bounds.")
            return

        cx0, cy0 = cell_center
        radius_cells = int(radius_m / res) + 1
        radius_sq = radius_m * radius_m

        ox = self.map.info.origin.position.x
        oy = self.map.info.origin.position.y

        for cy in range(max(0, cy0 - radius_cells), min(H, cy0 + radius_cells + 1)):
            for cx in range(max(0, cx0 - radius_cells), min(W, cx0 + radius_cells + 1)):
                # Compute cell center in world coordinates
                x = ox + (cx + 0.5) * res
                y = oy + (cy + 0.5) * res

                dx = x - origin_xy[0]
                dy = y - origin_xy[1]
                if dx * dx + dy * dy <= radius_sq:
                    # Optional: skip occupied cells
                    occ = self._occ_at(cx, cy)
                    if occ < 50:  # unknown or free
                        self.cov_accum[cy, cx] = 100

    def handle_reset(self, _req):
        """
        Reset service handler: clears the accumulated coverage (sets all cells to -1),
        then republishes the empty coverage map.
        """
        if self.map is not None and self.cov_accum is not None:
            self.cov_accum.fill(-1)
            self._publish_cov_accum()
            rospy.loginfo("[cam_coverage] coverage reset.")
        return EmptyResponse()
    
    def _init_cov_from_map(self, map_msg):
        """
        Initialize or reinitialize the accumulated coverage grid to match a map.

        Args:
            map_msg (OccupancyGrid): Reference map whose geometry (size, res, origin)
                                     is used to size cov_accum and cov_msg.
        """
        H, W = map_msg.info.height, map_msg.info.width
        self.cov_accum = np.full((H, W), -1, dtype=np.int16)

        # Prepare a reusable OccupancyGrid message for publishing
        cov = OccupancyGrid()
        cov.header.frame_id = map_msg.header.frame_id
        cov.info = map_msg.info
        cov.data = (-1 * np.ones(H * W, dtype=np.int8)).tolist()
        self.cov_msg        = cov
        self.pub.publish(cov)  # publish empty (latched), so RViz shows something immediately

        self.initial_region_marked = False

        rospy.loginfo("[cam_coverage] coverage (re)initialized: %dx%d, res=%.3fm",
                      W, H, map_msg.info.resolution)
        
    def _publish_cov_accum(self):
        """
        Publish the current accumulated coverage as an OccupancyGrid.
        """
        if self.cov_msg is None or self.map is None:
            return
        self.cov_msg.header.stamp = rospy.Time.now()
        # Flatten as int8 list
        self.cov_msg.data = self.cov_accum.flatten().astype(np.int8).tolist()
        self.pub.publish(self.cov_msg)

    def _map_geom_key(self, m):
        """
        Build a hashable key that describes the geometry of a map.

        Args:
            m (OccupancyGrid): Map message.

        Returns:
            tuple:
                (resolution, width, height, origin_x, origin_y, frame_id)
                used to detect changes in map geometry.
        """
        info = m.info
        ori  = info.origin.position
        return (info.resolution, info.width, info.height, ori.x, ori.y, m.header.frame_id)

    def _world_to_cell(self, x, y):
        """
        Convert world coordinates into integer map indices.

        Args:
            x (float): World x-coordinate [m].
            y (float): World y-coordinate [m].

        Returns:
            tuple[int, int] | None:
                (cx, cy) cell indices if inside map bounds, otherwise None.
        """
        res = self.map.info.resolution
        ox  = self.map.info.origin.position.x
        oy  = self.map.info.origin.position.y
        cx  = int((x - ox) / res)
        cy  = int((y - oy) / res)
        if 0 <= cx < self.map.info.width and 0 <= cy < self.map.info.height:
            return cx, cy
        return None

    def _occ_at(self, cx, cy):
        """
        Retrieve occupancy value at a given map cell.

        Args:
            cx (int): Cell x-index.
            cy (int): Cell y-index.

        Returns:
            int:
                Occupancy value:
                -1 = unknown, 0 = free, >= 50 = occupied.
        """

        idx = cy * self.map.info.width + cx
        v = self.map.data[idx]
        return v

    def _cast_ray(self, origin_xy, ang, steps, cov_inst):
        """
        Cast a single ray from the camera origin into the map.

        For each step:
        - Converts ray sample to map cell.
        - Stops when hitting an occupied cell or leaving bounds.
        - Marks free/unknown cells as 100 in cov_inst.

        Args:
            origin_xy (tuple[float, float]): Ray origin (x, y) in map frame [m].
            ang (float): Ray direction in radians.
            steps (int): Maximum number of steps (distance / resolution).
            cov_inst (np.ndarray): Instantaneous coverage grid (HxW) to update.
        """
        res = self.map.info.resolution
        x0, y0 = origin_xy
        for k in range(steps):
            xk = x0 + math.cos(ang) * k * res
            yk = y0 + math.sin(ang) * k * res
            cell = self._world_to_cell(xk, yk)
            if cell is None:
                break
            cx, cy = cell
            occ = self._occ_at(cx, cy)
            if occ >= 50:
                break  # hit an obstacle; ray ends here
            # mark as seen in the instantaneous snapshot
            if occ == -1 or occ == 0:
                cov_inst[cy, cx] = 100

if __name__ == "__main__":
    rospy.init_node("cam_coverage_node")
    CamCoverage()
    rospy.spin()
