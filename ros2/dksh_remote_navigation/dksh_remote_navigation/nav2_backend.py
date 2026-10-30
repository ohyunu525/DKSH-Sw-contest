"""Nav2 integration restricted to ROS simulation time and synthetic sensor inputs.

This is a simulation adapter, not a hardware locomotion or emergency stop layer.
"""
import hashlib
import math
import secrets
import time

from .map_model import GridMap


class Nav2SimulationBackend:
    name = "nav2_simulation"
    radius = .35

    def __init__(self):
        import rclpy
        from rclpy.action import ActionClient
        from rclpy.node import Node
        from rclpy.parameter import Parameter
        from rclpy.time import Time
        from rclpy.qos import QoSProfile, DurabilityPolicy
        from nav2_msgs.action import NavigateToPose
        from nav_msgs.msg import OccupancyGrid, Odometry, Path
        from sensor_msgs.msg import LaserScan
        from rosgraph_msgs.msg import Clock
        from tf2_ros import Buffer, TransformListener

        rclpy.init()
        self.rclpy = rclpy
        self.time_type = Time
        self.node = Node("dksh_remote_navigation", parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.action_type = NavigateToPose
        self.action = ActionClient(self.node, NavigateToPose, "navigate_to_pose")
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, self.node)
        self.map = None
        self.x = self.y = self.yaw = 0.
        self.state, self.detail, self.path = "Idle", "Waiting for simulation data.", []
        self.ready = False
        self.distance_remaining = 0.
        self.generation = 0
        self.pending, self.handle, self.canceling = False, None, False
        self.epoch = secrets.token_hex(8)
        self.clock_at = self.scan_at = self.odom_at = self.pose_at = -math.inf
        self.still_since = None
        self.last_clock = self.last_scan = self.last_odom = None
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.node.create_subscription(OccupancyGrid, "/map", self.on_map, qos)
        self.node.create_subscription(Odometry, "/odom", self.on_odom, 10)
        from rclpy.qos import qos_profile_sensor_data
        self.node.create_subscription(LaserScan, "/scan", self.on_scan, qos_profile_sensor_data)
        self.node.create_subscription(Clock, "/clock", self.on_clock, qos_profile_sensor_data)
        self.node.create_subscription(Path, "/plan", self.on_path, 1)

    @staticmethod
    def stamp(value):
        return value.sec, value.nanosec

    @staticmethod
    def yaw_of(q):
        return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

    def on_map(self, msg):
        if msg.header.frame_id != "map":
            self.map = None
            return
        info = msg.info
        geometry = (info.width, info.height, info.resolution,
                    info.origin.position.x, info.origin.position.y, self.yaw_of(info.origin.orientation),
                    self.stamp(info.map_load_time))
        revision = self.epoch + "-" + hashlib.sha256(repr(geometry).encode()).hexdigest()[:16]
        try:
            self.map = GridMap(int(info.width), int(info.height), float(info.resolution),
                               float(info.origin.position.x), float(info.origin.position.y),
                               self.yaw_of(info.origin.orientation), tuple(msg.data), revision)
        except ValueError:
            self.map = None

    def on_clock(self, msg):
        stamp = self.stamp(msg.clock)
        if self.last_clock is not None and stamp < self.last_clock:
            self.map = None
            self.epoch = secrets.token_hex(8)
            self.pose_at = self.odom_at = self.scan_at = -math.inf
            self.still_since = None
            self.last_scan = self.last_odom = None
        if stamp != self.last_clock:
            self.last_clock, self.clock_at = stamp, time.monotonic()

    def on_scan(self, msg):
        stamp = self.stamp(msg.header.stamp)
        if (self.fresh_stamp(msg.header.stamp) and
                (self.last_scan is None or stamp > self.last_scan) and msg.ranges):
            self.last_scan, self.scan_at = stamp, time.monotonic()

    def fresh_stamp(self, stamp):
        # Callback receipt time alone cannot establish freshness: queued ROS
        # messages may be delivered long after their sensor samples were taken.
        sim_now = self.node.get_clock().now().nanoseconds / 1e9
        sample_time = stamp.sec + stamp.nanosec / 1e9
        return 0 <= sim_now - sample_time < .5

    def on_odom(self, msg):
        stamp = self.stamp(msg.header.stamp)
        if (not self.fresh_stamp(msg.header.stamp) or
                (self.last_odom is not None and stamp <= self.last_odom)):
            return
        now = time.monotonic()
        if now - self.odom_at >= .5:
            self.still_since = None
        self.last_odom, self.odom_at = stamp, now
        twist = msg.twist.twist
        if not all(math.isfinite(v) for v in (twist.linear.x, twist.linear.y, twist.angular.z)):
            self.odom_at = -math.inf
            self.still_since = None
            return
        still = (math.hypot(twist.linear.x, twist.linear.y) < .01 and abs(twist.angular.z) < .02)
        if not still:
            self.still_since = None
        elif self.still_since is None:
            self.still_since = self.odom_at

    def on_path(self, msg):
        if msg.header.frame_id == "map" and self.state in ("Accepted", "Navigating") and not self.canceling:
            path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses[:4096]]
            self.path = path if all(math.isfinite(v) for point in path for v in point) else []

    @property
    def motion_stopped(self):
        now = time.monotonic()
        return (not self.pending and self.handle is None and now - self.odom_at < .5
                and self.still_since is not None and now - self.still_since >= .3)

    def start(self, waypoint):
        self.generation += 1
        generation = self.generation
        self.canceling, self.pending = False, True
        self.state, self.path = "Accepted", []
        goal = self.action_type.Goal()
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = self.node.get_clock().now().to_msg()
        goal.pose.pose.position.x, goal.pose.pose.position.y = waypoint.x, waypoint.y
        goal.pose.pose.orientation.z = math.sin(waypoint.yaw / 2)
        goal.pose.pose.orientation.w = math.cos(waypoint.yaw / 2)
        try:
            future = self.action.send_goal_async(goal, feedback_callback=lambda msg: self.feedback(generation, msg))
            future.add_done_callback(lambda result: self.accepted(generation, result))
        except Exception:
            self.pending = True
            self.state, self.detail = "Failed", "Nav2 goal submission failed."

    def feedback(self, generation, message):
        if (generation == self.generation and not self.canceling and
                self.state in ("Accepted", "Navigating")):
            distance = float(message.feedback.distance_remaining)
            if not math.isfinite(distance) or distance < 0:
                self.cancel()
                self.state, self.detail = "Failed", "Nav2 returned an invalid remaining distance."
                return
            self.distance_remaining = distance
            self.state = "Navigating"

    def accepted(self, generation, future):
        if generation != self.generation:
            # A stale future's exception must not change the current goal's state.
            try:
                handle = future.result()
                if handle.accepted:
                    handle.cancel_goal_async()
            except Exception:
                pass
            return
        try:
            handle = future.result()
            self.pending = False
            if not handle.accepted:
                self.state, self.detail = "Failed", "Nav2 rejected the goal."
                return
            self.handle = handle
            if not self.canceling:
                self.state = "Navigating"
            handle.get_result_async().add_done_callback(lambda result: self.finished(generation, result))
            if self.canceling and self.handle is handle:
                handle.cancel_goal_async()
        except Exception:
            # Without a response we cannot prove the submitted goal was rejected.
            self.pending = True
            self.state, self.detail = "Failed", "Nav2 goal acceptance failed."

    def finished(self, generation, future):
        if generation != self.generation:
            return
        try:
            result = future.result()
            self.handle, self.pending, self.path = None, False, []
            self.distance_remaining = 0.
            if self.canceling:
                self.state = "Canceled"
            elif result.status == 4 and result.result.error_code == 0:
                self.state = "Succeeded"
            else:
                self.state, self.detail = "Failed", "Nav2 did not complete the waypoint."
        except Exception:
            # An exception is not proof that the goal ended. Retain the outstanding handle.
            self.state, self.detail = "Failed", "Nav2 result is unavailable."

    def cancel(self):
        self.canceling = True
        self.path, self.distance_remaining = [], 0.
        if self.handle is not None:
            try:
                self.handle.cancel_goal_async()
            except Exception:
                self.detail = "Nav2 cancellation is unavailable."
        self.state = "Canceled"

    def tick(self, dt):
        # scan/odom/clock/TF and action callbacks can exceed the 20 Hz gateway tick.
        # Drain a bounded batch so callback backlog does not masquerade as fresh input.
        for _ in range(16):
            self.rclpy.spin_once(self.node, timeout_sec=0)
        now = time.monotonic()
        try:
            transform = self.tf.lookup_transform("map", "base_footprint", self.time_type())
            translation, rotation = transform.transform.translation, transform.transform.rotation
            sim_stamp = self.node.get_clock().now().nanoseconds / 1e9
            stamp = transform.header.stamp.sec + transform.header.stamp.nanosec / 1e9
            if 0 <= sim_stamp - stamp < .5 and all(math.isfinite(v) for v in (translation.x, translation.y, rotation.x, rotation.y, rotation.z, rotation.w)):
                self.x, self.y, self.yaw = translation.x, translation.y, self.yaw_of(rotation)
                self.pose_at = now
        except Exception:
            pass
        self.ready = (self.map is not None and self.action.server_is_ready()
                      and all(now - stamp < .75 for stamp in (self.clock_at, self.scan_at, self.odom_at, self.pose_at)))

    def distance(self):
        return self.distance_remaining

    def close(self):
        self.cancel()
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline and (self.pending or self.handle is not None):
            self.rclpy.spin_once(self.node, timeout_sec=.05)
        self.node.destroy_node()
        self.rclpy.shutdown()
