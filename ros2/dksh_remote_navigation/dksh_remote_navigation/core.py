"""Transport-independent ownership, lease and sequential mission state machine."""
from collections import OrderedDict
import math
import secrets
import time

from .map_model import Waypoint, demo_map, wrap_yaw


class ProtocolError(ValueError):
    pass


class MockBackend:
    name = "mock"
    radius = .35

    def __init__(self):
        self.map = demo_map()
        self.map.free_cells(self.radius)  # prepare before an operator's lease starts
        self.x, self.y, self.yaw = -5., -3., 0.
        self.state, self.detail = "Idle", ""
        self.path, self.target = [], None
        self.starts = 0
        self.ready = True
        self.motion_stopped = True

    def start(self, goal):
        self.path = self.map.plan((self.x, self.y), (goal.x, goal.y), self.radius)
        if not self.path:
            self.state, self.detail = "Failed", "No safe route."
            return
        self.target = goal
        self.path.pop(0)
        self.starts += 1
        self.motion_stopped = False
        self.state = "Accepted"

    def cancel(self):
        self.path = []
        self.target = None
        self.motion_stopped = True
        self.state = "Canceled"

    def tick(self, dt):
        if self.state == "Accepted":
            self.state = "Navigating"
            return
        if self.state != "Navigating":
            return
        dt = max(0, min(dt, .1))
        budget = dt * .65
        while self.path and budget > 0:
            next_x, next_y = self.path[0]
            dx, dy = next_x - self.x, next_y - self.y
            distance = math.hypot(dx, dy)
            step = (next_x, next_y) if distance <= budget else (
                self.x + dx / distance * budget, self.y + dy / distance * budget)
            if not self.map.segment_free((self.x, self.y), step, self.radius):
                self.cancel()
                self.state, self.detail = "Failed", "Route became unsafe."
                return
            if distance > .0001:
                self.yaw = math.atan2(dy, dx)
            if distance <= budget:
                self.x, self.y = next_x, next_y
                self.path.pop(0)
                budget -= distance
            else:
                self.x, self.y = step
                budget = 0
        if self.path:
            return
        error = wrap_yaw(self.target.yaw - self.yaw)
        self.yaw = wrap_yaw(self.yaw + max(-dt, min(dt, error)))
        if abs(error) < .025:
            self.yaw = self.target.yaw
            self.state, self.motion_stopped = "Succeeded", True

    def distance(self):
        points = [(self.x, self.y)] + list(self.path)
        return sum(math.dist(a, b) for a, b in zip(points, points[1:]))


class MissionGateway:
    lease_seconds = 1.0
    challenge_seconds = 1.0

    def __init__(self, backend=None, clock=time.monotonic, journal=None):
        self.backend = backend or MockBackend()
        self.clock = clock
        self.session = ""
        self.state, self.detail = "Idle", "Select and execute a mission."
        self.stop_latched = False
        self.mission_id = ""
        self.goals, self.index = [], 0
        self.last_heartbeat = 0
        self.last_command_seq, self.status_seq = 0, 0
        self.challenges, self.requests = OrderedDict(), OrderedDict()
        self.heartbeat_requests, self.missions = OrderedDict(), set()
        self.cancel_started = 0
        self.map_revision = self.backend.map.revision if self.backend.map else ""
        self.journal, self.last_recorded_state = journal, None

    def record(self, event, **fields):
        if self.journal is not None:
            self.journal(event, backend=self.backend.name, map_revision=self.map_revision, **fields)

    def record_state(self):
        signature = (self.state, self.mission_id, self.index, self.stop_latched, self.map_revision)
        if signature == self.last_recorded_state:
            return
        self.last_recorded_state = signature
        self.record("state", state=self.state, detail=self.detail, mission_id=self.mission_id,
                    waypoint_index=self.index, waypoint_count=len(self.goals),
                    x=self.backend.x, y=self.backend.y, yaw=self.backend.yaw,
                    stop_latched=self.stop_latched, motion_stopped=self.backend.motion_stopped)

    @property
    def busy(self):
        return self.state in ("Accepted", "Navigating", "Canceling", "StopUnconfirmed")

    def acquire(self):
        if self.session:
            raise ProtocolError("Another operator owns the gateway.")
        self.session = secrets.token_hex(16)
        self.last_command_seq, self.status_seq = 0, 0
        self.last_heartbeat = self.clock()
        self.challenges.clear()
        self.requests.clear()
        self.heartbeat_requests.clear()
        self.missions.clear()
        self.record("operator_connected")
        self.record_state()
        return self.session

    def disconnect(self, session):
        if session != self.session or not self.session:
            return
        self.stop("Operator disconnected; a new mission requires an explicit stop reset.")
        self.session = ""
        self.challenges.clear()
        self.record("operator_disconnected", mission_id=self.mission_id)

    def issue_challenge(self):
        now = self.clock()
        while self.challenges and next(iter(self.challenges.values())) < now:
            self.challenges.popitem(last=False)
        challenge = secrets.token_hex(16)
        self.challenges[challenge] = now + self.challenge_seconds
        while len(self.challenges) > 32:
            self.challenges.popitem(last=False)
        return challenge

    @staticmethod
    def identifier(value):
        return isinstance(value, str) and 1 <= len(value) <= 64

    def handle(self, packet):
        try:
            return self._handle(packet)
        finally:
            self.record_state()

    def _handle(self, packet):
        now = self.clock()
        if not isinstance(packet, dict) or type(packet.get("v")) is not int or packet.get("v") != 1 or packet.get("type") != "command":
            raise ProtocolError("Unsupported command envelope.")
        if not self.session or packet.get("session") != self.session:
            raise ProtocolError("Invalid operator session.")
        command = packet.get("command")
        if command not in ("heartbeat", "submit", "cancel", "stop", "reset_stop"):
            raise ProtocolError("Command is not allowed.")
        seq, request = packet.get("seq"), packet.get("request_id")
        if type(seq) is not int or not 0 < seq <= 2**53 or not self.identifier(request):
            raise ProtocolError("Invalid command identifiers.")
        # Replays never extend the lease, even when they have a fresh sequence number.
        if seq <= self.last_command_seq or request in self.requests or request in self.heartbeat_requests:
            return False
        challenge = packet.get("challenge")
        if not self.identifier(challenge):
            raise ProtocolError("Invalid freshness challenge.")
        if self.challenges.get(challenge, -1) < now:
            return False
        self.last_command_seq = seq
        if command == "heartbeat":
            self.heartbeat_requests[request] = True
            if len(self.heartbeat_requests) > 64:
                self.heartbeat_requests.popitem(last=False)
            self.last_heartbeat = now
            return True
        if len(self.requests) >= 512:
            raise ProtocolError("Command limit reached. Reconnect to create a new session.")
        self.requests[request] = True
        self.record("command", command=command, request_id=request,
                    mission_id=packet.get("mission_id") if self.identifier(packet.get("mission_id")) else "")
        # Cancellation and stop are allowed while waiting for a goal acceptance.
        if command == "stop":
            self.stop("Stop latched by the operator.")
            return True
        if command == "cancel":
            if packet.get("mission_id") == self.mission_id and self.busy:
                self.backend.cancel()
                self.state, self.detail = "Canceling", "Waiting for cancellation and stopped motion."
                self.cancel_started = now
            return True
        if command == "reset_stop":
            if self.backend.motion_stopped and not self.busy and self.backend.ready and now - self.last_heartbeat <= self.lease_seconds:
                self.stop_latched = False
                self.state, self.detail = "Idle", "Stop reset. Submit a new mission."
            return True
        mission = packet.get("mission_id")
        if not self.identifier(mission):
            raise ProtocolError("Invalid mission identifier.")
        if mission in self.missions:
            return False
        if len(self.missions) >= 128:
            raise ProtocolError("Mission limit reached. Reconnect to create a new session.")
        if self.busy:
            self.detail = "Another mission is still active."
            return False
        reason = self.validate_submission(packet)
        if reason:
            self.state, self.detail, self.mission_id = "Rejected", reason, mission
            return False
        self.goals = [Waypoint(float(p["x"]), float(p["y"]), wrap_yaw(float(p["yaw"]))) for p in packet["waypoints"]]
        self.index, self.mission_id = 0, mission
        self.missions.add(mission)
        self.backend.start(self.goals[0])
        self.state, self.detail = "Accepted", "Mission accepted by the simulation gateway."
        self.record("mission_accepted", request_id=request, mission_id=mission,
                    waypoints=[goal.packet() for goal in self.goals])
        return True

    def validate_submission(self, packet):
        if self.stop_latched:
            return "Reset the latched stop before submitting a new mission."
        if not self.backend.ready or self.clock() - self.last_heartbeat > self.lease_seconds:
            return "Robot state or operator lease is not ready."
        if self.backend.map is None or packet.get("map_revision") != self.backend.map.revision:
            return "Map revision changed. Select new waypoints."
        points = packet.get("waypoints")
        if not isinstance(points, list) or not 1 <= len(points) <= 10:
            return "A mission requires 1 to 10 waypoints."
        previous = self.backend.x, self.backend.y
        for point in points:
            if not isinstance(point, dict) or any(type(point.get(k)) not in (int, float) for k in ("x", "y", "yaw")):
                return "Waypoint coordinates must be numbers."
            if any(abs(point[k]) > 1e6 for k in ("x", "y", "yaw")) or not all(math.isfinite(point[k]) for k in ("x", "y", "yaw")):
                return "Waypoint coordinates must be finite."
            if not self.backend.map.valid(point["x"], point["y"], self.backend.radius):
                return "Waypoint is outside free space or has insufficient clearance."
            if isinstance(self.backend, MockBackend) and not self.backend.map.plan(previous, (point["x"], point["y"]), self.backend.radius):
                return "A waypoint is unreachable."
            previous = point["x"], point["y"]
        return ""

    def stop(self, reason):
        self.stop_latched = True
        self.backend.cancel()
        self.state = "Stopped" if self.backend.motion_stopped else "Canceling"
        self.cancel_started = self.clock()
        self.detail = reason
        self.record_state()

    def check_fences(self):
        if self.session and self.clock() - self.last_heartbeat > self.lease_seconds and not self.stop_latched:
            self.stop("Operator heartbeat expired. Mission cannot resume automatically.")
        revision = self.backend.map.revision if self.backend.map else ""
        if revision != self.map_revision:
            self.map_revision = revision
            if self.busy:
                self.stop("Map frame or identity changed.")

    def tick(self, dt):
        try:
            self._tick(dt)
        finally:
            self.record_state()

    def _tick(self, dt):
        # Do not advance virtual motion after a lease or known map identity expires.
        self.check_fences()
        self.backend.tick(dt)
        # ROS callbacks may invalidate the map during spin_once.
        self.check_fences()
        if self.state in ("Canceling", "StopUnconfirmed"):
            if self.backend.motion_stopped:
                self.state = "Stopped" if self.stop_latched else "Canceled"
                self.detail = "Stop confirmed in simulation." if self.stop_latched else "Cancellation and stopped motion confirmed."
            elif self.clock() - self.cancel_started > 3:
                self.state, self.detail = "StopUnconfirmed", "Cancellation requested; stopped motion is not yet confirmed."
            return
        if self.state not in ("Accepted", "Navigating"):
            return
        if not self.backend.ready:
            self.stop("Robot pose or navigation backend became unavailable.")
        elif self.backend.state == "Failed":
            self.stop(self.backend.detail or "Navigation failed.")
            if self.backend.motion_stopped:
                self.state = "Failed"
        elif self.backend.state == "Succeeded":
            self.index += 1
            if self.index == len(self.goals):
                self.state, self.detail = "Succeeded", "All waypoints reached."
            else:
                self.backend.start(self.goals[self.index])
                self.state, self.detail = "Accepted", "Starting the next waypoint."
        else:
            self.state = self.backend.state
            self.detail = "Navigating to the current waypoint."

    def snapshot(self):
        self.status_seq += 1
        return {"v": 1, "type": "status", "session": self.session, "seq": self.status_seq,
                "challenge": self.issue_challenge(), "backend": self.backend.name,
                "state": self.state, "detail": self.detail, "mission_id": self.mission_id,
                "map_revision": self.map_revision, "ready": self.backend.ready,
                "stop_latched": self.stop_latched, "x": self.backend.x, "y": self.backend.y,
                "yaw": self.backend.yaw, "footprint_radius": self.backend.radius,
                "waypoint_index": self.index, "waypoint_count": len(self.goals),
                "distance_remaining": self.backend.distance(),
                "path": [Waypoint(x, y).packet() for x, y in self.backend.path]}
