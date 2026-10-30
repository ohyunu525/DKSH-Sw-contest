import asyncio
import json
import math
import unittest

from dksh_remote_navigation.core import MissionGateway, MockBackend, ProtocolError
from dksh_remote_navigation.map_model import GridMap, Waypoint
from dksh_remote_navigation.server import GatewayServer, decode


class FakeClock:
    def __init__(self):
        self.now = 100.
    def __call__(self):
        return self.now


class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.backend = MockBackend()
        self.core = MissionGateway(self.backend, self.clock)
        self.session = self.core.acquire()
        self.seq = 0

    def command(self, command, **fields):
        self.seq += 1
        result = {"v": 1, "type": "command", "session": self.session,
                  "challenge": self.core.issue_challenge(), "seq": self.seq,
                  "request_id": f"request-{self.seq}", "command": command,
                  "mission_id": "mission-1", "map_revision": self.backend.map.revision,
                  "waypoints": [{"x": -4., "y": -3., "yaw": 0.}]}
        result.update(fields)
        return result

    def test_duplicate_request_and_old_sequence_do_not_start_twice(self):
        request = self.command("submit")
        self.assertTrue(self.core.handle(request))
        self.assertFalse(self.core.handle(request))
        request["seq"] += 100
        self.assertFalse(self.core.handle(request))
        self.assertEqual(self.backend.starts, 1)

    def test_replayed_heartbeat_does_not_extend_lease(self):
        self.core.handle(self.command("submit"))
        heartbeat = self.command("heartbeat")
        self.core.handle(heartbeat)
        self.clock.now += .8
        self.core.handle(heartbeat)
        self.clock.now += .3
        self.core.tick(.1)
        self.assertTrue(self.core.stop_latched)
        self.assertEqual(self.core.state, "Stopped")

    def test_expired_challenge_cannot_renew_lease(self):
        heartbeat = self.command("heartbeat")
        self.clock.now += 1.01
        self.assertFalse(self.core.handle(heartbeat))
        self.core.tick(.1)
        self.assertTrue(self.core.stop_latched)

    def test_expired_lease_stops_before_advancing_motion(self):
        self.core.handle(self.command("submit"))
        self.core.tick(.1)
        position = self.backend.x, self.backend.y
        self.clock.now += 1.01
        self.core.tick(.1)
        self.assertEqual((self.backend.x, self.backend.y), position)
        self.assertEqual(self.core.state, "Stopped")

    def test_late_heartbeat_cannot_revive_expired_lease_between_ticks(self):
        self.core.handle(self.command("submit"))
        self.core.tick(.1)
        position = self.backend.x, self.backend.y
        self.clock.now += 1.01
        # A newly published challenge is still valid even though the old lease
        # expired. The heartbeat can renew freshness only after stop is latched.
        self.assertTrue(self.core.handle(self.command("heartbeat")))
        self.assertTrue(self.core.stop_latched)
        self.assertEqual(self.core.state, "Stopped")
        self.core.tick(.1)
        self.assertEqual((self.backend.x, self.backend.y), position)
        self.assertFalse(self.core.handle(self.command("submit", mission_id="late")))
        self.assertEqual(self.backend.starts, 1)

    def test_request_and_mission_replay_survive_many_heartbeats(self):
        old = self.command("submit")
        self.core.handle(old)
        self.core.handle(self.command("cancel"))
        self.core.tick(.1)
        for _ in range(600):
            self.core.handle(self.command("heartbeat"))
        old["seq"] = 10000
        old["challenge"] = self.core.issue_challenge()
        self.assertFalse(self.core.handle(old))
        self.assertFalse(self.core.handle(self.command("submit")))
        self.assertEqual(self.backend.starts, 1)
        self.assertLessEqual(len(self.core.challenges), 32)

    def test_oversized_number_is_rejected_before_float_conversion(self):
        self.assertFalse(self.core.handle(self.command("submit", waypoints=[
            {"x": 10**400, "y": -3, "yaw": 0}])))
        self.assertEqual(self.backend.starts, 0)

    def test_freshness_challenge_requires_an_identifier(self):
        with self.assertRaises(ProtocolError):
            self.core.handle(self.command("stop", challenge={"invalid": True}))

    def test_cancel_before_acceptance_and_mid_move(self):
        self.core.handle(self.command("submit"))
        self.core.handle(self.command("cancel"))
        self.core.tick(.1)
        self.assertEqual(self.core.state, "Canceled")
        self.assertEqual(self.backend.x, -5)
        self.core.handle(self.command("submit", mission_id="mission-2"))
        self.core.tick(.1)
        self.core.tick(.1)
        self.core.handle(self.command("cancel", mission_id="mission-2"))
        stopped = self.backend.x
        for _ in range(20):
            self.core.tick(.1)
        self.assertEqual(self.backend.x, stopped)

    def test_disconnect_and_new_session_do_not_resume(self):
        self.core.handle(self.command("submit"))
        self.core.disconnect(self.session)
        self.session = self.core.acquire()
        self.assertFalse(self.core.handle(self.command("submit")))
        self.core.handle(self.command("reset_stop"))
        self.assertFalse(self.core.stop_latched)
        self.core.tick(.1)
        self.assertEqual(self.backend.x, -5)

    def test_only_one_operator_can_acquire(self):
        with self.assertRaises(ProtocolError):
            self.core.acquire()

    def test_unconfirmed_stop_blocks_reset_and_new_missions(self):
        def cancel_without_motion_confirmation():
            self.backend.state = "Canceled"
            self.backend.motion_stopped = False
        self.backend.cancel = cancel_without_motion_confirmation
        self.core.handle(self.command("submit"))
        self.core.handle(self.command("stop"))
        self.clock.now += 3.01
        self.core.tick(.1)
        self.assertEqual(self.core.state, "StopUnconfirmed")
        self.core.handle(self.command("heartbeat"))
        self.core.handle(self.command("reset_stop"))
        self.assertTrue(self.core.stop_latched)
        self.assertFalse(self.core.handle(self.command("submit", mission_id="second")))
        self.backend.motion_stopped = True
        self.core.tick(.1)
        self.assertEqual(self.core.state, "Stopped")
        self.core.handle(self.command("reset_stop"))
        self.assertFalse(self.core.stop_latched)
        self.assertEqual(self.backend.starts, 1)

    def test_bad_waypoints_never_start(self):
        for point in ({"x": math.nan, "y": 0, "yaw": 0},
                      {"x": 0, "y": 0, "yaw": math.inf},
                      {"x": 100, "y": 100, "yaw": 0},
                      {"x": -5, "y": -3, "yaw": "zero"},
                      {"x": 6, "y": 4, "yaw": 0}):
            self.assertFalse(self.core.handle(self.command("submit", waypoints=[point])))
        self.assertFalse(self.core.handle(self.command("submit", waypoints=[])))
        self.assertFalse(self.core.handle(self.command("submit", map_revision="stale")))
        self.assertEqual(self.backend.starts, 0)

    def test_stale_mission_cancel_does_not_cancel_new_mission(self):
        self.core.handle(self.command("submit", mission_id="new"))
        self.core.handle(self.command("cancel", mission_id="old"))
        self.core.tick(.1)
        self.assertEqual(self.core.state, "Navigating")

    def test_map_identity_change_stops_active_mission(self):
        self.core.handle(self.command("submit"))
        self.core.tick(.1)
        position = self.backend.x, self.backend.y
        old = self.backend.map
        self.backend.map = GridMap(old.width, old.height, old.resolution, old.origin_x,
                                   old.origin_y, old.origin_yaw, old.cells, "replacement")
        self.core.tick(.1)
        self.assertTrue(self.core.stop_latched)
        self.assertEqual(self.core.state, "Stopped")
        self.assertEqual((self.backend.x, self.backend.y), position)

    def test_second_waypoint_failure_prevents_third(self):
        self.core.handle(self.command("submit", waypoints=[
            {"x": -4.9, "y": -3, "yaw": 0}, {"x": -4.8, "y": -3, "yaw": 0},
            {"x": -4.7, "y": -3, "yaw": 0}]))
        for _ in range(30):
            self.core.tick(.1)
            if self.backend.starts == 2:
                break
        self.backend.state, self.backend.detail = "Failed", "Injected failure."
        self.core.tick(.1)
        for _ in range(20):
            self.core.tick(.1)
        self.assertEqual(self.backend.starts, 2)
        self.assertTrue(self.core.stop_latched)

    def test_successive_goals_reach_final_heading(self):
        self.core.handle(self.command("submit", waypoints=[
            {"x": -4.8, "y": -3, "yaw": .2}, {"x": -4.6, "y": -3, "yaw": -1.}]))
        for _ in range(100):
            self.core.tick(.1)
            if self.core.state == "Succeeded":
                break
        self.assertEqual(self.core.state, "Succeeded")
        self.assertEqual(self.core.index, 2)
        self.assertAlmostEqual(self.backend.yaw, -1.)

    def test_rotated_map_and_corner_obstacles(self):
        map_ = GridMap(20, 12, .2, -8., 3., .7, (0,) * 240, "rotated")
        point = map_.centre(8, 5)
        self.assertEqual(map_.indices(*point), (8, 5))
        self.assertTrue(map_.valid(*point, .2))
        corner = GridMap(2, 2, 1, 0, 0, 0, (0, 100, 100, 0), "corner")
        self.assertFalse(corner.plan(corner.centre(0, 0), corner.centre(1, 1), 0))

    def test_protocol_rejects_duplicate_keys_nonfinite_and_topic_injection(self):
        for line in ('{"v":1,"v":2}', '{"x":NaN}', '[]', '{'):
            with self.assertRaises(ProtocolError):
                decode(line)
        with self.assertRaises(ProtocolError):
            self.core.handle(self.command("publish_cmd_vel"))


class SocketTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.service = GatewayServer(token="integration-secret")
        self.server = await asyncio.start_server(self.service.handle_connection, "127.0.0.1", 0, limit=2 * 1024 * 1024)
        self.port = self.server.sockets[0].getsockname()[1]
        self.tick = asyncio.create_task(self.service.tick())
        self.connections = []

    async def asyncTearDown(self):
        for writer in self.connections:
            writer.close()
            await writer.wait_closed()
        self.tick.cancel()
        await asyncio.gather(self.tick, return_exceptions=True)
        self.server.close()
        await self.server.wait_closed()
        await asyncio.sleep(.02)

    async def connect(self, token):
        reader, writer = await asyncio.open_connection("127.0.0.1", self.port)
        self.connections.append(writer)
        writer.write((json.dumps({"v": 1, "type": "hello", "token": token}) + "\n").encode())
        await writer.drain()
        return reader, writer

    async def receive(self, reader):
        return decode(await asyncio.wait_for(reader.readline(), 3))

    async def test_tcp_authentication_and_operator_collision(self):
        reader, _ = await self.connect("wrong")
        self.assertEqual((await self.receive(reader))["type"], "error")
        first, _ = await self.connect("integration-secret")
        self.assertEqual((await self.receive(first))["type"], "welcome")
        second, _ = await self.connect("integration-secret")
        self.assertEqual((await self.receive(second))["type"], "error")

    async def test_command_flood_closes_connection_and_latches_stop(self):
        reader, writer = await self.connect("integration-secret")
        welcome = await self.receive(reader)
        await self.receive(reader)
        state = await self.receive(reader)
        for i in range(31):
            packet = {"v": 1, "type": "command", "command": "heartbeat", "session": welcome["session"],
                      "challenge": state["challenge"], "seq": i + 1, "request_id": f"flood-{i}"}
            writer.write((json.dumps(packet) + "\n").encode())
        await writer.drain()
        response = await self.receive(reader)
        self.assertEqual(response["type"], "error")
        await asyncio.sleep(.05)
        self.assertTrue(self.service.gateway.stop_latched)

    async def test_invalid_outgoing_state_disconnects_and_latches_stop(self):
        reader, _ = await self.connect("integration-secret")
        await self.receive(reader)
        await self.receive(reader)
        await self.receive(reader)
        self.service.gateway.backend.radius = float("nan")
        while await asyncio.wait_for(reader.readline(), 3):
            pass
        self.assertTrue(self.service.gateway.stop_latched)
        self.assertEqual(self.service.gateway.session, "")

    async def test_tcp_map_goal_completion_cancel_and_disconnect(self):
        reader, writer = await self.connect("integration-secret")
        welcome = await self.receive(reader)
        self.assertEqual(welcome["type"], "welcome")
        map_ = await self.receive(reader)
        self.assertEqual(len(map_["cells"]), 4800)
        state = await self.receive(reader)
        seq = 0

        async def command(name, mission="socket-mission", points=None):
            nonlocal seq
            seq += 1
            packet = {"v": 1, "type": "command", "command": name, "session": welcome["session"],
                      "challenge": state["challenge"], "seq": seq, "request_id": f"socket-{seq}",
                      "mission_id": mission, "map_revision": map_["map_revision"],
                      "waypoints": points or [{"x": -4.8, "y": -3, "yaw": 0}]}
            writer.write((json.dumps(packet) + "\n").encode())
            await writer.drain()

        await command("submit")
        for _ in range(30):
            state = await self.receive(reader)
            await command("heartbeat")
            if state["state"] == "Succeeded":
                break
        self.assertEqual(state["state"], "Succeeded")
        self.assertAlmostEqual(state["x"], -4.8)
        await command("submit", "second", [{"x": -4., "y": -3., "yaw": 0}])
        await command("cancel", "second")
        for _ in range(10):
            state = await self.receive(reader)
            await command("heartbeat")
            if state["state"] == "Canceled":
                break
        self.assertEqual(state["state"], "Canceled")
        await command("submit", "third", [{"x": -4., "y": -3., "yaw": 0}])
        writer.close()
        await writer.wait_closed()
        await asyncio.sleep(.15)
        self.assertTrue(self.service.gateway.stop_latched)
        self.assertEqual(self.service.gateway.state, "Stopped")


if __name__ == "__main__":
    unittest.main()
