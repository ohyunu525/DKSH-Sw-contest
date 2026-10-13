import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from dksh_remote_navigation.core import MissionGateway
from dksh_remote_navigation.journal import EventJournal


class JournalTests(unittest.TestCase):
    def test_mission_state_records_exclude_transport_credentials(self):
        with TemporaryDirectory() as directory:
            journal = EventJournal(directory)
            core = MissionGateway(journal=journal)
            session = core.acquire()
            challenge = core.issue_challenge()
            packet = {"v": 1, "type": "command", "command": "submit", "session": session,
                      "token": "never-record-this-token", "challenge": challenge, "seq": 1,
                      "request_id": "request", "mission_id": "journal-mission",
                      "map_revision": core.backend.map.revision,
                      "waypoints": [{"x": -4.9, "y": -3., "yaw": .2}]}
            self.assertTrue(core.handle(packet))
            for _ in range(20):
                core.tick(.1)
            core.disconnect(session)
            journal.close()
            text = (Path(directory) / "events.jsonl").read_text(encoding="utf-8")
            for secret in (session, challenge, packet["token"]):
                self.assertNotIn(secret, text)
            records = [json.loads(line) for line in text.splitlines()]
            mission = next(r for r in records if r["event"] == "mission_accepted")
            self.assertEqual(len(mission["waypoints"]), 1)
            for key in ("x", "y", "yaw"):
                self.assertAlmostEqual(mission["waypoints"][0][key], packet["waypoints"][0][key])
            success = next(r for r in records if r.get("state") == "Succeeded")
            self.assertAlmostEqual(success["x"], -4.9)
            self.assertEqual(success["waypoint_index"], 1)
            self.assertTrue(records[-2]["motion_stopped"])

    def test_rotation_bounds_log_storage(self):
        with TemporaryDirectory() as directory:
            journal = EventJournal(directory, max_bytes=300, backups=2)
            for index in range(40):
                journal("state", index=index, detail="demo" * 15)
            journal.close()
            paths = list(Path(directory).glob("events.jsonl*"))
            self.assertEqual(len(paths), 3)
            for path in paths:
                self.assertLess(path.stat().st_size, 500)
                for line in path.read_text(encoding="utf-8").splitlines():
                    self.assertIn("utc", json.loads(line))
