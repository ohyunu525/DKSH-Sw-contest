"""Action lifecycle tests using fake futures; a ROS runtime is not needed."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dksh_remote_navigation.nav2_backend import Nav2SimulationBackend


class Future:
    def __init__(self, result=None, error=None):
        self.value, self.error, self.callback = result, error, None
    def result(self):
        if self.error:
            raise self.error
        return self.value
    def add_done_callback(self, callback):
        self.callback = callback


class Handle:
    accepted = True
    def __init__(self):
        self.cancels, self.future = 0, Future()
    def cancel_goal_async(self):
        self.cancels += 1
        return Future()
    def get_result_async(self):
        return self.future


class Nav2CallbackTests(unittest.TestCase):
    def setUp(self):
        self.backend = Nav2SimulationBackend.__new__(Nav2SimulationBackend)
        self.backend.generation = 4
        self.backend.pending, self.backend.handle, self.backend.canceling = True, None, False
        self.backend.state, self.backend.path, self.backend.distance_remaining = "Accepted", [], 0
        self.sim_now = 2.
        self.backend.node = SimpleNamespace(get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(nanoseconds=int(self.sim_now * 1e9))))

    def test_cancel_while_acceptance_is_pending_cancels_late_handle(self):
        self.backend.cancel()
        handle = Handle()
        self.backend.accepted(4, Future(handle))
        self.assertEqual(handle.cancels, 1)
        self.assertIs(self.backend.handle, handle)
        handle.future.value = SimpleNamespace(status=5, result=SimpleNamespace(error_code=0))
        handle.future.callback(handle.future)
        self.assertEqual(self.backend.state, "Canceled")
        self.assertIsNone(self.backend.handle)

    def test_old_feedback_and_results_cannot_overwrite_new_goal(self):
        handle = Handle()
        self.backend.handle = handle
        self.backend.feedback(3, SimpleNamespace(feedback=SimpleNamespace(distance_remaining=88)))
        self.backend.finished(3, Future(SimpleNamespace(status=4, result=SimpleNamespace(error_code=0))))
        self.assertIs(self.backend.handle, handle)
        self.assertEqual(self.backend.state, "Accepted")
        self.assertEqual(self.backend.distance_remaining, 0)

    def test_feedback_after_completed_result_cannot_revive_goal(self):
        self.backend.finished(4, Future(SimpleNamespace(status=4, result=SimpleNamespace(error_code=0))))
        self.backend.feedback(4, SimpleNamespace(feedback=SimpleNamespace(distance_remaining=88)))
        self.assertEqual(self.backend.state, "Succeeded")
        self.assertEqual(self.backend.distance_remaining, 0)
        self.assertIsNone(self.backend.handle)

    def test_old_accepted_handle_is_canceled(self):
        handle = Handle()
        self.backend.accepted(3, Future(handle))
        self.assertEqual(handle.cancels, 1)
        self.assertIsNone(self.backend.handle)

    def test_old_acceptance_exception_cannot_fail_current_goal(self):
        self.backend.pending = False
        current = Handle()
        self.backend.handle, self.backend.state = current, "Navigating"
        self.backend.accepted(3, Future(error=RuntimeError("Old request failed")))
        self.assertFalse(self.backend.pending)
        self.assertIs(self.backend.handle, current)
        self.assertEqual(self.backend.state, "Navigating")

    def test_old_handle_cancel_exception_cannot_fail_current_goal(self):
        old = Handle()
        def unavailable():
            raise RuntimeError("Old cancel unavailable")
        old.cancel_goal_async = unavailable
        self.backend.accepted(3, Future(old))
        self.assertIsNone(self.backend.handle)
        self.assertEqual(self.backend.state, "Accepted")

    def test_nonfinite_feedback_cancels_current_goal(self):
        handle = Handle()
        self.backend.handle = handle
        self.backend.feedback(4, SimpleNamespace(feedback=SimpleNamespace(distance_remaining=float("nan"))))
        self.assertEqual(handle.cancels, 1)
        self.assertTrue(self.backend.canceling)
        self.assertEqual(self.backend.state, "Failed")
        self.assertEqual(self.backend.distance_remaining, 0)

    def test_lost_result_does_not_confirm_stopped_motion(self):
        handle = Handle()
        self.backend.handle = handle
        self.backend.finished(4, Future(error=RuntimeError("Lost result")))
        self.assertIs(self.backend.handle, handle)
        self.assertEqual(self.backend.state, "Failed")

    def test_already_completed_goal_remains_succeeded_after_acceptance(self):
        handle = Handle()
        handle.future.value = SimpleNamespace(status=4, result=SimpleNamespace(error_code=0))
        handle.future.add_done_callback = lambda callback: callback(handle.future)
        self.backend.accepted(4, Future(handle))
        self.assertEqual(self.backend.state, "Succeeded")
        self.assertIsNone(self.backend.handle)
        self.assertFalse(self.backend.pending)

    def test_lost_acceptance_retains_pending_goal_uncertainty(self):
        self.backend.accepted(4, Future(error=RuntimeError("Lost acceptance")))
        self.assertTrue(self.backend.pending)
        self.assertEqual(self.backend.state, "Failed")

    def test_simulation_clock_rollback_invalidates_map_identity_and_inputs(self):
        self.backend.last_clock, self.backend.epoch = (100, 0), "old-epoch"
        self.backend.map = object()
        self.backend.on_clock(SimpleNamespace(clock=SimpleNamespace(sec=1, nanosec=0)))
        self.assertIsNone(self.backend.map)
        self.assertNotEqual(self.backend.epoch, "old-epoch")
        self.assertEqual(self.backend.pose_at, float("-inf"))
        self.assertIsNone(self.backend.still_since)

    def test_odom_gap_requires_new_stillness_window(self):
        self.backend.pending = False
        self.backend.last_odom, self.backend.odom_at, self.backend.still_since = (1, 0), 100., 99.
        msg = SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=2, nanosec=0)),
                              twist=SimpleNamespace(twist=SimpleNamespace(
                                  linear=SimpleNamespace(x=0., y=0.), angular=SimpleNamespace(z=0.))))
        with patch("dksh_remote_navigation.nav2_backend.time.monotonic", return_value=101.):
            self.backend.on_odom(msg)
            self.assertFalse(self.backend.motion_stopped)
            self.assertEqual(self.backend.still_since, 101.)
        msg.header.stamp.sec = 3
        self.sim_now = 3.
        with patch("dksh_remote_navigation.nav2_backend.time.monotonic", return_value=101.4):
            self.backend.on_odom(msg)
            self.assertTrue(self.backend.motion_stopped)

    def test_queued_or_future_scan_does_not_refresh_sensor_readiness(self):
        self.backend.last_scan, self.backend.scan_at = (1, 0), 100.
        self.sim_now = 10.
        for sec in (2, 11):
            self.backend.on_scan(SimpleNamespace(
                header=SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=0)), ranges=[1.]))
            self.assertEqual(self.backend.scan_at, 100.)
            self.assertEqual(self.backend.last_scan, (1, 0))

    def test_current_scan_refreshes_sensor_readiness(self):
        self.backend.last_scan, self.backend.scan_at = (1, 0), 100.
        msg = SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=2, nanosec=0)), ranges=[1.])
        with patch("dksh_remote_navigation.nav2_backend.time.monotonic", return_value=101.):
            self.backend.on_scan(msg)
        self.assertEqual(self.backend.scan_at, 101.)
        self.assertEqual(self.backend.last_scan, (2, 0))

    def test_queued_or_future_odom_cannot_confirm_stopped_motion(self):
        self.backend.pending = False
        self.backend.last_odom, self.backend.odom_at, self.backend.still_since = (1, 0), 100., 99.
        self.sim_now = 10.
        for sec in (2, 11):
            msg = SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=0)),
                                  twist=SimpleNamespace(twist=SimpleNamespace(
                                      linear=SimpleNamespace(x=0., y=0.), angular=SimpleNamespace(z=0.))))
            with patch("dksh_remote_navigation.nav2_backend.time.monotonic", return_value=101.):
                self.backend.on_odom(msg)
                self.assertFalse(self.backend.motion_stopped)
            self.assertEqual(self.backend.odom_at, 100.)

    def test_out_of_order_sensor_samples_do_not_refresh_freshness(self):
        self.backend.last_scan, self.backend.scan_at = (2, 0), 100.
        self.backend.last_odom, self.backend.odom_at = (2, 0), 100.
        self.sim_now = 2.2
        older = SimpleNamespace(sec=1, nanosec=900000000)
        self.backend.on_scan(SimpleNamespace(header=SimpleNamespace(stamp=older), ranges=[1.]))
        self.backend.on_odom(SimpleNamespace(header=SimpleNamespace(stamp=older)))
        self.assertEqual(self.backend.scan_at, 100.)
        self.assertEqual(self.backend.odom_at, 100.)
