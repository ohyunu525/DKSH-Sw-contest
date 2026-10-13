using System;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;

namespace DKSH.Spiderbot.RemoteNavigation.Tests
{
    public sealed class NavigationTests
    {
        [Test]
        public void RotatedNegativeOriginAndViewportRoundTrip()
        {
            var map = new NavigationMap(30, 12, .2f, new Vector2(-12, 4), .73f, new int[360], "rotated");
            var viewport = new MapViewport { Bounds = new Rect(12, 18, 730, 510), Zoom = 2.4f, Pan = new Vector2(45, -37) };
            var point = map.CellCentre(13, 7);
            Vector2 screen = viewport.MapToScreen(map, point);
            Assert.That(viewport.ScreenToMap(map, screen, out var result), Is.True);
            Assert.That(Vector2.Distance(result, point), Is.LessThan(.0001f));
            Assert.That(map.MapToCell(result), Is.EqualTo(new Vector2Int(13, 7)));
            viewport.ZoomAt(map, screen, .7f);
            Assert.That(Vector2.Distance(viewport.MapToScreen(map, point), screen), Is.LessThan(.001f));
        }

        [Test]
        public void LetterboxDoesNotBecomeSelectableMapSpace()
        {
            var map = new NavigationMap(40, 10, .1f, Vector2.zero, 0, new int[400], "wide");
            var viewport = new MapViewport { Bounds = new Rect(0, 0, 400, 400) };
            Assert.That(viewport.ScreenToMap(map, new Vector2(20, 20), out _), Is.False);
            Assert.That(viewport.ScreenToMap(map, new Vector2(200, 200), out var centre), Is.True);
            Assert.That(Vector2.Distance(centre, new Vector2(2, .5f)), Is.LessThan(.0001f));
        }

        [Test]
        public void UnknownCellsAndObstacleEdgesRequireFootprintClearance()
        {
            var data = new int[100]; data[5 * 10 + 5] = -1; data[5 * 10 + 7] = 100;
            var map = new NavigationMap(10, 10, 1, Vector2.zero, 0, data, "clearance");
            Assert.That(map.Validate(new Vector2(4.8f, 5.5f), .35f, out _), Is.False);
            Assert.That(map.Validate(new Vector2(6.8f, 5.5f), .35f, out _), Is.False);
            Assert.That(map.Validate(new Vector2(.1f, 2), .35f, out _), Is.False);
            Assert.That(map.Validate(new Vector2(3, 3), .35f, out _), Is.True);
        }

        [Test]
        public void RouteAroundWallHasSafeSegmentsAndExactEndpoint()
        {
            var map = NavigationMap.CreateDemo(); var path = new List<Vector2>();
            Assert.That(new WaypointPlanner().Plan(map, new Vector2(-5, -3), new Vector2(5, 1), .35f, path), Is.True);
            Assert.That(path.Count, Is.GreaterThan(2));
            Assert.That(path[path.Count - 1], Is.EqualTo(new Vector2(5, 1)));
            for (int i = 1; i < path.Count; i++) Assert.That(map.SegmentFree(path[i - 1], path[i], .35f), Is.True);
        }

        [Test]
        public void DiagonalGapBetweenObstaclesIsNotAPath()
        {
            var cells = new[] { 0, 100, 100, 0 };
            var map = new NavigationMap(2, 2, 1, Vector2.zero, 0, cells, "corners");
            Assert.That(new WaypointPlanner().Plan(map, map.CellCentre(0, 0), map.CellCentre(1, 1), 0, new List<Vector2>()), Is.False);
        }

        [Test]
        public void InvalidOrStaleMissionNeverMovesRobot()
        {
            using var client = new SimulatedNavigationClient();
            Assert.That(client.Submit(new[] { new Waypoint(new Vector2(100, 100), 0) }, client.Map.Revision), Is.False);
            Assert.That(client.Submit(new[] { new Waypoint(new Vector2(-4, -3), 0) }, "old-map"), Is.False);
            client.Tick(.1f);
            Assert.That(client.Status.x, Is.EqualTo(-5)); Assert.That(client.Status.y, Is.EqualTo(-3));
        }

        [Test]
        public void CancelBeforeFirstTickAndDuringMotionStopsRemainingWaypoints()
        {
            using var client = new SimulatedNavigationClient();
            var goals = new[] { new Waypoint(new Vector2(-4, -3), 0), new Waypoint(new Vector2(-3, -3), 0) };
            Assert.That(client.Submit(goals, client.Map.Revision), Is.True); client.Cancel();
            for (int i = 0; i < 20; i++) client.Tick(.1f);
            Assert.That(client.Status.x, Is.EqualTo(-5)); Assert.That(client.Status.state, Is.EqualTo("Canceled"));
            Assert.That(client.Submit(goals, client.Map.Revision), Is.True);
            for (int i = 0; i < 5; i++) client.Tick(.1f);
            client.Cancel(); float stopped = client.Status.x;
            for (int i = 0; i < 30; i++) client.Tick(.1f);
            Assert.That(client.Status.x, Is.EqualTo(stopped)); Assert.That(client.Status.waypoint_index, Is.Zero);
        }

        [Test]
        public void LinkRestorationDoesNotResumeAndStopNeedsReset()
        {
            using var client = new SimulatedNavigationClient();
            var goal = new[] { new Waypoint(new Vector2(-4, -3), 0) };
            client.Submit(goal, client.Map.Revision); client.Tick(.1f); client.Tick(.1f);
            client.SetLinkAvailable(false); float stopped = client.Status.x;
            client.SetLinkAvailable(true);
            for (int i = 0; i < 40; i++) client.Tick(.1f);
            Assert.That(client.Status.x, Is.EqualTo(stopped)); Assert.That(client.CanSubmit, Is.False);
            client.ResetStop(); Assert.That(client.CanSubmit, Is.True);
            Assert.That(client.Status.x, Is.EqualTo(stopped));
        }

        [Test]
        public void SequentialGoalsReachPositionAndFinalHeading()
        {
            using var client = new SimulatedNavigationClient();
            var goal = new[] { new Waypoint(new Vector2(-4.6f, -3), .4f), new Waypoint(new Vector2(-4.2f, -3), -1.2f) };
            Assert.That(client.Submit(goal, client.Map.Revision), Is.True);
            for (int i = 0; i < 200 && client.Status.state != "Succeeded"; i++) client.Tick(.1f);
            Assert.That(client.Status.state, Is.EqualTo("Succeeded"));
            Assert.That(client.Status.x, Is.EqualTo(-4.2f).Within(.001));
            Assert.That(client.Status.yaw, Is.EqualTo(-1.2f).Within(.001));
            Assert.That(client.Status.waypoint_index, Is.EqualTo(2));
        }

        [Test]
        public void DraftRejectsOverflowAndDropsOldMapCoordinates()
        {
            var editor = new WaypointEditor { AppendMode = true };
            for (int i = 0; i < 10; i++) Assert.That(editor.Add(new Waypoint(new Vector2(i, 0), 0), "first"), Is.True);
            Assert.That(editor.Add(new Waypoint(Vector2.zero, 0), "first"), Is.False);
            editor.Add(new Waypoint(Vector2.one, 0), "second");
            Assert.That(editor.Points.Count, Is.EqualTo(1)); Assert.That(editor.MapRevision, Is.EqualTo("second"));
        }
    }
}
