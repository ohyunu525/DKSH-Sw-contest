using System;
using System.IO;
using NUnit.Framework;
using UnityEngine;

namespace DKSH.Spiderbot.RemoteNavigation.Tests
{
    public sealed class MissionDraftTests
    {
        private string path;
        private NavigationMap map;
        private WaypointEditor editor;
        private readonly Waypoint[] goals =
        {
            new Waypoint(new Vector2(-4, -3), .4f),
            new Waypoint(new Vector2(-3, -3), -1.2f)
        };

        [SetUp]
        public void SetUp()
        {
            path = Path.Combine(Application.temporaryCachePath, "mission-draft-test-" + Guid.NewGuid().ToString("N") + ".json");
            map = NavigationMap.CreateDemo(); editor = new WaypointEditor();
            editor.Add(new Waypoint(new Vector2(-5, -3), 0), map.Revision);
        }

        [TearDown]
        public void TearDown()
        {
            if (File.Exists(path)) File.Delete(path);
            if (File.Exists(path + ".tmp")) File.Delete(path + ".tmp");
        }

        [Test]
        public void SaveLoadKeepsOrderAndHeadingWithoutSendingAMission()
        {
            using var client = new SimulatedNavigationClient();
            Assert.That(MissionDraftStore.Save(path, map, goals, .35f, out var reason), Is.True, reason);
            Assert.That(MissionDraftStore.Load(path, client.Map, .35f, editor, out reason), Is.True, reason);
            Assert.That(editor.Points.Count, Is.EqualTo(2));
            Assert.That(editor.Points[0].Position, Is.EqualTo(goals[0].Position));
            Assert.That(editor.Points[1].yaw, Is.EqualTo(goals[1].yaw).Within(.0001));
            Assert.That(editor.MapRevision, Is.EqualTo(map.Revision));
            client.Tick(.1f);
            Assert.That(client.Status.state, Is.EqualTo("Idle"));
            Assert.That(client.Status.x, Is.EqualTo(-5));
        }

        [Test]
        public void SaveReplacesPreviousDraftAndDoesNotPersistSessionSecrets()
        {
            Assert.That(MissionDraftStore.Save(path, map, goals, .35f, out _), Is.True);
            Assert.That(MissionDraftStore.Save(path, map, new[] { goals[1] }, .35f, out var reason), Is.True, reason);
            Assert.That(MissionDraftStore.Load(path, map, .35f, editor, out reason), Is.True, reason);
            Assert.That(editor.Points.Count, Is.EqualTo(1));
            Assert.That(editor.Points[0].Position, Is.EqualTo(goals[1].Position));
            string json = File.ReadAllText(path);
            Assert.That(json, Does.Not.Contain("token").And.Not.Contain("session").And.Not.Contain("mission_id"));
            Assert.That(File.Exists(path + ".tmp"), Is.False);
        }

        [TestCase(false)]
        [TestCase(true)]
        public void ChangedMapIdentityOrGeometryPreservesExistingDraft(bool geometryOnly)
        {
            Assert.That(MissionDraftStore.Save(path, map, goals, .35f, out _), Is.True);
            var changed = new NavigationMap(map.Width, map.Height, map.Resolution,
                geometryOnly ? map.Origin + Vector2.one : map.Origin, map.OriginYaw, map.CopyCells(),
                geometryOnly ? map.Revision : "new-map-session");
            Assert.That(MissionDraftStore.Load(path, changed, .35f, editor, out _), Is.False);
            Assert.That(editor.Points.Count, Is.EqualTo(1));
            Assert.That(editor.Points[0].Position, Is.EqualTo(new Vector2(-5, -3)));
        }

        [Test]
        public void UpdatedOccupancyRevalidatesEveryWaypointBeforeReplacingDraft()
        {
            Assert.That(MissionDraftStore.Save(path, map, goals, .35f, out _), Is.True);
            var cells = map.CopyCells(); var blocked = map.MapToCell(goals[1].Position);
            cells[blocked.y * map.Width + blocked.x] = 100;
            var changed = new NavigationMap(map.Width, map.Height, map.Resolution, map.Origin, map.OriginYaw, cells, map.Revision);
            Assert.That(MissionDraftStore.Load(path, changed, .35f, editor, out _), Is.False);
            Assert.That(editor.Points.Count, Is.EqualTo(1));
            Assert.That(editor.Points[0].Position, Is.EqualTo(new Vector2(-5, -3)));
        }

        [TestCase("")]
        [TestCase("{ broken json")]
        [TestCase("{\"version\":99}")]
        [TestCase("null")]
        public void InvalidFilePreservesExistingDraft(string json)
        {
            File.WriteAllText(path, json);
            Assert.That(MissionDraftStore.Load(path, map, .35f, editor, out _), Is.False);
            Assert.That(editor.Points.Count, Is.EqualTo(1));
        }

        [Test]
        public void MissingOrOversizedFilePreservesExistingDraft()
        {
            Assert.That(MissionDraftStore.Load(path, map, .35f, editor, out _), Is.False);
            File.WriteAllText(path, new string(' ', MissionDraftStore.MaximumBytes + 1));
            Assert.That(MissionDraftStore.Load(path, map, .35f, editor, out _), Is.False);
            Assert.That(editor.Points.Count, Is.EqualTo(1));
        }

        [Test]
        public void OversizedSaveCannotReplaceAnExistingFile()
        {
            Assert.That(MissionDraftStore.Save(path, map, goals, .35f, out _), Is.True);
            string previous = File.ReadAllText(path);
            var oversized = new NavigationMap(map.Width, map.Height, map.Resolution, map.Origin,
                map.OriginYaw, map.CopyCells(), new string('x', MissionDraftStore.MaximumBytes));
            Assert.That(MissionDraftStore.Save(path, oversized, goals, .35f, out _), Is.False);
            Assert.That(File.ReadAllText(path), Is.EqualTo(previous));
        }

        [Test]
        public void InvalidSavePreservesPreviouslySavedDraft()
        {
            Assert.That(MissionDraftStore.Save(path, map, goals, .35f, out _), Is.True);
            string previous = File.ReadAllText(path);
            var invalid = new Waypoint { x = float.NaN, y = -3, yaw = 0 };
            Assert.That(MissionDraftStore.Save(path, map, new[] { invalid }, .35f, out _), Is.False);
            Assert.That(File.ReadAllText(path), Is.EqualTo(previous));
            Assert.That(editor.Replace(new[] { goals[0], null }, map, .35f, out _), Is.False);
            Assert.That(editor.Replace(new Waypoint[11], map, .35f, out _), Is.False);
            Assert.That(editor.Points.Count, Is.EqualTo(1));
        }
    }
}
