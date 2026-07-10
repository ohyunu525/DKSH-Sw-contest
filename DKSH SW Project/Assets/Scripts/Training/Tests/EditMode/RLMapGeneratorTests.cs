using System;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;

namespace DKSH.Spiderbot.Training.Tests
{
    public sealed class RLMapGeneratorTests
    {
        private readonly List<GameObject> cleanupObjects = new List<GameObject>();

        [TearDown]
        public void TearDown()
        {
            for (var i = cleanupObjects.Count - 1; i >= 0; i--)
            {
                if (cleanupObjects[i] != null)
                {
                    UnityEngine.Object.DestroyImmediate(cleanupObjects[i]);
                }
            }

            cleanupObjects.Clear();
        }

        [Test]
        public void Generate_CreatesRequestedEnvironmentsWithSequentialSeedsAndGridLayout()
        {
            var generator = CreateGenerator();
            generator.SelectedLevel = RLMapLevel.Flat;
            generator.MapCount = 5;
            generator.BaseSeed = 1000;
            generator.Spacing = 25f;

            generator.Generate();

            Assert.That(generator.Environments.Count, Is.EqualTo(5));

            var columns = Mathf.CeilToInt(Mathf.Sqrt(generator.MapCount));
            for (var i = 0; i < generator.Environments.Count; i++)
            {
                var environment = generator.Environments[i];
                Assert.That(environment.EnvironmentIndex, Is.EqualTo(i));
                Assert.That(environment.Seed, Is.EqualTo(1000 + i));

                var expectedPosition = new Vector3((i % columns) * generator.Spacing, 0f, (i / columns) * generator.Spacing);
                Assert.That(Vector3.Distance(expectedPosition, environment.transform.localPosition), Is.LessThan(0.0001f));
            }
        }

        [Test]
        public void Generate_AllLevels_CreateValidStartTargetAndPath()
        {
            foreach (RLMapLevel level in Enum.GetValues(typeof(RLMapLevel)))
            {
                var generator = CreateGenerator();
                generator.SelectedLevel = level;
                generator.MapCount = 2;
                generator.BaseSeed = 2200;
                generator.Spacing = 35f;

                generator.Generate();

                Assert.That(generator.Environments.Count, Is.EqualTo(2), level.ToString());
                for (var i = 0; i < generator.Environments.Count; i++)
                {
                    var environment = generator.Environments[i];
                    Assert.That(environment.Level, Is.EqualTo(level), level.ToString());
                    Assert.NotNull(environment.StartPoint, level.ToString());
                    Assert.NotNull(environment.TargetPoint, level.ToString());
                    Assert.False(environment.IsCellUnsafe(environment.StartCell), level.ToString());
                    Assert.False(environment.IsCellUnsafe(environment.TargetCell), level.ToString());
                    Assert.True(environment.HasPathFromStartToTarget(), level.ToString());
                }
            }
        }

        [Test]
        public void Generate_SameSeedAndSettings_AreDeterministic()
        {
            var generator = CreateGenerator();
            generator.SelectedLevel = RLMapLevel.BuildingSpreadingFireRandomCollapse;
            generator.MapCount = 1;
            generator.BaseSeed = 777;
            generator.Spacing = 30f;

            generator.Generate();
            var firstSnapshot = Snapshot(generator.Environments[0]);

            generator.Generate();
            var secondSnapshot = Snapshot(generator.Environments[0]);

            Assert.That(secondSnapshot, Is.EqualTo(firstSnapshot));
        }

        [Test]
        public void Generate_StairMap_ChoosesReachableStartAndKeepsTargetOnUpperPlatform()
        {
            var generator = CreateGenerator();
            generator.SelectedLevel = RLMapLevel.SeededStair;
            generator.MapCount = 1;
            generator.BaseSeed = 3101;

            generator.Generate();

            var environment = generator.Environments[0];
            Assert.That(environment.Level, Is.EqualTo(RLMapLevel.SeededStair));
            Assert.That(environment.TargetPoint.localPosition.y, Is.GreaterThan(0.8f));
            Assert.That(environment.StartCell, Is.Not.EqualTo(environment.TargetCell));
            Assert.False(environment.IsCellUnsafe(environment.StartCell));
            Assert.True(environment.HasPathFromStartToTarget());
            Assert.That(environment.StartPoint.localPosition.y, Is.EqualTo(0.25f).Within(0.001f));
        }

        [Test]
        public void Generate_HillMap_UsesPerlinTerrainAndValidSurfaceMarkers()
        {
            var generator = CreateGenerator();
            generator.SelectedLevel = RLMapLevel.SeededHill;
            generator.MapCount = 1;
            generator.BaseSeed = 4100;

            generator.Generate();

            var environment = generator.Environments[0];
            Assert.NotNull(environment.GeometryRoot.Find("PerlinHillTerrain"));
            Assert.That(environment.StartPoint.localPosition.y, Is.GreaterThan(0.05f));
            Assert.That(environment.TargetPoint.localPosition.y, Is.GreaterThan(0.05f));
        }

        [Test]
        public void Generate_BuildingMap_CreatesLayeredPathAcrossFloors()
        {
            var generator = CreateGenerator();
            generator.SelectedLevel = RLMapLevel.SeededBuilding;
            generator.MapCount = 1;
            generator.BaseSeed = 5100;

            generator.Generate();

            var environment = generator.Environments[0];
            Assert.True(environment.UsesLayeredNavigation);
            Assert.That(environment.FloorCount, Is.GreaterThanOrEqualTo(2));
            Assert.That(environment.StartNode.z, Is.Not.EqualTo(environment.TargetNode.z));
            Assert.True(environment.HasPathFromStartToTarget());
        }

        [Test]
        public void Generate_CollapseLevel_CreatesDeterministicFallingDebrisSchedule()
        {
            var generator = CreateGenerator();
            generator.SelectedLevel = RLMapLevel.BuildingSpreadingFireRandomCollapse;
            generator.MapCount = 1;
            generator.BaseSeed = 6100;

            generator.Generate();
            var firstSchedule = CollapseSchedule(generator.Environments[0]);

            generator.Generate();
            var secondSchedule = CollapseSchedule(generator.Environments[0]);

            Assert.That(firstSchedule, Is.Not.Empty);
            Assert.That(secondSchedule, Is.EqualTo(firstSchedule));
        }

        private RLMapGenerator CreateGenerator()
        {
            var generatorObject = new GameObject("RLMapGeneratorTest");
            cleanupObjects.Add(generatorObject);
            var generator = generatorObject.AddComponent<RLMapGenerator>();
            generator.RegenerateOnPlay = false;
            return generator;
        }

        private static string Snapshot(GeneratedTrainingEnvironment environment)
        {
            return string.Format(
                "{0}->{1}|N:{2}->{3}|O:{4}|H:{5}|C:{6}",
                CellToString(environment.StartCell),
                CellToString(environment.TargetCell),
                NodeToString(environment.StartNode),
                NodeToString(environment.TargetNode),
                CellsToString(environment.ObstacleCells),
                CellsToString(environment.HazardCells),
                CellsToString(environment.CollapseCells));
        }

        private static string CollapseSchedule(GeneratedTrainingEnvironment environment)
        {
            var controller = environment.HazardsRoot.GetComponentInChildren<RLRandomCollapseController>();
            Assert.NotNull(controller);

            var parts = new string[controller.ScheduledDebris.Count];
            for (var i = 0; i < controller.ScheduledDebris.Count; i++)
            {
                var debris = controller.ScheduledDebris[i];
                parts[i] = string.Format("{0}@{1:0.000}", CellToString(debris.Cell), debris.ReleaseTime);
            }

            return string.Join(",", parts);
        }

        private static string CellsToString(IReadOnlyList<Vector2Int> cells)
        {
            var parts = new string[cells.Count];
            for (var i = 0; i < cells.Count; i++)
            {
                parts[i] = CellToString(cells[i]);
            }

            return string.Join(",", parts);
        }

        private static string CellToString(Vector2Int cell)
        {
            return string.Format("{0}:{1}", cell.x, cell.y);
        }

        private static string NodeToString(Vector3Int node)
        {
            return string.Format("{0}:{1}:{2}", node.x, node.y, node.z);
        }
    }
}
