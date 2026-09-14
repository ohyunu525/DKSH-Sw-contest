#if UNITY_EDITOR
using System;
using DKSH.Spiderbot.Exploration;
using DKSH.Spiderbot.Mapping;
using DKSH.Spiderbot.Sensors;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace DKSH.Spiderbot.Editor
{
    public static class FrontierSceneSetup
    {
        private const string LidarScenePath = "Assets/Scenes/LidarScene.unity";
        private const string LidarObjectName = "LiDAR Sensor";
        private const string GridObjectName = "Occupancy Grid 2D";

        [MenuItem("DKSH/Spiderbot/Exploration/Configure LidarScene Frontier Selection")]
        public static void ConfigureLidarScene()
        {
            if (EditorApplication.isPlayingOrWillChangePlaymode || EditorApplication.isCompiling)
            {
                throw new InvalidOperationException(
                    "Wait for Edit Mode compilation before configuring Frontier selection.");
            }

            OccupancyGridSceneSetup.ConfigureLidarScene();
            var sceneWasLoaded = SceneManager.GetSceneByPath(LidarScenePath).isLoaded;
            var scene = sceneWasLoaded
                ? SceneManager.GetSceneByPath(LidarScenePath)
                : EditorSceneManager.OpenScene(LidarScenePath, OpenSceneMode.Additive);

            try
            {
                ConfigureScene(scene);
                EditorSceneManager.MarkSceneDirty(scene);
                EditorSceneManager.SaveScene(scene);
                AssetDatabase.SaveAssets();
                Debug.Log(
                    "DKSH_FRONTIER_SCENE_CONFIGURED: 8-neighbor detection/clustering, " +
                    "minimal A*, 0.25 m clearance, LOS waypoints, and CharacterController following.");
            }
            finally
            {
                if (!sceneWasLoaded && scene.isLoaded)
                {
                    EditorSceneManager.CloseScene(scene, true);
                }
            }
        }

        public static void ConfigureLidarSceneFromCommandLine()
        {
            ConfigureLidarScene();
        }

        [MenuItem("DKSH/Spiderbot/Exploration/Run Frontier Verification")]
        public static void RunVerification()
        {
            RunSimpleRoomTest();
            RunCorridorTest();
            RunForkAndProgressTest();
            RunSmallNoiseTest();
            RunGridPlannerTests();
            RunReachableFrontierSelectionTest();
            RunConfiguredSceneTest();
            Debug.Log(
                "DKSH_FRONTIER_TESTS_PASS: Frontier behavior, A*, clearance, corner blocking, " +
                "reachability fallback, path compression, and standalone scene wiring passed.");
        }

        public static void RunVerificationFromCommandLine()
        {
            RunVerification();
        }

        private static void RunSimpleRoomTest()
        {
            var testObject = new GameObject("FrontierRoomVerification");
            try
            {
                var grid = testObject.AddComponent<OccupancyGrid2D>();
                grid.Configure(1f, 12, 12, Vector3.zero);
                MarkFreeRectangle(grid, 2, 2, 7, 7);

                testObject.transform.position = grid.GridToWorld(4, 4);
                var selector = testObject.AddComponent<FrontierGoalSelector>();
                selector.Configure(grid, null, testObject.transform);
                selector.ConfigureSettings(
                    FrontierConnectivity.Eight,
                    FrontierConnectivity.Eight,
                    3,
                    2f,
                    0.5f,
                    1.5f,
                    0f,
                    0f);

                Require(selector.EvaluateNow(true), "A partially observed room must produce a goal.");
                Require(selector.FrontierCells.Count > 0,
                    "A partially observed room must produce Frontier cells.");
                Require(selector.Clusters.Count == 1,
                    "The connected room boundary must form one Frontier cluster.");

                for (var i = 0; i < selector.FrontierCells.Count; i++)
                {
                    var cell = selector.FrontierCells[i];
                    Require(grid.GetCell(cell.x, cell.y) == OccupancyCellState.Free,
                        "Every Frontier cell must be Free.");
                    Require(FrontierDetector.IsFrontierCell(
                            grid,
                            cell.x,
                            cell.y,
                            FrontierConnectivity.Eight),
                        "Every detected Frontier cell must border an in-bounds Unknown cell.");
                }

                var cluster = selector.SelectedCluster;
                Require(grid.GetCell(selector.SelectedGoalGrid.x, selector.SelectedGoalGrid.y) ==
                        OccupancyCellState.Free,
                    "The representative Frontier goal must be a Free cell.");

                var expectedDistance = HorizontalDistance(
                    testObject.transform.position,
                    selector.SelectedGoalWorld);
                Require(Mathf.Abs(cluster.DistanceFromRobot - expectedDistance) < 0.0001f,
                    "Robot-to-Frontier straight-line distance must be correct.");

                var expectedScore = 2f * cluster.CellCount - 0.5f * expectedDistance;
                Require(Mathf.Abs(cluster.Score - expectedScore) < 0.0001f,
                    "Frontier score must use information gain and distance weights.");
                Require(ReferenceEquals(cluster, FindHighestScore(selector)),
                    "SelectedCluster must be the highest-scoring candidate.");
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(testObject);
            }
        }

        private static void RunCorridorTest()
        {
            var testObject = new GameObject("FrontierCorridorVerification");
            try
            {
                var grid = testObject.AddComponent<OccupancyGrid2D>();
                grid.Configure(1f, 12, 12, Vector3.zero);
                grid.BeginBatchUpdate();
                for (var x = 0; x <= 8; x++)
                {
                    grid.MarkOccupied(x, 4);
                    grid.MarkOccupied(x, 6);
                }

                grid.MarkOccupied(0, 5);
                for (var x = 1; x <= 8; x++)
                {
                    grid.MarkFree(x, 5);
                }

                grid.EndBatchUpdate();

                var detector = new FrontierDetector(grid)
                {
                    Connectivity = FrontierConnectivity.Eight
                };
                var frontiers = detector.DetectFrontierCells();
                Require(frontiers.Count == 1 && frontiers[0] == new Vector2Int(8, 5),
                    "Only the unexplored corridor end must be detected as Frontier.");
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(testObject);
            }
        }

        private static void RunForkAndProgressTest()
        {
            var testObject = new GameObject("FrontierForkVerification");
            try
            {
                var grid = testObject.AddComponent<OccupancyGrid2D>();
                grid.Configure(1f, 15, 15, Vector3.zero);
                BuildForkMap(grid);
                testObject.transform.position = grid.GridToWorld(5, 6);

                var selector = testObject.AddComponent<FrontierGoalSelector>();
                selector.Configure(grid, null, testObject.transform);
                selector.ConfigureSettings(
                    FrontierConnectivity.Eight,
                    FrontierConnectivity.Eight,
                    1,
                    2f,
                    1f,
                    1.5f,
                    100f,
                    0f);

                var goalChangeCount = 0;
                selector.OnFrontierGoalChanged += _ => goalChangeCount++;
                Require(selector.EvaluateNow(true), "A fork must produce a selected goal.");
                Require(selector.Clusters.Count == 2,
                    "Separated left and right fork ends must form two clusters.");
                Require(selector.SelectedGoalGrid == new Vector2Int(3, 6),
                    "The nearer left fork must initially win the score comparison.");
                var heldGoal = selector.SelectedGoalGrid;

                testObject.transform.position = grid.GridToWorld(10, 6);
                selector.EvaluateNow(false);
                Require(selector.SelectedGoalGrid == heldGoal,
                    "A still-valid goal must be held during minimumGoalHoldTime.");

                selector.EvaluateNow(true);
                Require(selector.SelectedGoalGrid == new Vector2Int(11, 6),
                    "Forced operator reevaluation must select the now-nearer right fork.");

                var previousGoal = selector.SelectedGoalGrid;
                BuildProgressedMap(grid);
                testObject.transform.position = grid.GridToWorld(7, 3);
                Require(selector.EvaluateNow(false),
                    "Exploration progress must produce the next valid Frontier goal.");
                Require(selector.SelectedGoalGrid != previousGoal,
                    "An invalidated old Frontier must be replaced by a new goal.");
                Require(!Contains(selector.FrontierCells, previousGoal),
                    "The explored old Frontier must disappear from detection results.");
                Require(goalChangeCount >= 3,
                    "OnFrontierGoalChanged must report initial, forced, and progressed goals.");
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(testObject);
            }
        }

        private static void RunSmallNoiseTest()
        {
            var testObject = new GameObject("FrontierNoiseVerification");
            try
            {
                var grid = testObject.AddComponent<OccupancyGrid2D>();
                grid.Configure(1f, 12, 12, Vector3.zero);
                grid.BeginBatchUpdate();
                for (var y = 0; y < grid.Height; y++)
                {
                    for (var x = 0; x < grid.Width; x++)
                    {
                        if (x != 6 || y != 6)
                        {
                            grid.MarkFree(x, y);
                        }
                    }
                }

                grid.EndBatchUpdate();
                testObject.transform.position = grid.GridToWorld(2, 2);

                var selector = testObject.AddComponent<FrontierGoalSelector>();
                selector.Configure(grid, null, testObject.transform);
                selector.ConfigureSettings(
                    FrontierConnectivity.Eight,
                    FrontierConnectivity.Eight,
                    9,
                    1f,
                    1f,
                    1.5f,
                    0f,
                    0f);

                selector.EvaluateNow(true);
                Require(selector.FrontierCells.Count == 8,
                    "A one-cell Unknown hole must have eight neighboring Frontier cells.");
                Require(selector.Clusters.Count == 0 && !selector.HasValidGoal,
                    "minClusterSize must reject the small-hole Frontier cluster.");
                Require(selector.LastRejectedSmallClusterCount == 1,
                    "The rejected noise cluster must be reported in statistics.");
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(testObject);
            }
        }

        private static void RunGridPlannerTests()
        {
            var testObject = new GameObject("GridAStarVerification");
            try
            {
                var grid = testObject.AddComponent<OccupancyGrid2D>();
                var planner = new GridAStarPlanner();
                var path = new System.Collections.Generic.List<Vector2Int>();

                grid.Configure(1f, 7, 7, Vector3.zero);
                MarkFreeRectangle(grid, 0, 0, 6, 6);
                Require(planner.TryFindPath(
                        grid,
                        new Vector2Int(1, 3),
                        new Vector2Int(5, 3),
                        0f,
                        path),
                    "A straight Free route must be plannable.");
                Require(path.Count == 2 && path[0] == new Vector2Int(1, 3) &&
                        path[1] == new Vector2Int(5, 3),
                    "Line-of-sight compression must reduce a straight route to its endpoints.");

                for (var y = 0; y <= 5; y++)
                {
                    grid.MarkOccupied(3, y);
                }

                Require(planner.TryFindPath(
                        grid,
                        new Vector2Int(1, 3),
                        new Vector2Int(5, 3),
                        0f,
                        path),
                    "A* must route around a wall with an open end.");
                Require(ContainsCellAtOrAbove(path, 6),
                    "The detour must pass through the wall opening.");

                grid.MarkOccupied(3, 6);
                Require(!planner.TryFindPath(
                        grid,
                        new Vector2Int(1, 3),
                        new Vector2Int(5, 3),
                        0f,
                        path),
                    "A completely separating wall must be reported as unreachable.");

                grid.Configure(1f, 3, 3, Vector3.zero);
                MarkFreeRectangle(grid, 0, 0, 2, 2);
                grid.MarkOccupied(1, 0);
                grid.MarkOccupied(0, 1);
                Require(!planner.TryFindPath(
                        grid,
                        new Vector2Int(0, 0),
                        new Vector2Int(1, 1),
                        0f,
                        path),
                    "Diagonal movement must not cut between two blocked orthogonal cells.");

                grid.Configure(1f, 7, 5, Vector3.zero);
                MarkAllOccupied(grid);
                for (var x = 0; x < grid.Width; x++)
                {
                    ForceFree(grid, x, 2);
                }

                Require(planner.TryFindPath(
                        grid,
                        new Vector2Int(0, 2),
                        new Vector2Int(6, 2),
                        0f,
                        path),
                    "The one-cell corridor must be usable without clearance.");
                Require(!planner.TryFindPath(
                        grid,
                        new Vector2Int(0, 2),
                        new Vector2Int(6, 2),
                        1f,
                        path),
                    "Binary obstacle clearance must close a corridor that is too narrow.");
                Require(planner.TryFindPath(
                        grid,
                        new Vector2Int(0, 2),
                        new Vector2Int(0, 2),
                        1f,
                        path),
                    "A Free start cell must remain valid when the robot is already inside inflation.");
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(testObject);
            }
        }

        private static void RunReachableFrontierSelectionTest()
        {
            var testObject = new GameObject("ReachableFrontierVerification");
            try
            {
                var grid = testObject.AddComponent<OccupancyGrid2D>();
                grid.Configure(1f, 12, 8, Vector3.zero);
                grid.BeginBatchUpdate();
                for (var y = 0; y < grid.Height; y++)
                {
                    for (var x = 0; x < grid.Width; x++)
                    {
                        var intentionallyUnknown = (x == 2 && y >= 2 && y <= 4) ||
                                                   (x == 10 && y == 3);
                        if (!intentionallyUnknown)
                        {
                            grid.MarkOccupied(x, y);
                        }
                    }
                }

                for (var y = 2; y <= 4; y++)
                {
                    ForceFree(grid, 3, y);
                }

                for (var x = 6; x <= 9; x++)
                {
                    ForceFree(grid, x, 3);
                }

                grid.EndBatchUpdate();
                testObject.transform.position = grid.GridToWorld(6, 3);

                var selector = testObject.AddComponent<FrontierGoalSelector>();
                selector.Configure(grid, null, testObject.transform);
                selector.ConfigureSettings(
                    FrontierConnectivity.Eight,
                    FrontierConnectivity.Eight,
                    1,
                    10f,
                    1f,
                    1.5f,
                    0f,
                    0f);

                Require(selector.EvaluateNow(true) && selector.SelectedGoalGrid == new Vector2Int(3, 3),
                    "The larger but isolated Frontier must win the original score-only selection.");

                var planner = new GridAStarPlanner();
                var scratchPath = new System.Collections.Generic.List<Vector2Int>();
                selector.SetGoalCellValidator(goal => planner.TryFindPath(
                    grid,
                    new Vector2Int(6, 3),
                    goal,
                    0f,
                    scratchPath));

                Require(selector.EvaluateNow(true),
                    "A reachable fallback Frontier must remain selectable.");
                Require(selector.SelectedGoalGrid == new Vector2Int(9, 3),
                    "Candidates must be path-checked in score order until a reachable goal is found.");
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(testObject);
            }
        }

        private static void RunConfiguredSceneTest()
        {
            ConfigureLidarScene();
            var sceneWasLoaded = SceneManager.GetSceneByPath(LidarScenePath).isLoaded;
            var scene = sceneWasLoaded
                ? SceneManager.GetSceneByPath(LidarScenePath)
                : EditorSceneManager.OpenScene(LidarScenePath, OpenSceneMode.Additive);

            try
            {
                var gridObject = FindInScene(scene, GridObjectName);
                var lidarObject = FindInScene(scene, LidarObjectName);
                var grid = gridObject.GetComponent<OccupancyGrid2D>();
                var mapper = gridObject.GetComponent<LidarOccupancyGridMapper>();
                var selector = gridObject.GetComponent<FrontierGoalSelector>();
                var visualizer = gridObject.GetComponent<FrontierGizmoVisualizer>();
                var sensor = lidarObject.GetComponent<SimulatedLidarSensor>();
                var coordinator = gridObject.GetComponent<FrontierExplorerCoordinator>();
                var follower = lidarObject.GetComponent<FrontierPathFollower>();
                var characterController = lidarObject.GetComponent<CharacterController>();

                Require(selector != null && visualizer != null && coordinator != null &&
                        follower != null && characterController != null,
                    "The configured scene must contain selection, planning, following, and visualization components.");
                Require(selector.Grid == grid && visualizer.GoalSelector == selector &&
                        visualizer.ExplorerCoordinator == coordinator && coordinator.Grid == grid &&
                        coordinator.GoalSelector == selector && coordinator.PathFollower == follower,
                    "Standalone exploration scene references must be connected.");

                grid.ResetGrid();
                Physics.SyncTransforms();
                mapper.ProcessScan(sensor.ScanNow());
                Require(selector.EvaluateNow(true),
                    "The configured partially observed room must produce a Frontier goal.");
                Require(selector.FrontierCells.Count > 0 && selector.Clusters.Count > 0,
                    "The configured room scan must produce visible Frontier data.");
                Require(grid.GetCell(selector.SelectedGoalGrid.x, selector.SelectedGoalGrid.y) ==
                        OccupancyCellState.Free,
                    "The configured scene goal must be directly consumable as a Free grid cell.");
            }
            finally
            {
                var gridObject = FindInScene(scene, GridObjectName);
                if (gridObject != null)
                {
                    gridObject.GetComponent<OccupancyGrid2D>().ResetGrid();
                }

                if (!sceneWasLoaded && scene.isLoaded)
                {
                    EditorSceneManager.CloseScene(scene, true);
                }
            }
        }

        private static void ConfigureScene(Scene scene)
        {
            var gridObject = FindInScene(scene, GridObjectName);
            var lidarObject = FindInScene(scene, LidarObjectName);
            if (gridObject == null || lidarObject == null)
            {
                throw new InvalidOperationException(
                    "Run Occupancy Grid scene setup before Frontier scene setup.");
            }

            var grid = gridObject.GetComponent<OccupancyGrid2D>();
            var mapper = gridObject.GetComponent<LidarOccupancyGridMapper>();
            var selector = GetOrAddComponent<FrontierGoalSelector>(gridObject);
            selector.Configure(grid, mapper, lidarObject.transform);
            selector.ConfigureSettings(
                FrontierConnectivity.Eight,
                FrontierConnectivity.Eight,
                3,
                1f,
                1f,
                1.5f,
                2f,
                1f);

            var capsuleCollider = lidarObject.GetComponent<CapsuleCollider>();
            if (capsuleCollider != null)
            {
                UnityEngine.Object.DestroyImmediate(capsuleCollider);
            }

            var characterController = GetOrAddComponent<CharacterController>(lidarObject);
            characterController.radius = 0.5f;
            characterController.height = 2f;
            characterController.center = Vector3.zero;
            characterController.slopeLimit = 50f;
            characterController.stepOffset = 0.1f;
            characterController.skinWidth = 0.05f;
            characterController.minMoveDistance = 0f;

            var follower = GetOrAddComponent<FrontierPathFollower>(lidarObject);
            var coordinator = GetOrAddComponent<FrontierExplorerCoordinator>(gridObject);
            coordinator.Configure(grid, selector, follower, lidarObject.transform);
            coordinator.ConfigureSettings(0.25f, 0.35f);

            var visualizer = GetOrAddComponent<FrontierGizmoVisualizer>(gridObject);
            visualizer.Configure(selector, coordinator);
            EditorUtility.SetDirty(selector);
            EditorUtility.SetDirty(characterController);
            EditorUtility.SetDirty(follower);
            EditorUtility.SetDirty(coordinator);
            EditorUtility.SetDirty(visualizer);
        }

        private static void MarkAllOccupied(OccupancyGrid2D grid)
        {
            grid.BeginBatchUpdate();
            for (var y = 0; y < grid.Height; y++)
            {
                for (var x = 0; x < grid.Width; x++)
                {
                    grid.MarkOccupied(x, y);
                }
            }

            grid.EndBatchUpdate();
        }

        private static bool ContainsCellAtOrAbove(
            System.Collections.Generic.IReadOnlyList<Vector2Int> cells,
            int minimumY)
        {
            for (var i = 0; i < cells.Count; i++)
            {
                if (cells[i].y >= minimumY)
                {
                    return true;
                }
            }

            return false;
        }

        private static void BuildForkMap(OccupancyGrid2D grid)
        {
            grid.ResetGrid();
            grid.BeginBatchUpdate();
            for (var y = 0; y < grid.Height; y++)
            {
                for (var x = 0; x < grid.Width; x++)
                {
                    if ((x == 2 && y == 6) || (x == 12 && y == 6))
                    {
                        continue;
                    }

                    grid.MarkOccupied(x, y);
                }
            }

            for (var y = 2; y <= 6; y++)
            {
                ForceFree(grid, 7, y);
            }

            for (var x = 3; x <= 6; x++)
            {
                ForceFree(grid, x, 6);
            }

            for (var x = 8; x <= 11; x++)
            {
                ForceFree(grid, x, 6);
            }

            grid.EndBatchUpdate();
        }

        private static void BuildProgressedMap(OccupancyGrid2D grid)
        {
            grid.ResetGrid();
            grid.BeginBatchUpdate();
            for (var y = 0; y < grid.Height; y++)
            {
                for (var x = 0; x < grid.Width; x++)
                {
                    if (x == 7 && y == 12)
                    {
                        continue;
                    }

                    grid.MarkOccupied(x, y);
                }
            }

            for (var y = 2; y <= 11; y++)
            {
                ForceFree(grid, 7, y);
            }

            grid.EndBatchUpdate();
        }

        private static void MarkFreeRectangle(
            OccupancyGrid2D grid,
            int minimumX,
            int minimumY,
            int maximumX,
            int maximumY)
        {
            grid.BeginBatchUpdate();
            for (var y = minimumY; y <= maximumY; y++)
            {
                for (var x = minimumX; x <= maximumX; x++)
                {
                    grid.MarkFree(x, y);
                }
            }

            grid.EndBatchUpdate();
        }

        private static void ForceFree(OccupancyGrid2D grid, int x, int y)
        {
            for (var attempt = 0;
                 attempt < 32 && grid.GetCell(x, y) != OccupancyCellState.Free;
                 attempt++)
            {
                grid.MarkFree(x, y);
            }

            Require(grid.GetCell(x, y) == OccupancyCellState.Free,
                "Test setup could not make the requested cell Free.");
        }

        private static FrontierCluster FindHighestScore(FrontierGoalSelector selector)
        {
            FrontierCluster highest = null;
            for (var i = 0; i < selector.Clusters.Count; i++)
            {
                if (highest == null || selector.Clusters[i].Score > highest.Score)
                {
                    highest = selector.Clusters[i];
                }
            }

            return highest;
        }

        private static bool Contains(System.Collections.Generic.IReadOnlyList<Vector2Int> cells, Vector2Int cell)
        {
            for (var i = 0; i < cells.Count; i++)
            {
                if (cells[i] == cell)
                {
                    return true;
                }
            }

            return false;
        }

        private static float HorizontalDistance(Vector3 first, Vector3 second)
        {
            var delta = first - second;
            delta.y = 0f;
            return delta.magnitude;
        }

        private static GameObject FindInScene(Scene scene, string objectName)
        {
            var roots = scene.GetRootGameObjects();
            for (var i = 0; i < roots.Length; i++)
            {
                var match = FindInHierarchy(roots[i].transform, objectName);
                if (match != null)
                {
                    return match.gameObject;
                }
            }

            return null;
        }

        private static Transform FindInHierarchy(Transform current, string objectName)
        {
            if (current.name == objectName)
            {
                return current;
            }

            for (var i = 0; i < current.childCount; i++)
            {
                var match = FindInHierarchy(current.GetChild(i), objectName);
                if (match != null)
                {
                    return match;
                }
            }

            return null;
        }

        private static T GetOrAddComponent<T>(GameObject gameObject) where T : Component
        {
            var component = gameObject.GetComponent<T>();
            return component != null ? component : gameObject.AddComponent<T>();
        }

        private static void Require(bool condition, string message)
        {
            if (!condition)
            {
                throw new InvalidOperationException("Frontier verification failed: " + message);
            }
        }
    }
}
#endif
