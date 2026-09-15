#if UNITY_EDITOR
using System;
using DKSH.Spiderbot.Mapping;
using DKSH.Spiderbot.Sensors;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace DKSH.Spiderbot.Editor
{
    public static class OccupancyGridSceneSetup
    {
        private const string LidarScenePath = "Assets/Scenes/LidarScene.unity";
        private const string LidarObjectName = "LiDAR Sensor";
        private const string GridObjectName = "Occupancy Grid 2D";
        private const string RobotLayerName = "Robot";

        [MenuItem("DKSH/Spiderbot/Mapping/Configure LidarScene Occupancy Grid")]
        public static void ConfigureLidarScene()
        {
            if (EditorApplication.isPlayingOrWillChangePlaymode || EditorApplication.isCompiling)
            {
                throw new InvalidOperationException("Wait for Edit Mode compilation before configuring the scene.");
            }

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
                    "DKSH_OCCUPANCY_GRID_SCENE_CONFIGURED: X-Z grid, 0.1 m resolution, " +
                    "120x120 cells, Robot layer excluded from LiDAR.");
            }
            finally
            {
                if (!sceneWasLoaded && scene.isLoaded)
                {
                    EditorSceneManager.CloseScene(scene, true);
                }
            }
        }

        [MenuItem("DKSH/Spiderbot/Mapping/Assign Robot Layer To Selection")]
        public static void AssignRobotLayerToSelection()
        {
            if (Selection.activeGameObject == null)
            {
                throw new InvalidOperationException("Select the robot root GameObject first.");
            }

            var robotLayer = EnsureLayer(RobotLayerName);
            SetLayerRecursively(Selection.activeGameObject, robotLayer);

            var sensors = Selection.activeGameObject.GetComponentsInChildren<SimulatedLidarSensor>(true);
            for (var i = 0; i < sensors.Length; i++)
            {
                ExcludeLayerFromSensor(sensors[i], robotLayer);
                EditorUtility.SetDirty(sensors[i]);
            }

            EditorUtility.SetDirty(Selection.activeGameObject);
            Debug.LogFormat(
                Selection.activeGameObject,
                "Assigned Robot layer to {0} and excluded it from {1} LiDAR sensor(s).",
                Selection.activeGameObject.name,
                sensors.Length);
        }

        public static void ConfigureLidarSceneFromCommandLine()
        {
            ConfigureLidarScene();
        }

        [MenuItem("DKSH/Spiderbot/Mapping/Run Occupancy Grid Verification")]
        public static void RunVerification()
        {
            RunCoreDataAndRayTests();
            RunConfiguredSceneScanTest();
            Debug.Log("DKSH_OCCUPANCY_GRID_TESTS_PASS: coordinate, state, ray, scene, and self-filter checks passed.");
        }

        public static void RunVerificationFromCommandLine()
        {
            RunVerification();
        }

        private static void RunCoreDataAndRayTests()
        {
            var testObject = new GameObject("OccupancyGridVerification");
            try
            {
                var grid = testObject.AddComponent<OccupancyGrid2D>();
                grid.Configure(1f, 10, 10, Vector3.zero);
                grid.ConfigureEvidence(3, 4, 1, -20, 20);

                Require(grid.UnknownCellCount == 100, "A reset grid must start completely Unknown.");
                Require(grid.WorldToGrid(new Vector3(2.25f, 7f, 3.75f), out var x, out var y),
                    "An in-bounds world point must convert to a grid cell.");
                Require(x == 2 && y == 3, "WorldToGrid must use the Unity X-Z plane.");
                Require(Vector3.Distance(grid.GridToWorld(2, 3), new Vector3(2.5f, 0f, 3.5f)) < 0.0001f,
                    "GridToWorld must return the cell center.");
                Require(!grid.WorldToGrid(new Vector3(-0.01f, 0f, 0f), out _, out _),
                    "Points below the map origin must be outside the grid.");

                grid.MarkFree(1, 1);
                Require(grid.GetCell(1, 1) == OccupancyCellState.Free,
                    "A traversed cell must become Free.");
                grid.MarkOccupied(2, 2);
                Require(grid.GetCell(2, 2) == OccupancyCellState.Occupied,
                    "A hit cell must become Occupied.");

                grid.ResetGrid();
                var mapper = testObject.AddComponent<LidarOccupancyGridMapper>();
                mapper.Configure(null, grid, null);

                var hitSample = new LidarSample(
                    0,
                    0,
                    new Vector3(1.5f, 0f, 1.5f),
                    Vector3.forward,
                    new Vector3(1.5f, 0f, 5.5f),
                    Vector3.back,
                    4f,
                    true,
                    123);
                var hitFrame = new LidarScanFrame(
                    1.0,
                    hitSample.origin,
                    Quaternion.identity,
                    Quaternion.identity,
                    1,
                    1,
                    1,
                    new[] { hitSample });

                Require(mapper.ProcessScan(hitFrame), "A valid hit scan must be integrated.");
                for (var cellY = 1; cellY < 5; cellY++)
                {
                    Require(grid.GetCell(1, cellY) == OccupancyCellState.Free,
                        "Bresenham traversal cells before a hit must be Free.");
                }

                Require(grid.GetCell(1, 5) == OccupancyCellState.Occupied,
                    "The hit endpoint must be Occupied.");
                Require(grid.GetCell(9, 9) == OccupancyCellState.Unknown,
                    "Unobserved cells must remain Unknown.");

                grid.ResetGrid();
                var missSample = new LidarSample(
                    0,
                    0,
                    new Vector3(1.5f, 0f, 1.5f),
                    Vector3.forward,
                    new Vector3(1.5f, 0f, 4.5f),
                    Vector3.zero,
                    3f,
                    false,
                    0);
                var missFrame = new LidarScanFrame(
                    2.0,
                    missSample.origin,
                    Quaternion.identity,
                    Quaternion.identity,
                    1,
                    1,
                    0,
                    new[] { missSample });

                Require(mapper.ProcessScan(missFrame), "A valid miss scan must be integrated.");
                Require(grid.GetCell(1, 4) == OccupancyCellState.Free,
                    "A miss ray must mark its maximum-range endpoint Free.");
                Require(grid.OccupiedCellCount == 0, "A miss ray must not create an Occupied cell.");
                Require(mapper.UsesKnownPoseFallback,
                    "The current stage must explicitly report use of the known-pose fallback.");
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(testObject);
            }
        }

        private static void RunConfiguredSceneScanTest()
        {
            ConfigureLidarScene();
            var sceneWasLoaded = SceneManager.GetSceneByPath(LidarScenePath).isLoaded;
            var scene = sceneWasLoaded
                ? SceneManager.GetSceneByPath(LidarScenePath)
                : EditorSceneManager.OpenScene(LidarScenePath, OpenSceneMode.Additive);

            try
            {
                var lidarObject = FindInScene(scene, LidarObjectName);
                var gridObject = FindInScene(scene, GridObjectName);
                var sensor = lidarObject.GetComponent<SimulatedLidarSensor>();
                var grid = gridObject.GetComponent<OccupancyGrid2D>();
                var mapper = gridObject.GetComponent<LidarOccupancyGridMapper>();
                var robotLayer = LayerMask.NameToLayer(RobotLayerName);

                Require(robotLayer >= 0, "Robot layer must exist.");
                Require(lidarObject.layer == robotLayer, "The LiDAR robot surrogate must use the Robot layer.");
                Require((sensor.ActiveSettings.detectionMask.value & (1 << robotLayer)) == 0,
                    "The LiDAR detection mask must exclude the Robot layer.");

                var originalPosition = lidarObject.transform.position;
                var originalRotation = lidarObject.transform.rotation;
                try
                {
                    grid.ResetGrid();
                    Physics.SyncTransforms();
                    mapper.ProcessScan(sensor.ScanNow());

                    Require(grid.FreeCellCount > 0, "The configured room scan must produce Free cells.");
                    Require(grid.OccupiedCellCount > 0, "The configured room scan must produce Occupied cells.");
                    Require(grid.UnknownCellCount > 0, "Occluded or unobserved cells must remain Unknown.");
                    var firstObservedCount = grid.FreeCellCount + grid.OccupiedCellCount;

                    lidarObject.transform.position = originalPosition + Vector3.right;
                    lidarObject.transform.rotation = Quaternion.Euler(0f, 45f, 0f) * originalRotation;
                    Physics.SyncTransforms();
                    mapper.ProcessScan(sensor.ScanNow());

                    var movedObservedCount = grid.FreeCellCount + grid.OccupiedCellCount;
                    Require(movedObservedCount >= firstObservedCount,
                        "Moving and rotating the known-pose sensor must not reduce observed map coverage.");
                    Require(mapper.LastIntegratedRayCount > 0,
                        "The scene mapper must integrate at least one near-horizontal LiDAR ray.");
                }
                finally
                {
                    lidarObject.transform.SetPositionAndRotation(originalPosition, originalRotation);
                    Physics.SyncTransforms();
                    grid.ResetGrid();
                }
            }
            finally
            {
                if (!sceneWasLoaded && scene.isLoaded)
                {
                    EditorSceneManager.CloseScene(scene, true);
                }
            }
        }

        private static void ConfigureScene(Scene scene)
        {
            var robotLayer = EnsureLayer(RobotLayerName);
            var lidarObject = FindInScene(scene, LidarObjectName);
            if (lidarObject == null)
            {
                throw new InvalidOperationException(LidarObjectName + " was not found in " + LidarScenePath + ".");
            }

            var lidarSensor = lidarObject.GetComponent<SimulatedLidarSensor>();
            if (lidarSensor == null)
            {
                throw new InvalidOperationException(LidarObjectName + " does not contain SimulatedLidarSensor.");
            }

            SetLayerRecursively(lidarObject, robotLayer);
            ExcludeLayerFromSensor(lidarSensor, robotLayer);

            var gridObject = FindInScene(scene, GridObjectName);
            if (gridObject == null)
            {
                gridObject = new GameObject(GridObjectName);
                SceneManager.MoveGameObjectToScene(gridObject, scene);
            }

            gridObject.transform.SetPositionAndRotation(Vector3.zero, Quaternion.identity);
            gridObject.transform.localScale = Vector3.one;

            var grid = GetOrAddComponent<OccupancyGrid2D>(gridObject);
            grid.Configure(0.1f, 120, 120, new Vector3(-6f, 0.02f, -6f));
            grid.ConfigureEvidence(3, 4, 1, -20, 20);

            var mapper = GetOrAddComponent<LidarOccupancyGridMapper>(gridObject);
            mapper.Configure(lidarSensor, grid, null);

            var renderer = GetOrAddComponent<OccupancyGrid2DRenderer>(gridObject);
            renderer.Configure(grid, lidarObject.transform);

            EditorUtility.SetDirty(lidarSensor);
            EditorUtility.SetDirty(grid);
            EditorUtility.SetDirty(mapper);
            EditorUtility.SetDirty(renderer);
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


        private static int EnsureLayer(string layerName)
        {
            var existingLayer = LayerMask.NameToLayer(layerName);
            if (existingLayer >= 0)
            {
                return existingLayer;
            }

            var tagManagerAssets = AssetDatabase.LoadAllAssetsAtPath("ProjectSettings/TagManager.asset");
            if (tagManagerAssets == null || tagManagerAssets.Length == 0)
            {
                throw new InvalidOperationException("Could not load ProjectSettings/TagManager.asset.");
            }

            var tagManager = new SerializedObject(tagManagerAssets[0]);
            var layers = tagManager.FindProperty("layers");
            for (var layer = 8; layer < 32; layer++)
            {
                var property = layers.GetArrayElementAtIndex(layer);
                if (!string.IsNullOrEmpty(property.stringValue))
                {
                    continue;
                }

                property.stringValue = layerName;
                tagManager.ApplyModifiedPropertiesWithoutUndo();
                AssetDatabase.SaveAssets();
                return layer;
            }

            throw new InvalidOperationException("No empty user layer is available for " + layerName + ".");
        }

        private static void ExcludeLayerFromSensor(SimulatedLidarSensor sensor, int excludedLayer)
        {
            var settings = sensor.ActiveSettings;
            settings.detectionMask = settings.detectionMask.value & ~(1 << excludedLayer);
            sensor.Configure(settings, true);
        }

        private static void SetLayerRecursively(GameObject root, int layer)
        {
            root.layer = layer;
            for (var i = 0; i < root.transform.childCount; i++)
            {
                SetLayerRecursively(root.transform.GetChild(i).gameObject, layer);
            }
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
                throw new InvalidOperationException("Occupancy grid verification failed: " + message);
            }
        }
    }
}
#endif
