#if UNITY_EDITOR
using System;
using DKSH.Spiderbot.Exploration;
using DKSH.Spiderbot.ROS;
using DKSH.Spiderbot.Sensors;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace DKSH.Spiderbot.Editor
{
    public static class Ros2SceneSetup
    {
        private const string LidarScenePath = "Assets/Scenes/LidarScene.unity";
        private const string LidarObjectName = "LiDAR Sensor";
        private const string GridObjectName = "Occupancy Grid 2D";

        [MenuItem("DKSH/Spiderbot/ROS2/Configure Disabled LidarScene Bridge")]
        public static void ConfigureLidarSceneBridge()
        {
            if (EditorApplication.isPlayingOrWillChangePlaymode || EditorApplication.isCompiling)
            {
                throw new InvalidOperationException(
                    "Wait for Edit Mode compilation before configuring the ROS2 bridge.");
            }

            var sceneWasLoaded = SceneManager.GetSceneByPath(LidarScenePath).isLoaded;
            var scene = sceneWasLoaded
                ? SceneManager.GetSceneByPath(LidarScenePath)
                : EditorSceneManager.OpenScene(LidarScenePath, OpenSceneMode.Additive);

            try
            {
                var lidarObject = FindInScene(scene, LidarObjectName);
                var gridObject = FindInScene(scene, GridObjectName);
                if (lidarObject == null || gridObject == null)
                {
                    throw new InvalidOperationException(
                        "Configure the occupancy grid and Frontier scene before the ROS2 bridge.");
                }

                var sensor = lidarObject.GetComponent<SimulatedLidarSensor>();
                var follower = lidarObject.GetComponent<FrontierPathFollower>();
                var coordinator = gridObject.GetComponent<FrontierExplorerCoordinator>();
                if (sensor == null || follower == null || coordinator == null)
                {
                    throw new InvalidOperationException(
                        "The ROS2 bridge requires the existing LiDAR and standalone Frontier loop.");
                }

                var adapter = GetOrAddComponent<HexapodVelocityCommandAdapter>(lidarObject);
                adapter.ConfigureStandaloneMode(follower, coordinator);
                adapter.ConfigureLimits(0.35f, 0.2f, 0.25f, 0.8f, 0.5f);

                var bridge = GetOrAddComponent<Ros2UnityBridge>(lidarObject);
                bridge.Configure(sensor, lidarObject.transform, adapter, "127.0.0.1", 10000, 10f);

                // Standalone Unity exploration remains the default. Enabling the bridge
                // in the Inspector atomically hands motion ownership to ROS2.
                bridge.enabled = false;
                adapter.enabled = false;

                EditorUtility.SetDirty(adapter);
                EditorUtility.SetDirty(bridge);
                EditorSceneManager.MarkSceneDirty(scene);
                EditorSceneManager.SaveScene(scene);
                AssetDatabase.SaveAssets();
                Debug.Log(
                    "DKSH_ROS2_BRIDGE_CONFIGURED_DISABLED: enable Ros2UnityBridge and " +
                    "HexapodVelocityCommandAdapter only for ROS2/Nav2 validation.");
            }
            finally
            {
                if (!sceneWasLoaded && scene.isLoaded)
                {
                    EditorSceneManager.CloseScene(scene, true);
                }
            }
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
    }
}
#endif
