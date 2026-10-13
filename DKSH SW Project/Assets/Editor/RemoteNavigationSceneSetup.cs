#if UNITY_EDITOR
using System;
using DKSH.Spiderbot.RemoteNavigation;
using DKSH.Spiderbot.Exploration;
using DKSH.Spiderbot.ROS;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UIElements;

namespace DKSH.Spiderbot.Editor
{
    public static class RemoteNavigationSceneSetup
    {
        public const string ScenePath = "Assets/Scenes/RemoteNavigation.unity";
        public const string Nav2ScenePath = "Assets/Scenes/RemoteNavigationNav2.unity";
        [MenuItem("DKSH/Spiderbot/Remote Navigation/Create or Open Sensor-Free Demo")]
        public static void CreateOrOpen()
        {
            if (EditorApplication.isPlayingOrWillChangePlaymode || EditorApplication.isCompiling)
                throw new InvalidOperationException("Exit Play Mode and wait for compilation.");
            if (AssetDatabase.LoadAssetAtPath<SceneAsset>(ScenePath) != null)
            { OpenSafely(ScenePath); return; }
            Scene scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Additive);
            var cameraObject = new GameObject("Main Camera"); SceneManager.MoveGameObjectToScene(cameraObject, scene);
            var camera = cameraObject.AddComponent<Camera>(); cameraObject.tag = "MainCamera";
            camera.clearFlags = CameraClearFlags.SolidColor; camera.backgroundColor = new Color(.04f, .07f, .1f);
            cameraObject.transform.position = new Vector3(0, 10, -10); cameraObject.transform.rotation = Quaternion.Euler(45, 0, 0);
            var lightObject = new GameObject("Directional Light"); SceneManager.MoveGameObjectToScene(lightObject, scene);
            lightObject.AddComponent<Light>().type = LightType.Directional;
            var panel = ScriptableObject.CreateInstance<PanelSettings>();
            panel.name = "Remote Navigation Panel"; panel.scaleMode = PanelScaleMode.ScaleWithScreenSize;
            panel.referenceResolution = new Vector2Int(1280, 860); panel.match = .5f;
            panel.themeStyleSheet = AssetDatabase.LoadAssetAtPath<ThemeStyleSheet>(
                "Assets/Resources/RemoteNavigationTheme.tss");
            string panelPath = "Assets/Resources/RemoteNavigationPanel.asset";
            AssetDatabase.CreateAsset(panel, panelPath);
            var app = new GameObject("Remote Navigation"); SceneManager.MoveGameObjectToScene(app, scene);
            app.AddComponent<UIDocument>().panelSettings = panel;
            app.AddComponent<RemoteNavigationApp>();
            EditorSceneManager.SaveScene(scene, ScenePath); AssetDatabase.SaveAssets();
            SceneManager.SetActiveScene(scene);
            OpenSafely(ScenePath);
            Debug.Log("DKSH_REMOTE_DEMO_READY: open RemoteNavigation alone and press Play.");
        }

        [MenuItem("DKSH/Spiderbot/Remote Navigation/Create or Open Nav2 Simulation")]
        public static void CreateNav2Simulation()
        {
            if (EditorApplication.isPlayingOrWillChangePlaymode || EditorApplication.isCompiling)
                throw new InvalidOperationException("Exit Play Mode and wait for compilation.");
            if (AssetDatabase.LoadAssetAtPath<SceneAsset>(Nav2ScenePath) != null)
            { OpenSafely(Nav2ScenePath); return; }
            if (!AssetDatabase.CopyAsset("Assets/Scenes/LidarScene.unity", Nav2ScenePath))
                throw new InvalidOperationException("The existing LidarScene could not be copied.");
            Scene scene = EditorSceneManager.OpenScene(Nav2ScenePath, OpenSceneMode.Additive);
            Ros2UnityBridge bridge = null;
            foreach (var root in scene.GetRootGameObjects())
            {
                foreach (var follower in root.GetComponentsInChildren<FrontierPathFollower>(true)) follower.enabled = false;
                foreach (var coordinator in root.GetComponentsInChildren<FrontierExplorerCoordinator>(true)) coordinator.enabled = false;
                foreach (var found in root.GetComponentsInChildren<Ros2UnityBridge>(true)) bridge = found;
            }
            if (bridge == null) throw new InvalidOperationException("LidarScene has no configured ROS2 bridge.");
            bridge.GetComponent<HexapodVelocityCommandAdapter>().enabled = true;
            bridge.enabled = true;
            var appObject = new GameObject("Remote Navigation"); SceneManager.MoveGameObjectToScene(appObject, scene);
            var panel = AssetDatabase.LoadAssetAtPath<PanelSettings>("Assets/Resources/RemoteNavigationPanel.asset");
            if (panel == null) throw new InvalidOperationException("Create the Sensor-Free Demo first.");
            appObject.AddComponent<UIDocument>().panelSettings = panel;
            var app = appObject.AddComponent<RemoteNavigationApp>();
            var serialized = new SerializedObject(app);
            serialized.FindProperty("startWithGateway").boolValue = true;
            serialized.ApplyModifiedPropertiesWithoutUndo();
            EditorSceneManager.SaveScene(scene, Nav2ScenePath);
            OpenSafely(Nav2ScenePath);
            Debug.Log("DKSH_REMOTE_NAV2_SCENE_READY: synthetic scan/odom/TF/clock, external cmd_vel, and gateway UI. Nav2 runtime validation is separate.");
        }

        private static void OpenSafely(string path)
        {
            for (int i = 0; i < SceneManager.sceneCount; i++)
                if (SceneManager.GetSceneAt(i).isDirty)
                { Debug.Log("Demo saved. Open it alone after saving your other scene changes."); return; }
            EditorSceneManager.OpenScene(path, OpenSceneMode.Single);
        }
    }
}
#endif
