#if UNITY_EDITOR
using System.Linq;
using DKSH.Spiderbot.Navigation;
using DKSH.Spiderbot.Training;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace DKSH.Spiderbot.Editor
{
    [InitializeOnLoad]
    public static class SpiderTrainingSceneBuilder
    {
        private const string SourceScenePath = "Assets/Scenes/MapForML.unity";
        private const string TrainingScenePath = "Assets/Scenes/SpiderNavigationTraining.unity";
        private const string ProfileFolderPath = "Assets/Training/Profiles";
        private const string ProfilePath = ProfileFolderPath + "/SpiderRobotPhysicalProfile.asset";

        static SpiderTrainingSceneBuilder()
        {
            EditorApplication.delayCall += EnsureTrainingAssets;
        }

        [MenuItem("DKSH/Spiderbot/Rebuild Navigation Training Scene")]
        public static void RebuildTrainingScene()
        {
            if (EditorApplication.isPlayingOrWillChangePlaymode || EditorApplication.isCompiling)
            {
                Debug.LogWarning("Wait for Edit Mode compilation to finish before rebuilding the training scene.");
                return;
            }

            var profile = GetOrCreateProfile();
            var scene = EditorSceneManager.OpenScene(SourceScenePath, OpenSceneMode.Single);
            if (!scene.IsValid())
            {
                Debug.LogError("Could not open source scene: " + SourceScenePath);
                return;
            }

            if (!EditorSceneManager.SaveScene(scene, TrainingScenePath, false))
            {
                Debug.LogError("Could not save training scene: " + TrainingScenePath);
                return;
            }

            DisableAssemblyOnlyModel(scene);
            ConfigureMapManager(scene, profile);
            EditorSceneManager.MarkSceneDirty(scene);
            EditorSceneManager.SaveScene(scene);

            EditorBuildSettings.scenes = new[]
            {
                new EditorBuildSettingsScene(TrainingScenePath, true)
            };
            AssetDatabase.SaveAssets();
            Debug.Log("Spider navigation training scene rebuilt: " + TrainingScenePath);
        }

        private static void EnsureTrainingAssets()
        {
            if (EditorApplication.isPlayingOrWillChangePlaymode || EditorApplication.isCompiling)
            {
                EditorApplication.delayCall += EnsureTrainingAssets;
                return;
            }

            var sceneExists = AssetDatabase.LoadAssetAtPath<SceneAsset>(TrainingScenePath) != null;
            var profileExists = AssetDatabase.LoadAssetAtPath<SpiderRobotPhysicalProfile>(ProfilePath) != null;
            if (!sceneExists || !profileExists)
            {
                RebuildTrainingScene();
                return;
            }

            EditorBuildSettings.scenes = new[]
            {
                new EditorBuildSettingsScene(TrainingScenePath, true)
            };
        }

        private static SpiderRobotPhysicalProfile GetOrCreateProfile()
        {
            EnsureFolder(ProfileFolderPath);
            var profile = AssetDatabase.LoadAssetAtPath<SpiderRobotPhysicalProfile>(ProfilePath);
            if (profile != null)
            {
                return profile;
            }

            profile = ScriptableObject.CreateInstance<SpiderRobotPhysicalProfile>();
            AssetDatabase.CreateAsset(profile, ProfilePath);
            return profile;
        }

        private static void DisableAssemblyOnlyModel(Scene scene)
        {
            var completeLeg = scene.GetRootGameObjects().FirstOrDefault(root => root.name == "completeLEG");
            if (completeLeg != null)
            {
                completeLeg.SetActive(false);
                EditorUtility.SetDirty(completeLeg);
            }
        }

        private static void ConfigureMapManager(Scene scene, SpiderRobotPhysicalProfile profile)
        {
            var mapManager = scene.GetRootGameObjects().FirstOrDefault(root => root.name == "MapManager");
            if (mapManager == null)
            {
                Debug.LogError("MapManager was not found in " + SourceScenePath);
                return;
            }

            var generator = mapManager.GetComponent<RLMapGenerator>();
            if (generator == null)
            {
                generator = mapManager.AddComponent<RLMapGenerator>();
            }

            generator.SelectedLevel = RLMapLevel.Flat;
            generator.MapCount = 8;
            generator.RegenerateOnPlay = true;
            EditorUtility.SetDirty(generator);

            var coordinator = mapManager.GetComponent<SpiderTrainingCoordinator>();
            if (coordinator == null)
            {
                coordinator = mapManager.AddComponent<SpiderTrainingCoordinator>();
            }

            var serializedCoordinator = new SerializedObject(coordinator);
            serializedCoordinator.FindProperty("physicalProfile").objectReferenceValue = profile;
            serializedCoordinator.FindProperty("editorDefaultLevel").enumValueIndex = 0;
            serializedCoordinator.FindProperty("parallelEnvironmentCount").intValue = 8;
            serializedCoordinator.FindProperty("physicsStepSeconds").floatValue = 0.005f;
            serializedCoordinator.ApplyModifiedPropertiesWithoutUndo();
            EditorUtility.SetDirty(coordinator);
        }

        private static void EnsureFolder(string folderPath)
        {
            if (AssetDatabase.IsValidFolder(folderPath))
            {
                return;
            }

            var parts = folderPath.Split('/');
            var current = parts[0];
            for (var i = 1; i < parts.Length; i++)
            {
                var next = current + "/" + parts[i];
                if (!AssetDatabase.IsValidFolder(next))
                {
                    AssetDatabase.CreateFolder(current, parts[i]);
                }

                current = next;
            }
        }
    }
}
#endif
