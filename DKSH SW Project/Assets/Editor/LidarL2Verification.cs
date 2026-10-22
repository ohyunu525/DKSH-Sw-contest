#if UNITY_EDITOR
using System;
using DKSH.Spiderbot.Sensors;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace DKSH.Spiderbot.Editor
{
    public static class LidarL2Verification
    {
        [MenuItem("DKSH/Spiderbot/Sensors/Run L2 Verification")]
        public static void RunVerification()
        {
            var settings = LidarScanSettings.Default;
            Require(Mathf.Approximately(settings.GetVerticalAngle(0), -6f), "NEGA lower edge must be -6 degrees.");
            Require(Mathf.Approximately(settings.GetVerticalAngle(settings.HorizontalSliceIndex), 0f),
                "The navigation slice must lie on the horizon.");
            Require(Mathf.Approximately(settings.GetVerticalAngle(settings.verticalResolution - 1), 90f),
                "The upper edge must end at zenith.");
            Require(Mathf.Abs(settings.SampleCount * settings.scanFrequencyHz - 64000f) < 100f,
                "Simulation ray budget must approximate the L2 effective point rate.");

            var sensorObject = new GameObject("L2VerificationSensor");
            var obstacle = GameObject.CreatePrimitive(PrimitiveType.Cube);
            try
            {
                sensorObject.transform.position = new Vector3(1000f, 0f, 1000f);
                var sensor = sensorObject.AddComponent<SimulatedLidarSensor>();
                settings.measurementAccuracy = 0f;
                sensor.Configure(settings, false);
                obstacle.transform.position = sensorObject.transform.position + Vector3.forward;
                obstacle.transform.localScale = new Vector3(0.1f, 0.05f, 0.1f);
                Physics.SyncTransforms();
                var frame = sensor.ScanNow();
                var horizontalHit = false;
                for (var i = 0; i < frame.Samples.Count; i++)
                {
                    var sample = frame.Samples[i];
                    if (sample.verticalIndex == settings.HorizontalSliceIndex && sample.hit &&
                        Vector3.Dot(sample.direction, Vector3.forward) > 0.999f)
                    {
                        horizontalHit = true;
                        Require(Mathf.Abs(sample.distance - 0.95f) <= settings.measurementResolution,
                            "The horizontal wall distance must use L2 quantization.");
                    }
                    if (sample.verticalIndex == 0)
                    {
                        Require(Mathf.Abs(sample.direction.y - Mathf.Sin(-6f * Mathf.Deg2Rad)) < 0.0001f,
                            "NEGA rays must point 6 degrees below the mounting plane.");
                    }
                }
                Require(horizontalHit, "The horizontal scan must detect a wall at sensor height.");

                obstacle.transform.position = sensorObject.transform.position +
                    new Vector3(0f, -Mathf.Tan(6f * Mathf.Deg2Rad), 1f);
                Physics.SyncTransforms();
                frame = sensor.ScanNow();
                var negativeHit = false;
                for (var i = 0; i < frame.Samples.Count; i++)
                {
                    negativeHit |= frame.Samples[i].verticalIndex == 0 && frame.Samples[i].hit;
                }
                Require(negativeHit, "The expanded negative slice must detect a low obstacle.");
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(obstacle);
                UnityEngine.Object.DestroyImmediate(sensorObject);
            }

            const string scenePath = "Assets/Scenes/LidarScene.unity";
            var loaded = SceneManager.GetSceneByPath(scenePath).isLoaded;
            var scene = loaded ? SceneManager.GetSceneByPath(scenePath) :
                EditorSceneManager.OpenScene(scenePath, OpenSceneMode.Additive);
            try
            {
                SimulatedLidarSensor sensor = null;
                foreach (var root in scene.GetRootGameObjects())
                {
                    sensor = root.GetComponentInChildren<SimulatedLidarSensor>(true);
                    if (sensor != null) break;
                }
                Require(sensor != null, "LidarScene must contain a simulated sensor.");
                var saved = sensor.ActiveSettings;
                Require(Mathf.Approximately(saved.verticalMinAngleDegrees, -6f) &&
                    Mathf.Approximately(saved.verticalFovDegrees, 96f) &&
                    Mathf.Approximately(saved.scanFrequencyHz, 5.55f) &&
                    Mathf.Approximately(saved.measurementResolution, 0.0045f),
                    "The saved scene must use the active L2 measurement contract.");
            }
            finally
            {
                if (!loaded) EditorSceneManager.CloseScene(scene, true);
            }
            Debug.Log("DKSH_L2_TESTS_PASS: NEGA rays, horizontal projection, quantization, and saved scene passed.");
        }

        public static void RunVerificationFromCommandLine()
        {
            RunVerification();
        }

        private static void Require(bool condition, string message)
        {
            if (!condition) throw new InvalidOperationException("L2 verification failed: " + message);
        }
    }
}
#endif
