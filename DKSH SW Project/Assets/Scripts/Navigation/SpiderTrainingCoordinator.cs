using DKSH.Spiderbot.Sensors;
using DKSH.Spiderbot.Training;
using Unity.MLAgents;
using Unity.MLAgents.Policies;
using UnityEngine;

namespace DKSH.Spiderbot.Navigation
{
    [DisallowMultipleComponent]
    [RequireComponent(typeof(RLMapGenerator))]
    public sealed class SpiderTrainingCoordinator : MonoBehaviour
    {
        [SerializeField] private SpiderRobotPhysicalProfile physicalProfile;
        [SerializeField] private RLMapLevel editorDefaultLevel = RLMapLevel.Flat;
        [SerializeField, Min(1)] private int parallelEnvironmentCount = 8;
        [SerializeField, Min(0.001f)] private float physicsStepSeconds = 0.005f;

        private RLMapGenerator mapGenerator;

        private void Awake()
        {
            mapGenerator = GetComponent<RLMapGenerator>();
            if (physicalProfile == null)
            {
                physicalProfile = SpiderRobotPhysicalProfile.CreateRuntimeDefault();
            }

            Time.fixedDeltaTime = physicsStepSeconds;
            mapGenerator.MapCount = parallelEnvironmentCount;
            mapGenerator.RegenerateOnPlay = false;

            var configuredLevel = Academy.Instance.EnvironmentParameters.GetWithDefault(
                "map_level",
                (float)editorDefaultLevel);
            mapGenerator.SelectedLevel = (RLMapLevel)Mathf.Clamp(
                Mathf.RoundToInt(configuredLevel),
                (int)RLMapLevel.Flat,
                (int)RLMapLevel.BuildingSpreadingFireRandomCollapse);
        }

        private void Start()
        {
            mapGenerator.Generate();

            for (var i = 0; i < mapGenerator.Environments.Count; i++)
            {
                var environment = mapGenerator.Environments[i];
                if (environment != null)
                {
                    SpawnAgent(environment, i);
                }
            }
        }

private void SpawnAgent(GeneratedTrainingEnvironment environment, int index)
        {
            var agentObject = new GameObject("SpiderNavigationAgent_" + index);
            agentObject.SetActive(false);
            agentObject.transform.SetParent(environment.EnvironmentRoot, true);

            var controller = agentObject.AddComponent<CharacterController>();
            controller.enabled = true;

            var behavior = agentObject.AddComponent<BehaviorParameters>();
            behavior.BehaviorName = SpiderNavigationAgent.NavigationBehaviorName;
            behavior.BehaviorType = BehaviorType.Default;
            behavior.BrainParameters.VectorObservationSize = SpiderNavigationAgent.VectorObservationSize;
            behavior.BrainParameters.NumStackedVectorObservations = 1;
            behavior.BrainParameters.ActionSpec =
                Unity.MLAgents.Actuators.ActionSpec.MakeContinuous(
                    SpiderNavigationAgent.ContinuousActionCount);

            var lidar = agentObject.AddComponent<SimulatedLidarSensor>();
            var lidarSettings = LidarScanSettings.Default;
            lidarSettings.horizontalResolution = 180;
            lidarSettings.verticalResolution = 11;
            lidarSettings.verticalFovDegrees = 90f;
            lidarSettings.maxDistance = 30f;
            lidarSettings.scanFrequencyHz = 11f;
            lidar.Configure(lidarSettings, true);

            var locomotion = agentObject.AddComponent<SpiderProxyLocomotion>();
            locomotion.Configure(physicalProfile);

            var agent = agentObject.AddComponent<SpiderNavigationAgent>();
            agent.Configure(environment);
            agentObject.AddComponent<DecisionRequester>();
            CreateProxyVisual(agentObject.transform);
            agentObject.SetActive(true);
        }

        private static void CreateProxyVisual(Transform parent)
        {
            var visual = GameObject.CreatePrimitive(PrimitiveType.Capsule);
            visual.name = "ProxyBodyVisual";
            visual.transform.SetParent(parent, false);
            visual.transform.localScale = new Vector3(0.32f, 0.12f, 0.42f);
            var collider = visual.GetComponent<Collider>();
            if (collider != null)
            {
                Destroy(collider);
            }
        }

        private void OnValidate()
        {
            parallelEnvironmentCount = Mathf.Max(1, parallelEnvironmentCount);
            physicsStepSeconds = Mathf.Clamp(physicsStepSeconds, 0.001f, 0.02f);
        }
    }
}
