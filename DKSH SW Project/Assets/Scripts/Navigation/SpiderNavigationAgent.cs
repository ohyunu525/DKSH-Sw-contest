using DKSH.Spiderbot.Sensors;
using DKSH.Spiderbot.Training;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;
using UnityEngine;

namespace DKSH.Spiderbot.Navigation
{
    [DisallowMultipleComponent]
    [RequireComponent(typeof(BehaviorParameters))]
    [RequireComponent(typeof(SimulatedLidarSensor))]
    [RequireComponent(typeof(SpiderProxyLocomotion))]
    public sealed class SpiderNavigationAgent : Agent
    {
        public const string NavigationBehaviorName = "SpiderNavigation";
        public const int ContinuousActionCount = 3;
        public const int LidarHorizontalBins = 36;
        public const int LidarVerticalBands = 3;
        public const int VectorObservationSize = 119;

        [Header("Episode")]
        [SerializeField] private GeneratedTrainingEnvironment environment;
        [SerializeField, Min(0.05f)] private float goalRadiusM = 0.45f;
        [SerializeField, Min(1f)] private float episodeBoundaryMarginM = 1.5f;

        [Header("Rewards")]
        [SerializeField, Min(0f)] private float goalReward = 2f;
        [SerializeField, Min(0f)] private float failurePenalty = 1f;
        [SerializeField, Min(0f)] private float progressRewardPerMeter = 0.1f;
        [SerializeField, Min(0f)] private float decisionPenalty = 0.0005f;
        [SerializeField, Min(0f)] private float collisionPenaltyScale = 0.002f;
        [SerializeField, Min(0f)] private float actionChangePenaltyScale = 0.0005f;

        private SpiderProxyLocomotion locomotion;
        private SimulatedLidarSensor lidar;
        private float lastDistanceToTarget;
        private Vector3 previousAction;
        private bool episodeIsEnding;

        public GeneratedTrainingEnvironment Environment { get { return environment; } }

        public void Configure(GeneratedTrainingEnvironment generatedEnvironment)
        {
            environment = generatedEnvironment;
        }

        public override void Initialize()
        {
            locomotion = GetComponent<SpiderProxyLocomotion>();
            lidar = GetComponent<SimulatedLidarSensor>();
            ConfigureMlAgentsComponents();
            locomotion.CollisionOccurred -= HandleCollision;
            locomotion.CollisionOccurred += HandleCollision;
        }

        public override void OnEpisodeBegin()
        {
            episodeIsEnding = false;
            previousAction = Vector3.zero;

            if (!TryResolveEnvironment())
            {
                Debug.LogError("SpiderNavigationAgent needs a GeneratedTrainingEnvironment.", this);
                enabled = false;
                return;
            }

            var start = environment.StartPoint;
            var target = environment.TargetPoint;
            if (start == null || target == null)
            {
                Debug.LogError("Training environment is missing StartPoint or TargetPoint.", environment);
                enabled = false;
                return;
            }

            var toTarget = target.position - start.position;
            toTarget.y = 0f;
            var rotation = toTarget.sqrMagnitude > 0.0001f
                ? Quaternion.LookRotation(toTarget.normalized, Vector3.up)
                : Quaternion.identity;
            locomotion.ResetMotion(start.position, rotation);
            lastDistanceToTarget = Vector3.Distance(transform.position, target.position);
            lidar.ScanNow();
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            if (!TryResolveEnvironment() || environment.TargetPoint == null)
            {
                AddZeroObservations(sensor, VectorObservationSize);
                return;
            }

            var targetOffset = environment.TargetPoint.position - transform.position;
            var distance = targetOffset.magnitude;
            var localTargetDirection = distance > 0.0001f
                ? transform.InverseTransformDirection(targetOffset / distance)
                : Vector3.zero;
            var mapDiagonal = Mathf.Max(
                1f,
                new Vector2(environment.MapSize.x, environment.MapSize.y).magnitude * environment.CellSize);

            sensor.AddObservation(localTargetDirection);
            sensor.AddObservation(Mathf.Clamp01(distance / mapDiagonal));

            var profile = locomotion.PhysicalProfile;
            var maxSpeed = profile != null ? profile.ProxyMaxSpeedMps : 1f;
            var maxYawRadians = profile != null
                ? profile.ProxyYawSpeedDegreesPerSecond * Mathf.Deg2Rad
                : Mathf.PI;
            var localVelocity = locomotion.LocalVelocity / Mathf.Max(0.01f, maxSpeed);
            sensor.AddObservation(Mathf.Clamp(localVelocity.x, -1f, 1f));
            sensor.AddObservation(Mathf.Clamp(localVelocity.z, -1f, 1f));
            sensor.AddObservation(Mathf.Clamp(locomotion.YawRateRadians / Mathf.Max(0.01f, maxYawRadians), -1f, 1f));
            sensor.AddObservation(Mathf.Clamp(Vector3.Dot(transform.up, Vector3.up), -1f, 1f));
            sensor.AddObservation(previousAction);

            AddLidarObservations(sensor);
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            if (episodeIsEnding || !TryResolveEnvironment())
            {
                return;
            }

            var continuous = actions.ContinuousActions;
            var action = new Vector3(
                Mathf.Clamp(continuous[0], -1f, 1f),
                Mathf.Clamp(continuous[2], -1f, 1f),
                Mathf.Clamp(continuous[1], -1f, 1f));
            locomotion.SetCommand(new Vector3(action.x, 0f, action.z), action.y);

            var actionDelta = action - previousAction;
            AddReward(-actionChangePenaltyScale * actionDelta.sqrMagnitude);
            previousAction = action;

            var distance = Vector3.Distance(transform.position, environment.TargetPoint.position);
            AddReward((lastDistanceToTarget - distance) * progressRewardPerMeter - decisionPenalty);
            lastDistanceToTarget = distance;

            if (distance <= goalRadiusM)
            {
                FinishEpisode(goalReward);
                return;
            }

            if (HasLeftEnvironment() || HasFallen())
            {
                FinishEpisode(-failurePenalty);
            }
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            var actions = actionsOut.ContinuousActions;
            actions[0] = 0f;
            actions[1] = 0f;
            actions[2] = 0f;
        }

        private void OnTriggerEnter(Collider other)
        {
            var hazard = other.GetComponentInParent<RLHazardZone>();
            if (hazard != null && hazard.IsActive)
            {
                FinishEpisode(-failurePenalty);
            }
        }

        private void OnDisable()
        {
            if (locomotion != null)
            {
                locomotion.CollisionOccurred -= HandleCollision;
            }
        }

        private void ConfigureMlAgentsComponents()
        {
            var behavior = GetComponent<BehaviorParameters>();
            behavior.BehaviorName = NavigationBehaviorName;
            behavior.BehaviorType = BehaviorType.Default;
            behavior.BrainParameters.VectorObservationSize = VectorObservationSize;
            behavior.BrainParameters.NumStackedVectorObservations = 1;
            behavior.BrainParameters.ActionSpec = ActionSpec.MakeContinuous(ContinuousActionCount);

            var requester = GetComponent<DecisionRequester>();
            var profile = locomotion != null ? locomotion.PhysicalProfile : null;
            var commandPeriod = profile != null ? profile.CommandPeriodSeconds : 0.02f;
            requester.DecisionPeriod = Mathf.Max(1, Mathf.RoundToInt(commandPeriod / Time.fixedDeltaTime));
            requester.TakeActionsBetweenDecisions = true;
            MaxStep = 5000;
        }

        private bool TryResolveEnvironment()
        {
            if (environment == null)
            {
                environment = GetComponentInParent<GeneratedTrainingEnvironment>();
            }

            return environment != null && environment.TargetPoint != null;
        }

        private void AddLidarObservations(VectorSensor sensor)
        {
            var bins = new float[LidarHorizontalBins * LidarVerticalBands];
            for (var i = 0; i < bins.Length; i++)
            {
                bins[i] = 1f;
            }

            LidarScanFrame frame;
            if (lidar != null && lidar.TryGetLatestFrame(out frame))
            {
                var maxDistance = Mathf.Max(0.01f, lidar.ActiveSettings.maxDistance);
                var horizontalResolution = Mathf.Max(1, frame.HorizontalResolution);
                var verticalResolution = Mathf.Max(1, frame.VerticalResolution);
                var samples = frame.Samples;
                for (var i = 0; i < samples.Count; i++)
                {
                    var sample = samples[i];
                    var horizontalBin = Mathf.Clamp(
                        sample.horizontalIndex * LidarHorizontalBins / horizontalResolution,
                        0,
                        LidarHorizontalBins - 1);
                    var verticalBand = Mathf.Clamp(
                        sample.verticalIndex * LidarVerticalBands / verticalResolution,
                        0,
                        LidarVerticalBands - 1);
                    var index = verticalBand * LidarHorizontalBins + horizontalBin;
                    bins[index] = Mathf.Min(bins[index], Mathf.Clamp01(sample.distance / maxDistance));
                }
            }

            for (var i = 0; i < bins.Length; i++)
            {
                sensor.AddObservation(bins[i]);
            }
        }

        private bool HasLeftEnvironment()
        {
            var root = environment.EnvironmentRoot != null ? environment.EnvironmentRoot : environment.transform;
            var local = root.InverseTransformPoint(transform.position);
            var halfWidth = environment.MapSize.x * environment.CellSize * 0.5f + episodeBoundaryMarginM;
            var halfDepth = environment.MapSize.y * environment.CellSize * 0.5f + episodeBoundaryMarginM;
            return Mathf.Abs(local.x) > halfWidth || Mathf.Abs(local.z) > halfDepth;
        }

        private bool HasFallen()
        {
            var root = environment.EnvironmentRoot != null ? environment.EnvironmentRoot : environment.transform;
            return root.InverseTransformPoint(transform.position).y < -2f
                || Vector3.Dot(transform.up, Vector3.up) < 0.3f;
        }

        private void HandleCollision(float speedMps)
        {
            if (!episodeIsEnding)
            {
                AddReward(-collisionPenaltyScale * Mathf.Clamp(speedMps, 0f, 2f));
            }
        }

        private void FinishEpisode(float reward)
        {
            if (episodeIsEnding)
            {
                return;
            }

            episodeIsEnding = true;
            SetReward(reward);
            EndEpisode();
        }

        private static void AddZeroObservations(VectorSensor sensor, int count)
        {
            for (var i = 0; i < count; i++)
            {
                sensor.AddObservation(0f);
            }
        }

        private void OnValidate()
        {
            goalRadiusM = Mathf.Max(0.05f, goalRadiusM);
            episodeBoundaryMarginM = Mathf.Max(1f, episodeBoundaryMarginM);
        }
    }
}
