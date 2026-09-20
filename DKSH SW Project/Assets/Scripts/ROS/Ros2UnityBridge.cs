#if !ROS2
#error The DKSH ROS bridge requires the ROS2 scripting define for Standalone builds.
#endif

using System;
using DKSH.Spiderbot.Sensors;
using RosMessageTypes.BuiltinInterfaces;
using RosMessageTypes.Geometry;
using RosMessageTypes.Nav;
using RosMessageTypes.Rosgraph;
using RosMessageTypes.Sensor;
using RosMessageTypes.Std;
using RosMessageTypes.Tf2;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.ROSGeometry;
using UnityEngine;

namespace DKSH.Spiderbot.ROS
{
    /// <summary>
    /// Thin transport adapter between the existing Unity simulation and ROS2.
    /// Navigation remains in ROS2; this component publishes sensors/pose and consumes cmd_vel.
    /// </summary>
    [DisallowMultipleComponent]
    [RequireComponent(typeof(SimulatedLidarSensor))]
    [RequireComponent(typeof(HexapodVelocityCommandAdapter))]
    public sealed class Ros2UnityBridge : MonoBehaviour
    {
        [Header("Scene references")]
        [SerializeField] private SimulatedLidarSensor lidarSensor;
        [SerializeField] private Transform robotRoot;
        [SerializeField] private HexapodVelocityCommandAdapter locomotionAdapter;

        [Header("ROS TCP")]
        [SerializeField] private string rosIpAddress = "127.0.0.1";
        [SerializeField] private int rosTcpPort = 10000;

        [Header("Topics")]
        [SerializeField] private string scanTopic = "/scan";
        [SerializeField] private string odometryTopic = "/odom";
        [SerializeField] private string transformTopic = "/tf";
        [SerializeField] private string clockTopic = "/clock";
        [SerializeField] private string velocityCommandTopic = "/cmd_vel";

        [Header("Frames")]
        [SerializeField] private string odometryFrame = "odom";
        [SerializeField] private string baseFrame = "base_footprint";
        [SerializeField] private string laserFrame = "laser";

        [Header("Publication budget")]
        [SerializeField, Range(1f, 20f)] private float posePublishFrequencyHz = 10f;
        [SerializeField] private bool publishSimulationClock = true;
        [SerializeField] private bool disconnectOnDisable = true;

        private static readonly double[] PoseCovariance = CreatePoseCovariance();
        private static readonly double[] TwistCovariance = CreateTwistCovariance();

        private ROSConnection rosConnection;
        private Vector3 initialPosition;
        private Quaternion initialRotation;
        private Vector3 previousPosition;
        private float previousYawDegrees;
        private double previousPoseTime;
        private double nextPosePublishTime;
        private bool initialized;

        public void Configure(
            SimulatedLidarSensor sensor,
            Transform configuredRobotRoot,
            HexapodVelocityCommandAdapter adapter,
            string ipAddress,
            int tcpPort,
            float publishFrequencyHz)
        {
            lidarSensor = sensor;
            robotRoot = configuredRobotRoot;
            locomotionAdapter = adapter;
            rosIpAddress = string.IsNullOrWhiteSpace(ipAddress) ? "127.0.0.1" : ipAddress;
            rosTcpPort = Mathf.Clamp(tcpPort, 1, 65535);
            posePublishFrequencyHz = Mathf.Clamp(publishFrequencyHz, 1f, 20f);
        }

        private void OnEnable()
        {
            EnsureReferences();
            ResetOdometryOrigin();
            locomotionAdapter.SetExternalControlEnabled(true);
            lidarSensor.ScanCompleted += PublishScan;
            InitializeRosConnection();
        }

        private void OnDisable()
        {
            if (lidarSensor != null)
            {
                lidarSensor.ScanCompleted -= PublishScan;
            }

            if (locomotionAdapter != null)
            {
                locomotionAdapter.SetExternalControlEnabled(false);
            }

            if (rosConnection != null && initialized)
            {
                rosConnection.Unsubscribe(velocityCommandTopic);
                if (disconnectOnDisable)
                {
                    rosConnection.Disconnect();
                }
            }

            initialized = false;
        }

        private void Update()
        {
            if (!initialized || Time.timeAsDouble < nextPosePublishTime)
            {
                return;
            }

            var period = 1.0 / posePublishFrequencyHz;
            nextPosePublishTime = Time.timeAsDouble + period;
            PublishPoseAndClock(Time.timeAsDouble);
        }

        private void InitializeRosConnection()
        {
            rosConnection = ROSConnection.GetOrCreateInstance();
            rosConnection.RosIPAddress = rosIpAddress;
            rosConnection.RosPort = rosTcpPort;
            rosConnection.ConnectOnStart = false;
            rosConnection.ShowHud = false;
            rosConnection.KeepaliveTime = 2f;
            rosConnection.NetworkTimeoutSeconds = 2f;
            rosConnection.SleepTimeSeconds = 0.01f;
            rosConnection.listenForTFMessages = false;

            rosConnection.RegisterPublisher<LaserScanMsg>(scanTopic, 2);
            rosConnection.RegisterPublisher<OdometryMsg>(odometryTopic, 2);
            rosConnection.RegisterPublisher<TFMessageMsg>(transformTopic, 2);
            if (publishSimulationClock)
            {
                rosConnection.RegisterPublisher<ClockMsg>(clockTopic, 2);
            }

            rosConnection.Subscribe<TwistMsg>(velocityCommandTopic, ReceiveVelocityCommand);
            if (!rosConnection.HasConnectionThread)
            {
                rosConnection.Connect();
            }

            initialized = true;
        }

        private void ResetOdometryOrigin()
        {
            initialPosition = robotRoot.position;
            initialRotation = robotRoot.rotation;
            previousPosition = robotRoot.position;
            previousYawDegrees = robotRoot.eulerAngles.y;
            previousPoseTime = Time.timeAsDouble;
            nextPosePublishTime = Time.timeAsDouble;
        }

        private void PublishScan(LidarScanFrame frame)
        {
            if (!initialized || frame == null)
            {
                return;
            }

            var settings = lidarSensor.ActiveSettings;
            var horizontalCount = Mathf.Max(1, frame.HorizontalResolution);
            var selectedVerticalIndex = Mathf.Clamp(
                (frame.VerticalResolution - 1) / 2,
                0,
                Mathf.Max(0, frame.VerticalResolution - 1));
            var ranges = new float[horizontalCount];
            for (var i = 0; i < ranges.Length; i++)
            {
                ranges[i] = settings.maxDistance;
            }

            var fullCircle = settings.horizontalFovDegrees >= 359.999f;
            for (var i = 0; i < frame.Samples.Count; i++)
            {
                var sample = frame.Samples[i];
                if (sample.verticalIndex != selectedVerticalIndex ||
                    sample.horizontalIndex < 0 ||
                    sample.horizontalIndex >= horizontalCount)
                {
                    continue;
                }

                var rosIndex = ToRosScanIndex(
                    sample.horizontalIndex,
                    horizontalCount,
                    fullCircle);
                ranges[rosIndex] = Mathf.Clamp(
                    sample.distance,
                    settings.minDistance,
                    settings.maxDistance);
            }

            var fieldOfViewRadians = settings.horizontalFovDegrees * Mathf.Deg2Rad;
            var angleIncrement = horizontalCount <= 1
                ? 0f
                : fieldOfViewRadians / (fullCircle ? horizontalCount : horizontalCount - 1f);
            var angleMin = -fieldOfViewRadians * 0.5f;
            var angleMax = angleMin + angleIncrement * (horizontalCount - 1);
            var scanTime = 1f / settings.scanFrequencyHz;
            var message = new LaserScanMsg(
                new HeaderMsg(CreateStamp(frame.Timestamp), laserFrame),
                angleMin,
                angleMax,
                angleIncrement,
                horizontalCount > 0 ? scanTime / horizontalCount : 0f,
                scanTime,
                settings.minDistance,
                settings.maxDistance,
                ranges,
                Array.Empty<float>());
            rosConnection.Publish(scanTopic, message);
        }

        private void PublishPoseAndClock(double timestamp)
        {
            var stamp = CreateStamp(timestamp);
            if (publishSimulationClock)
            {
                rosConnection.Publish(clockTopic, new ClockMsg(stamp));
            }

            var relativePosition = Quaternion.Inverse(initialRotation) *
                                   (robotRoot.position - initialPosition);
            var relativeRotation = Quaternion.Inverse(initialRotation) * robotRoot.rotation;
            var deltaTime = Math.Max(0.0001, timestamp - previousPoseTime);
            var worldVelocity = (robotRoot.position - previousPosition) / (float)deltaTime;
            var bodyVelocity = Quaternion.Inverse(robotRoot.rotation) * worldVelocity;
            var yawRateDegrees = Mathf.DeltaAngle(
                                     previousYawDegrees,
                                     robotRoot.eulerAngles.y) /
                                 (float)deltaTime;
            var unityAngularVelocity = new Vector3(
                0f,
                yawRateDegrees * Mathf.Deg2Rad,
                0f);

            PointMsg rosPosition = relativePosition.To<FLU>();
            QuaternionMsg rosRotation = relativeRotation.To<FLU>();
            Vector3Msg rosLinearVelocity = bodyVelocity.To<FLU>();
            Vector3Msg rosAngularVelocity =
                Unity.Robotics.ROSTCPConnector.ROSGeometry.Vector3<FLU>
                    .FromUnityAngularVelocity(unityAngularVelocity);

            var pose = new PoseMsg(rosPosition, rosRotation);
            var twist = new TwistMsg(rosLinearVelocity, rosAngularVelocity);
            var odometry = new OdometryMsg(
                new HeaderMsg(stamp, odometryFrame),
                baseFrame,
                new PoseWithCovarianceMsg(pose, PoseCovariance),
                new TwistWithCovarianceMsg(twist, TwistCovariance));
            rosConnection.Publish(odometryTopic, odometry);

            Vector3Msg rosBaseTranslation = relativePosition.To<FLU>();
            QuaternionMsg rosBaseRotation = relativeRotation.To<FLU>();
            var baseTransform = new TransformStampedMsg(
                new HeaderMsg(stamp, odometryFrame),
                baseFrame,
                new TransformMsg(rosBaseTranslation, rosBaseRotation));

            var sensorPositionInBase = robotRoot.InverseTransformPoint(lidarSensor.transform.position);
            var sensorRotationInBase = Quaternion.Inverse(robotRoot.rotation) *
                                       lidarSensor.transform.rotation;
            Vector3Msg rosSensorTranslation = sensorPositionInBase.To<FLU>();
            QuaternionMsg rosSensorRotation = sensorRotationInBase.To<FLU>();
            var sensorTransform = new TransformStampedMsg(
                new HeaderMsg(stamp, baseFrame),
                laserFrame,
                new TransformMsg(rosSensorTranslation, rosSensorRotation));
            rosConnection.Publish(
                transformTopic,
                new TFMessageMsg(new[] { baseTransform, sensorTransform }));

            previousPosition = robotRoot.position;
            previousYawDegrees = robotRoot.eulerAngles.y;
            previousPoseTime = timestamp;
        }

        private void ReceiveVelocityCommand(TwistMsg message)
        {
            if (message == null || locomotionAdapter == null)
            {
                return;
            }

            locomotionAdapter.SetVelocityCommand(
                (float)message.linear.x,
                (float)message.linear.y,
                (float)message.angular.z);
        }

        private void EnsureReferences()
        {
            if (lidarSensor == null)
            {
                lidarSensor = GetComponent<SimulatedLidarSensor>();
            }

            if (robotRoot == null)
            {
                robotRoot = transform;
            }

            if (locomotionAdapter == null)
            {
                locomotionAdapter = GetComponent<HexapodVelocityCommandAdapter>();
            }
        }

        private static int ToRosScanIndex(int unityIndex, int count, bool fullCircle)
        {
            if (count <= 1)
            {
                return 0;
            }

            if (fullCircle && unityIndex == 0)
            {
                return 0;
            }

            return fullCircle
                ? count - unityIndex
                : count - 1 - unityIndex;
        }

        private static TimeMsg CreateStamp(double timestamp)
        {
            var seconds = Math.Floor(Math.Max(0.0, timestamp));
            var nanoseconds = (uint)Math.Round((timestamp - seconds) * 1_000_000_000.0);
            if (nanoseconds >= 1_000_000_000)
            {
                seconds += 1.0;
                nanoseconds = 0;
            }

            return new TimeMsg(checked((int)seconds), nanoseconds);
        }

        private static double[] CreatePoseCovariance()
        {
            var covariance = new double[36];
            covariance[0] = 0.0025;
            covariance[7] = 0.0025;
            covariance[14] = 999.0;
            covariance[21] = 999.0;
            covariance[28] = 999.0;
            covariance[35] = 0.01;
            return covariance;
        }

        private static double[] CreateTwistCovariance()
        {
            var covariance = new double[36];
            covariance[0] = 0.01;
            covariance[7] = 0.01;
            covariance[14] = 999.0;
            covariance[21] = 999.0;
            covariance[28] = 999.0;
            covariance[35] = 0.02;
            return covariance;
        }

        private void OnValidate()
        {
            rosTcpPort = Mathf.Clamp(rosTcpPort, 1, 65535);
            posePublishFrequencyHz = Mathf.Clamp(posePublishFrequencyHz, 1f, 20f);
        }
    }
}
