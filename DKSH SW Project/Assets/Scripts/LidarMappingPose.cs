using DKSH.Spiderbot.Sensors;
using UnityEngine;

namespace DKSH.Spiderbot.Mapping
{
    public struct LidarMappingPose
    {
        public LidarMappingPose(Vector3 position, Quaternion rotation, double timestamp)
        {
            Position = position;
            Rotation = rotation;
            Timestamp = timestamp;
        }

        public Vector3 Position { get; private set; }
        public Quaternion Rotation { get; private set; }
        public double Timestamp { get; private set; }

        public static LidarMappingPose FromKnownPose(LidarScanFrame frame)
        {
            return new LidarMappingPose(
                frame.SensorPosition,
                frame.SensorRotation,
                frame.Timestamp);
        }
    }

    /// <summary>
    /// Supplies the LiDAR sensor pose in the occupancy grid's map frame.
    /// The mapper falls back to the scan frame's Unity Transform pose when no
    /// provider is assigned. A future localization system can implement this
    /// interface without changing the occupancy grid integration code.
    /// </summary>
    public interface ILidarMappingPoseProvider
    {
        bool TryGetSensorPose(LidarScanFrame frame, out LidarMappingPose pose);
    }
}
