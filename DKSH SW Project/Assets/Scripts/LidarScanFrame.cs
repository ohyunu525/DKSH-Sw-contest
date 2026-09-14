using System.Collections.Generic;
using UnityEngine;

namespace DKSH.Spiderbot.Sensors
{
    public sealed class LidarScanFrame
    {
        private readonly LidarSample[] samples;

        public LidarScanFrame(
            double timestamp,
            Vector3 sensorPosition,
            Quaternion sensorRotation,
            int horizontalResolution,
            int verticalResolution,
            int hitCount,
            LidarSample[] samples)
            : this(
                timestamp,
                sensorPosition,
                sensorRotation,
                sensorRotation,
                horizontalResolution,
                verticalResolution,
                hitCount,
                samples)
        {
        }

        public LidarScanFrame(
            double timestamp,
            Vector3 sensorPosition,
            Quaternion sensorRotation,
            Quaternion rayRotation,
            int horizontalResolution,
            int verticalResolution,
            int hitCount,
            LidarSample[] samples)
        {
            Timestamp = timestamp;
            SensorPosition = sensorPosition;
            SensorRotation = sensorRotation;
            RayRotation = rayRotation;
            HorizontalResolution = horizontalResolution;
            VerticalResolution = verticalResolution;
            HitCount = hitCount;
            this.samples = samples ?? new LidarSample[0];
        }

        public double Timestamp { get; private set; }
        public Vector3 SensorPosition { get; private set; }
        public Quaternion SensorRotation { get; private set; }
        public Quaternion RayRotation { get; private set; }
        public int HorizontalResolution { get; private set; }
        public int VerticalResolution { get; private set; }
        public int HitCount { get; private set; }
        public int Count { get { return samples.Length; } }
        public IReadOnlyList<LidarSample> Samples { get { return samples; } }

        public LidarSample[] CopySamples()
        {
            var copy = new LidarSample[samples.Length];
            samples.CopyTo(copy, 0);
            return copy;
        }
    }
}
