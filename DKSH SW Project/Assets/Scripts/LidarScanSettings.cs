using System;
using UnityEngine;

namespace DKSH.Spiderbot.Sensors
{
    [Serializable]
    public struct LidarScanSettings
    {
        [Min(1)]
        public int horizontalResolution;

        [Min(1)]
        public int verticalResolution;

        [RangeAttribute(0.1f, 360f)]
        public float horizontalFovDegrees;

        [RangeAttribute(0f, 180f)]
        public float verticalFovDegrees;

        [Min(0.01f)]
        public float maxDistance;

        [Min(0.01f)]
        public float scanFrequencyHz;

        public LayerMask detectionMask;
        public QueryTriggerInteraction triggerInteraction;
        public bool rotateWithSensor;
        public bool includeMisses;

        public static LidarScanSettings Default
        {
            get
            {
                return new LidarScanSettings
                {
                    horizontalResolution = 180,
                    verticalResolution = 8,
                    horizontalFovDegrees = 360f,
                    verticalFovDegrees = 30f,
                    maxDistance = 12f,
                    scanFrequencyHz = 10f,
                    detectionMask = new LayerMask { value = Physics.DefaultRaycastLayers },
                    triggerInteraction = QueryTriggerInteraction.Ignore,
                    rotateWithSensor = true,
                    includeMisses = true
                };
            }
        }

        public int SampleCount
        {
            get { return Mathf.Max(1, horizontalResolution) * Mathf.Max(1, verticalResolution); }
        }

        public void Clamp()
        {
            horizontalResolution = Mathf.Max(1, horizontalResolution);
            verticalResolution = Mathf.Max(1, verticalResolution);
            horizontalFovDegrees = Mathf.Clamp(horizontalFovDegrees, 0.1f, 360f);
            verticalFovDegrees = Mathf.Clamp(verticalFovDegrees, 0f, 180f);
            maxDistance = Mathf.Max(0.01f, maxDistance);
            scanFrequencyHz = Mathf.Max(0.01f, scanFrequencyHz);
        }
    }
}
