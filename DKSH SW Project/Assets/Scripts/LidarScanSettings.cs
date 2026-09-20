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

        [Min(0f)]
        public float minDistance;

        [Min(0f)]
        public float measurementAccuracy;

        [Min(0.0001f)]
        public float measurementResolution;

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
                    // 180 * 11 * 11 Hz = 21,780 rays/s, close to the L1 RM's
                    // documented 21,600 effective points/s without allocating
                    // its complete proprietary scan pattern.
                    horizontalResolution = 180,
                    verticalResolution = 11,
                    horizontalFovDegrees = 360f,
                    verticalFovDegrees = 90f,
                    maxDistance = 30f,
                    minDistance = 0.05f,
                    measurementAccuracy = 0.02f,
                    measurementResolution = 0.008f,
                    scanFrequencyHz = 11f,
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
            minDistance = Mathf.Clamp(minDistance, 0f, maxDistance);
            measurementAccuracy = Mathf.Max(0f, measurementAccuracy);
            measurementResolution = Mathf.Max(0.0001f, measurementResolution);
            scanFrequencyHz = Mathf.Max(0.01f, scanFrequencyHz);
        }
    }
}
