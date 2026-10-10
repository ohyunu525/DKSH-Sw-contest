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

        [RangeAttribute(-90f, 90f)]
        public float verticalMinAngleDegrees;

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
                    // 360 * 32 * 5.55 Hz = 63,936 rays/s, close to the L2's
                    // documented 64,000 effective points/s without allocating
                    // its complete proprietary scan pattern.
                    horizontalResolution = 360,
                    verticalResolution = 32,
                    horizontalFovDegrees = 360f,
                    verticalFovDegrees = 96f,
                    verticalMinAngleDegrees = -6f,
                    maxDistance = 30f,
                    minDistance = 0.05f,
                    measurementAccuracy = 0.02f,
                    measurementResolution = 0.0045f,
                    scanFrequencyHz = 5.55f,
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
            verticalMinAngleDegrees = Mathf.Clamp(verticalMinAngleDegrees, -90f, 90f);
            verticalFovDegrees = Mathf.Clamp(verticalFovDegrees, 0f, 90f - verticalMinAngleDegrees);
            maxDistance = Mathf.Max(0.01f, maxDistance);
            minDistance = Mathf.Clamp(minDistance, 0f, maxDistance);
            measurementAccuracy = Mathf.Max(0f, measurementAccuracy);
            measurementResolution = Mathf.Max(0.0001f, measurementResolution);
            scanFrequencyHz = Mathf.Max(0.01f, scanFrequencyHz);
        }

        public float GetVerticalAngle(int verticalIndex)
        {
            if (verticalResolution <= 1)
            {
                return Mathf.Clamp(0f, verticalMinAngleDegrees,
                    verticalMinAngleDegrees + verticalFovDegrees);
            }

            // Include the horizon explicitly so the planar navigation slice
            // remains useful even when the field of view is asymmetric.
            if (verticalMinAngleDegrees < 0f &&
                verticalMinAngleDegrees + verticalFovDegrees > 0f &&
                verticalResolution >= 3)
            {
                if (verticalIndex == 0)
                {
                    return verticalMinAngleDegrees;
                }

                return Mathf.Lerp(0f, verticalMinAngleDegrees + verticalFovDegrees,
                    (verticalIndex - 1f) / (verticalResolution - 2f));
            }

            return Mathf.Lerp(verticalMinAngleDegrees,
                verticalMinAngleDegrees + verticalFovDegrees,
                verticalIndex / (verticalResolution - 1f));
        }

        public int HorizontalSliceIndex
        {
            get
            {
                var closest = 0;
                for (var i = 1; i < verticalResolution; i++)
                {
                    if (Mathf.Abs(GetVerticalAngle(i)) < Mathf.Abs(GetVerticalAngle(closest)))
                    {
                        closest = i;
                    }
                }
                return closest;
            }
        }
    }
}
