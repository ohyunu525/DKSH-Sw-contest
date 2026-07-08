using System.Collections.Generic;
using DKSH.Spiderbot.Sensors;
using UnityEngine;

namespace DKSH.Spiderbot.Mapping
{
    [DisallowMultipleComponent]
    public sealed class LidarPointCloudMap : MonoBehaviour
    {
        [SerializeField]
        private SimulatedLidarSensor lidarSensor;

        [SerializeField, Min(1)]
        private int maxPoints = 12000;

        [SerializeField]
        private bool clearOnEachFrame = false;

        [SerializeField]
        private bool accumulateMisses = false;

        [Header("Debug")]
        [SerializeField]
        private bool drawGizmos = true;

        [SerializeField]
        private Color pointColor = new Color(0.1f, 1f, 0.55f, 0.9f);

        [SerializeField, Min(0.001f)]
        private float pointRadius = 0.035f;

        [SerializeField, Min(1)]
        private int maxGizmoPoints = 2000;

        private readonly List<Vector3> points = new List<Vector3>();
        private int nextWriteIndex;

        public IReadOnlyList<Vector3> Points { get { return points; } }
        public int Count { get { return points.Count; } }

        public void Clear()
        {
            points.Clear();
            nextWriteIndex = 0;
        }

        private void Reset()
        {
            ResolveSensorReference();
        }

        private void Awake()
        {
            ResolveSensorReference();
        }

        private void OnEnable()
        {
            ResolveSensorReference();

            if (lidarSensor != null)
            {
                lidarSensor.ScanCompleted += HandleScanCompleted;
            }
        }

        private void OnDisable()
        {
            if (lidarSensor != null)
            {
                lidarSensor.ScanCompleted -= HandleScanCompleted;
            }
        }

        private void OnValidate()
        {
            maxPoints = Mathf.Max(1, maxPoints);
            pointRadius = Mathf.Max(0.001f, pointRadius);
            maxGizmoPoints = Mathf.Max(1, maxGizmoPoints);
        }

        private void ResolveSensorReference()
        {
            if (lidarSensor == null)
            {
                lidarSensor = GetComponent<SimulatedLidarSensor>();
            }

            if (lidarSensor == null)
            {
                lidarSensor = GetComponentInParent<SimulatedLidarSensor>();
            }
        }

        private void HandleScanCompleted(LidarScanFrame frame)
        {
            if (frame == null)
            {
                return;
            }

            if (clearOnEachFrame)
            {
                Clear();
            }

            var samples = frame.Samples;
            for (var i = 0; i < samples.Count; i++)
            {
                var sample = samples[i];
                if (sample.hit || accumulateMisses)
                {
                    AddPoint(sample.point);
                }
            }
        }

        private void AddPoint(Vector3 point)
        {
            if (points.Count < maxPoints)
            {
                points.Add(point);
                return;
            }

            points[nextWriteIndex] = point;
            nextWriteIndex = (nextWriteIndex + 1) % maxPoints;
        }

        private void OnDrawGizmosSelected()
        {
            if (!drawGizmos || points.Count == 0)
            {
                return;
            }

            Gizmos.color = pointColor;
            var stride = Mathf.Max(1, points.Count / maxGizmoPoints);
            for (var i = 0; i < points.Count; i += stride)
            {
                Gizmos.DrawSphere(points[i], pointRadius);
            }
        }
    }
}
