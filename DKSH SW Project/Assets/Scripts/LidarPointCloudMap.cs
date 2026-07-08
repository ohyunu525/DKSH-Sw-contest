using System;
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

        [SerializeField]
        private Transform realSpaceFloor = null;

        [SerializeField]
        private string realSpaceFloorName = "RealSpaceFloor";

        [SerializeField]
        private bool storePointsInRealSpaceFloor = true;

        [SerializeField, Min(1)]
        private int maxPoints = 250000;

        [SerializeField]
        private bool rejectDuplicateCells = true;

        [SerializeField, Min(0.001f)]
        private float cellSize = 0.05f;

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
        private readonly HashSet<Vector3Int> occupiedCells = new HashSet<Vector3Int>();

        public event Action<LidarPointCloudMap> PointsChanged;

        public IReadOnlyList<Vector3> Points { get { return points; } }
        public int Count { get { return points.Count; } }
        public int Version { get; private set; }
        public Transform RealSpaceFloor { get { return realSpaceFloor; } }
        public bool StoresPointsInRealSpaceFloor { get { return storePointsInRealSpaceFloor && realSpaceFloor != null; } }

        public void Clear()
        {
            Clear(true);
        }

        public void CopyPointsTo(List<Vector3> target)
        {
            if (target == null)
            {
                return;
            }

            target.Clear();
            target.AddRange(points);
        }

        private void Clear(bool notify)
        {
            points.Clear();
            occupiedCells.Clear();

            if (notify)
            {
                NotifyPointsChanged();
            }
        }

        private void Reset()
        {
            ResolveSensorReference();
            ResolveRealSpaceFloorReference();
        }

        private void Awake()
        {
            ResolveSensorReference();
            ResolveRealSpaceFloorReference();
        }

        private void OnEnable()
        {
            ResolveSensorReference();
            ResolveRealSpaceFloorReference();

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
            cellSize = Mathf.Max(0.001f, cellSize);
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
                Clear(false);
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

            NotifyPointsChanged();
        }

        private void AddPoint(Vector3 point)
        {
            var storedPoint = ToStoredPoint(point);

            if (points.Count >= maxPoints)
            {
                return;
            }

            if (rejectDuplicateCells)
            {
                var cell = ToCell(storedPoint);
                if (occupiedCells.Contains(cell))
                {
                    return;
                }

                occupiedCells.Add(cell);
            }

            points.Add(storedPoint);
        }

        private Vector3 ToStoredPoint(Vector3 worldPoint)
        {
            if (storePointsInRealSpaceFloor && realSpaceFloor != null)
            {
                return realSpaceFloor.InverseTransformPoint(worldPoint);
            }

            return worldPoint;
        }

        private Vector3 ToWorldPoint(Vector3 storedPoint)
        {
            if (storePointsInRealSpaceFloor && realSpaceFloor != null)
            {
                return realSpaceFloor.TransformPoint(storedPoint);
            }

            return storedPoint;
        }

        private Vector3Int ToCell(Vector3 storedPoint)
        {
            return new Vector3Int(
                Mathf.FloorToInt(storedPoint.x / cellSize),
                Mathf.FloorToInt(storedPoint.y / cellSize),
                Mathf.FloorToInt(storedPoint.z / cellSize));
        }

        private void NotifyPointsChanged()
        {
            Version++;

            if (PointsChanged != null)
            {
                PointsChanged(this);
            }
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
                Gizmos.DrawSphere(ToWorldPoint(points[i]), pointRadius);
            }
        }

        private void ResolveRealSpaceFloorReference()
        {
            if (realSpaceFloor != null || string.IsNullOrEmpty(realSpaceFloorName))
            {
                return;
            }

            var realSpaceFloorObject = GameObject.Find(realSpaceFloorName);
            if (realSpaceFloorObject != null)
            {
                realSpaceFloor = realSpaceFloorObject.transform;
            }
        }
    }
}
