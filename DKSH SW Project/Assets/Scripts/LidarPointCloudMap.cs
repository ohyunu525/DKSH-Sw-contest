using System;
using System.Collections.Generic;
using DKSH.Spiderbot.Sensors;
using UnityEngine;
using UnityEngine.Serialization;

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

        [SerializeField, FormerlySerializedAs("rejectDuplicateCells")]
        private bool keepLatestPointPerCell = true;

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
        private readonly Dictionary<Vector3Int, int> cellPointIndices = new Dictionary<Vector3Int, int>();
        private const float ScaleEpsilon = 0.0001f;
        private const float TraversalEpsilon = 0.000001f;

        public event Action<LidarPointCloudMap> PointsChanged;

        public IReadOnlyList<Vector3> Points { get { return points; } }
        public IReadOnlyList<Vector3> StoredPoints { get { return points; } }
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

        public Vector3 StoredPointToVirtualFloorPoint(Vector3 storedPoint, Transform virtualSpaceFloor)
        {
            if (virtualSpaceFloor == null)
            {
                return storedPoint;
            }

            if (StoresPointsInRealSpaceFloor)
            {
                // Stored points are real-floor local. Scale them by the floor
                // size ratio without using virtual inverse transforms.
                return Vector3.Scale(storedPoint, GetRealToVirtualFloorSizeRatio(virtualSpaceFloor));
            }

            return virtualSpaceFloor.InverseTransformPoint(storedPoint);
        }

        private Vector3 GetRealToVirtualFloorSizeRatio(Transform virtualSpaceFloor)
        {
            var realScale = realSpaceFloor.lossyScale;
            var virtualScale = virtualSpaceFloor.lossyScale;

            var xRatio = GetScaleRatio(virtualScale.x, realScale.x);
            var zRatio = GetScaleRatio(virtualScale.z, realScale.z);
            var yRatio = (xRatio + zRatio) * 0.5f;

            return new Vector3(xRatio, yRatio, zRatio);
        }

        private static float GetScaleRatio(float virtualScale, float realScale)
        {
            realScale = Mathf.Abs(realScale);

            if (realScale <= ScaleEpsilon)
            {
                return 1f;
            }

            return Mathf.Abs(virtualScale) / realScale;
        }

        private bool Clear(bool notify)
        {
            if (points.Count == 0 && cellPointIndices.Count == 0)
            {
                return false;
            }

            points.Clear();
            cellPointIndices.Clear();

            if (notify)
            {
                NotifyPointsChanged();
            }

            return true;
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

            var changed = false;

            if (clearOnEachFrame)
            {
                changed |= Clear(false);
            }

            var samples = frame.Samples;
            for (var i = 0; i < samples.Count; i++)
            {
                changed |= ClearFreeSpace(samples[i]);
            }

            for (var i = 0; i < samples.Count; i++)
            {
                var sample = samples[i];
                if (sample.hit || accumulateMisses)
                {
                    changed |= AddPoint(sample.point);
                }
            }

            if (changed)
            {
                NotifyPointsChanged();
            }
        }

        private bool AddPoint(Vector3 point)
        {
            var storedPoint = ToStoredPoint(point);
            var cell = ToCell(storedPoint);

            if (keepLatestPointPerCell)
            {
                return AddOrUpdateCellPoint(storedPoint, cell);
            }

            return AddNewPoint(storedPoint, cell);
        }

        private bool AddOrUpdateCellPoint(Vector3 storedPoint, Vector3Int cell)
        {
            int pointIndex;
            if (cellPointIndices.TryGetValue(cell, out pointIndex))
            {
                if (IsValidPointIndex(pointIndex))
                {
                    if (points[pointIndex] == storedPoint)
                    {
                        return false;
                    }

                    points[pointIndex] = storedPoint;
                    return true;
                }

                cellPointIndices.Remove(cell);
            }

            return AddNewPoint(storedPoint, cell);
        }

        private bool ClearFreeSpace(LidarSample sample)
        {
            if (sample.distance <= 0f)
            {
                return false;
            }

            var storedOrigin = ToStoredPoint(sample.origin);
            var storedEndPoint = ToStoredPoint(sample.point);

            return ClearCellsAlongSegment(storedOrigin, storedEndPoint, sample.hit);
        }

        private bool ClearCellsAlongSegment(Vector3 start, Vector3 end, bool excludeEndCell)
        {
            var delta = end - start;
            if (delta.sqrMagnitude <= TraversalEpsilon * TraversalEpsilon)
            {
                return false;
            }

            var currentCell = ToCell(start);
            var endCell = ToCell(end);
            var stepX = GetStep(delta.x);
            var stepY = GetStep(delta.y);
            var stepZ = GetStep(delta.z);
            var tMaxX = GetInitialBoundaryT(start.x, currentCell.x, delta.x, stepX);
            var tMaxY = GetInitialBoundaryT(start.y, currentCell.y, delta.y, stepY);
            var tMaxZ = GetInitialBoundaryT(start.z, currentCell.z, delta.z, stepZ);
            var tDeltaX = GetDeltaT(delta.x);
            var tDeltaY = GetDeltaT(delta.y);
            var tDeltaZ = GetDeltaT(delta.z);
            var maxSteps =
                Mathf.Abs(endCell.x - currentCell.x) +
                Mathf.Abs(endCell.y - currentCell.y) +
                Mathf.Abs(endCell.z - currentCell.z) +
                1;
            var changed = false;

            for (var step = 0; step <= maxSteps; step++)
            {
                if (excludeEndCell && currentCell == endCell)
                {
                    break;
                }

                changed |= RemoveCell(currentCell);

                if (currentCell == endCell)
                {
                    break;
                }

                var nextT = Mathf.Min(tMaxX, Mathf.Min(tMaxY, tMaxZ));
                if (float.IsPositiveInfinity(nextT) || nextT > 1f + TraversalEpsilon)
                {
                    break;
                }

                if (stepX != 0 && tMaxX <= nextT + TraversalEpsilon)
                {
                    currentCell.x += stepX;
                    tMaxX += tDeltaX;
                }

                if (stepY != 0 && tMaxY <= nextT + TraversalEpsilon)
                {
                    currentCell.y += stepY;
                    tMaxY += tDeltaY;
                }

                if (stepZ != 0 && tMaxZ <= nextT + TraversalEpsilon)
                {
                    currentCell.z += stepZ;
                    tMaxZ += tDeltaZ;
                }
            }

            return changed;
        }

        private bool RemoveCell(Vector3Int cell)
        {
            int pointIndex;
            if (!cellPointIndices.TryGetValue(cell, out pointIndex))
            {
                return false;
            }

            if (!IsValidPointIndex(pointIndex))
            {
                cellPointIndices.Remove(cell);
                return false;
            }

            RemovePointAt(pointIndex, cell);
            return true;
        }

        private void RemovePointAt(int pointIndex, Vector3Int cell)
        {
            var lastIndex = points.Count - 1;
            var lastPoint = points[lastIndex];

            cellPointIndices.Remove(cell);

            if (pointIndex != lastIndex)
            {
                points[pointIndex] = lastPoint;
                cellPointIndices[ToCell(lastPoint)] = pointIndex;
            }

            points.RemoveAt(lastIndex);
        }

        private bool AddNewPoint(Vector3 storedPoint, Vector3Int cell)
        {
            if (points.Count >= maxPoints)
            {
                return false;
            }

            cellPointIndices[cell] = points.Count;
            points.Add(storedPoint);
            return true;
        }

        private bool IsValidPointIndex(int pointIndex)
        {
            return pointIndex >= 0 && pointIndex < points.Count;
        }

        private int GetStep(float delta)
        {
            if (delta > 0f)
            {
                return 1;
            }

            if (delta < 0f)
            {
                return -1;
            }

            return 0;
        }

        private float GetInitialBoundaryT(float start, int cell, float delta, int step)
        {
            if (step == 0)
            {
                return float.PositiveInfinity;
            }

            var nextBoundary = step > 0 ? (cell + 1) * cellSize : cell * cellSize;
            return Mathf.Max(0f, (nextBoundary - start) / delta);
        }

        private float GetDeltaT(float delta)
        {
            if (Mathf.Approximately(delta, 0f))
            {
                return float.PositiveInfinity;
            }

            return Mathf.Abs(cellSize / delta);
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
