using System;
using System.Collections.Generic;
using System.Diagnostics;
using DKSH.Spiderbot.Sensors;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace DKSH.Spiderbot.Mapping
{
    [DisallowMultipleComponent]
    public sealed class LidarOccupancyGridMapper : MonoBehaviour
    {
        [SerializeField]
        private SimulatedLidarSensor lidarSensor;

        [SerializeField]
        private OccupancyGrid2D occupancyGrid;

        [Tooltip("Optional MonoBehaviour implementing ILidarMappingPoseProvider. " +
                 "When empty, the scan frame's Unity Transform known pose is used.")]
        [SerializeField]
        private MonoBehaviour poseProviderComponent;

        [Header("2D projection")]
        [SerializeField, Range(0.1f, 90f)]
        private float maximumAbsoluteRayElevationDegrees = 6f;

        [SerializeField]
        private float minimumOccupiedHeightRelativeToSensor = -0.1f;

        [SerializeField]
        private float maximumOccupiedHeightRelativeToSensor = 1.5f;

        [SerializeField, Min(8)]
        private int maximumTraversalCellsPerRay = 4096;

        [Header("Statistics")]
        [SerializeField]
        private bool logStatistics;

        [SerializeField, Min(1)]
        private int logEveryScans = 30;

        private readonly List<int> freeCellCandidates = new List<int>(4096);
        private readonly List<int> occupiedCellCandidates = new List<int>(512);
        private int[] freeCellVisitTokens;
        private int[] occupiedCellVisitTokens;
        private int scanToken;
        private bool warnedAboutInvalidPoseProvider;
        private SimulatedLidarSensor subscribedSensor;

        public event Action<LidarOccupancyGridMapper> ScanIntegrated;

        public OccupancyGrid2D Grid { get { return occupancyGrid; } }
        public bool UsesKnownPoseFallback { get { return GetPoseProvider() == null; } }
        public bool HasMappingPose { get; private set; }
        public LidarMappingPose LastMappingPose { get; private set; }
        public double LastScanTimestamp { get; private set; }
        public double LastScanMilliseconds { get; private set; }
        public int LastRayCount { get; private set; }
        public int LastIntegratedRayCount { get; private set; }
        public int LastChangedCellCount { get; private set; }
        public int LastFreeCandidateCount { get; private set; }
        public int LastOccupiedCandidateCount { get; private set; }
        public int TotalScansProcessed { get; private set; }

        public void Configure(
            SimulatedLidarSensor sensor,
            OccupancyGrid2D grid,
            MonoBehaviour poseProvider)
        {
            lidarSensor = sensor;
            occupancyGrid = grid;
            poseProviderComponent = poseProvider;
            warnedAboutInvalidPoseProvider = false;
            SubscribeToSensor();
        }

        public bool ProcessScan(LidarScanFrame frame)
        {
            if (frame == null)
            {
                return false;
            }

            ResolveReferences();
            if (occupancyGrid == null)
            {
                return false;
            }

            if (!TryResolveMappingPose(frame, out var mappingPose))
            {
                return false;
            }

            var startedAt = Stopwatch.GetTimestamp();
            PrepareCandidateStorage();

            var samples = frame.Samples;
            var integratedRayCount = 0;
            var inverseRayRotation = Quaternion.Inverse(frame.RayRotation);

            for (var i = 0; i < samples.Count; i++)
            {
                var sample = samples[i];
                var localDirection = inverseRayRotation * sample.direction;
                if (localDirection.sqrMagnitude <= 0.000001f)
                {
                    continue;
                }

                localDirection.Normalize();
                var elevationDegrees = Mathf.Abs(
                    Mathf.Asin(Mathf.Clamp(localDirection.y, -1f, 1f)) * Mathf.Rad2Deg);
                if (elevationDegrees > maximumAbsoluteRayElevationDegrees)
                {
                    continue;
                }

                var mapDirection = mappingPose.Rotation * localDirection;
                var rayEnd = mappingPose.Position + mapDirection * Mathf.Max(0f, sample.distance);
                if (!IsFinite(mappingPose.Position) || !IsFinite(rayEnd))
                {
                    continue;
                }

                var relativeHitHeight = rayEnd.y - mappingPose.Position.y;
                var endpointIsOccupied = sample.hit &&
                    relativeHitHeight >= minimumOccupiedHeightRelativeToSensor &&
                    relativeHitHeight <= maximumOccupiedHeightRelativeToSensor;

                AddRayCandidates(mappingPose.Position, rayEnd, endpointIsOccupied);
                integratedRayCount++;
            }

            occupancyGrid.BeginBatchUpdate();
            for (var i = 0; i < freeCellCandidates.Count; i++)
            {
                var index = freeCellCandidates[i];
                if (occupiedCellVisitTokens[index] != scanToken)
                {
                    occupancyGrid.MarkFreeByIndex(index);
                }
            }

            for (var i = 0; i < occupiedCellCandidates.Count; i++)
            {
                occupancyGrid.MarkOccupiedByIndex(occupiedCellCandidates[i]);
            }

            LastChangedCellCount = occupancyGrid.EndBatchUpdate();
            LastMappingPose = mappingPose;
            HasMappingPose = true;
            LastScanTimestamp = frame.Timestamp;
            LastRayCount = samples.Count;
            LastIntegratedRayCount = integratedRayCount;
            LastFreeCandidateCount = freeCellCandidates.Count;
            LastOccupiedCandidateCount = occupiedCellCandidates.Count;
            TotalScansProcessed++;
            LastScanMilliseconds = ElapsedMilliseconds(startedAt);

            if (logStatistics && TotalScansProcessed % logEveryScans == 0)
            {
                Debug.LogFormat(
                    this,
                    "Known-pose occupancy mapping scan {0}: {1:0.###} ms, rays {2}/{3}, " +
                    "free candidates {4}, occupied candidates {5}, changed cells {6}.",
                    TotalScansProcessed,
                    LastScanMilliseconds,
                    LastIntegratedRayCount,
                    LastRayCount,
                    LastFreeCandidateCount,
                    LastOccupiedCandidateCount,
                    LastChangedCellCount);
            }

            if (ScanIntegrated != null)
            {
                ScanIntegrated(this);
            }

            return true;
        }

        private void Reset()
        {
            ResolveReferences();
        }

        private void Awake()
        {
            ResolveReferences();
        }

        private void OnEnable()
        {
            ResolveReferences();
            SubscribeToSensor();
        }

        private void OnDisable()
        {
            UnsubscribeFromSensor();
        }

        private void OnValidate()
        {
            maximumAbsoluteRayElevationDegrees = Mathf.Clamp(
                maximumAbsoluteRayElevationDegrees,
                0.1f,
                90f);
            maximumOccupiedHeightRelativeToSensor = Mathf.Max(
                minimumOccupiedHeightRelativeToSensor,
                maximumOccupiedHeightRelativeToSensor);
            maximumTraversalCellsPerRay = Mathf.Max(8, maximumTraversalCellsPerRay);
            logEveryScans = Mathf.Max(1, logEveryScans);
            warnedAboutInvalidPoseProvider = false;
        }

        private void ResolveReferences()
        {
            if (lidarSensor == null)
            {
                lidarSensor = GetComponent<SimulatedLidarSensor>();
            }

            if (lidarSensor == null)
            {
                lidarSensor = GetComponentInParent<SimulatedLidarSensor>();
            }

            if (occupancyGrid == null)
            {
                occupancyGrid = GetComponent<OccupancyGrid2D>();
            }

            if (occupancyGrid == null)
            {
                occupancyGrid = FindAnyObjectByType<OccupancyGrid2D>();
            }
        }

        private void SubscribeToSensor()
        {
            if (!isActiveAndEnabled || subscribedSensor == lidarSensor)
            {
                return;
            }

            UnsubscribeFromSensor();
            subscribedSensor = lidarSensor;
            if (subscribedSensor != null)
            {
                subscribedSensor.ScanCompleted += HandleScanCompleted;
            }
        }

        private void UnsubscribeFromSensor()
        {
            if (subscribedSensor == null)
            {
                return;
            }

            subscribedSensor.ScanCompleted -= HandleScanCompleted;
            subscribedSensor = null;
        }

        private void HandleScanCompleted(LidarScanFrame frame)
        {
            ProcessScan(frame);
        }

        private bool TryResolveMappingPose(LidarScanFrame frame, out LidarMappingPose pose)
        {
            var provider = GetPoseProvider();
            if (provider != null)
            {
                return provider.TryGetSensorPose(frame, out pose);
            }

            pose = LidarMappingPose.FromKnownPose(frame);
            return true;
        }

        private ILidarMappingPoseProvider GetPoseProvider()
        {
            if (poseProviderComponent == null)
            {
                return null;
            }

            var provider = poseProviderComponent as ILidarMappingPoseProvider;
            if (provider == null && !warnedAboutInvalidPoseProvider)
            {
                warnedAboutInvalidPoseProvider = true;
                Debug.LogWarning(
                    "Assigned pose provider does not implement ILidarMappingPoseProvider. " +
                    "The mapper will use the scan frame's Unity known pose.",
                    this);
            }

            return provider;
        }

        private void PrepareCandidateStorage()
        {
            var cellCount = occupancyGrid.CellCount;
            if (freeCellVisitTokens == null || freeCellVisitTokens.Length != cellCount)
            {
                freeCellVisitTokens = new int[cellCount];
                occupiedCellVisitTokens = new int[cellCount];
                scanToken = 0;
            }

            if (scanToken == int.MaxValue)
            {
                Array.Clear(freeCellVisitTokens, 0, freeCellVisitTokens.Length);
                Array.Clear(occupiedCellVisitTokens, 0, occupiedCellVisitTokens.Length);
                scanToken = 0;
            }

            scanToken++;
            freeCellCandidates.Clear();
            occupiedCellCandidates.Clear();
        }

        private void AddRayCandidates(Vector3 rayStart, Vector3 rayEnd, bool endpointIsOccupied)
        {
            occupancyGrid.WorldToGridUnchecked(rayStart, out var x0, out var y0);
            occupancyGrid.WorldToGridUnchecked(rayEnd, out var x1, out var y1);

            var x = x0;
            var y = y0;
            var deltaX = Mathf.Abs(x1 - x0);
            var stepX = x0 < x1 ? 1 : -1;
            var deltaY = -Mathf.Abs(y1 - y0);
            var stepY = y0 < y1 ? 1 : -1;
            var error = deltaX + deltaY;

            for (var step = 0; step < maximumTraversalCellsPerRay; step++)
            {
                var isEndpoint = x == x1 && y == y1;
                if (!(isEndpoint && endpointIsOccupied))
                {
                    AddFreeCandidate(x, y);
                }

                if (isEndpoint)
                {
                    break;
                }

                var doubledError = error * 2;
                if (doubledError >= deltaY)
                {
                    error += deltaY;
                    x += stepX;
                }

                if (doubledError <= deltaX)
                {
                    error += deltaX;
                    y += stepY;
                }
            }

            if (endpointIsOccupied)
            {
                AddOccupiedCandidate(x1, y1);
            }
        }

        private void AddFreeCandidate(int x, int y)
        {
            var index = occupancyGrid.GetFlatIndex(x, y);
            if (index < 0 || freeCellVisitTokens[index] == scanToken)
            {
                return;
            }

            freeCellVisitTokens[index] = scanToken;
            freeCellCandidates.Add(index);
        }

        private void AddOccupiedCandidate(int x, int y)
        {
            var index = occupancyGrid.GetFlatIndex(x, y);
            if (index < 0 || occupiedCellVisitTokens[index] == scanToken)
            {
                return;
            }

            occupiedCellVisitTokens[index] = scanToken;
            occupiedCellCandidates.Add(index);
        }

        private static bool IsFinite(Vector3 value)
        {
            return IsFinite(value.x) && IsFinite(value.y) && IsFinite(value.z);
        }

        private static bool IsFinite(float value)
        {
            return !float.IsNaN(value) && !float.IsInfinity(value);
        }

        private static double ElapsedMilliseconds(long startedAt)
        {
            return (Stopwatch.GetTimestamp() - startedAt) * 1000.0 / Stopwatch.Frequency;
        }
    }
}
