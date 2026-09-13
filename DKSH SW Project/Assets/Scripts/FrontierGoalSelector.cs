using System;
using System.Collections.Generic;
using System.Diagnostics;
using DKSH.Spiderbot.Mapping;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace DKSH.Spiderbot.Exploration
{
    [DisallowMultipleComponent]
    public sealed class FrontierGoalSelector : MonoBehaviour
    {
        [Header("Inputs")]
        [SerializeField]
        private OccupancyGrid2D occupancyGrid;

        [Tooltip("Optional scan source. It is used only to request a rate-limited reevaluation after a scan.")]
        [SerializeField]
        private LidarOccupancyGridMapper mappingSource;

        [SerializeField]
        private Transform robotTransform;

        [Header("Frontier detection and clustering")]
        [SerializeField]
        private FrontierConnectivity frontierNeighborConnectivity = FrontierConnectivity.Eight;

        [SerializeField]
        private FrontierConnectivity clusterConnectivity = FrontierConnectivity.Eight;

        [SerializeField, Min(1)]
        private int minClusterSize = 3;

        [Header("Candidate score")]
        [SerializeField, Min(0f)]
        private float informationGainWeight = 1f;

        [SerializeField, Min(0f)]
        private float distanceWeight = 1f;

        [Header("Goal stability")]
        [SerializeField, Min(0.05f)]
        private float replanInterval = 1.5f;

        [SerializeField, Min(0f)]
        private float minimumGoalHoldTime = 2f;

        [SerializeField, Min(0f)]
        private float goalSwitchScoreMargin = 1f;

        [SerializeField, Min(0f)]
        private float robotMovementReevaluationDistance = 0.1f;

        [Header("Statistics")]
        [SerializeField]
        private bool logStatistics;

        [SerializeField, Min(1)]
        private int logEveryEvaluations = 20;

        private readonly FrontierDetector detector = new FrontierDetector();
        private readonly List<FrontierCluster> activeClusters = new List<FrontierCluster>(32);
        private readonly Stack<FrontierCluster> clusterPool = new Stack<FrontierCluster>(32);
        private readonly Queue<int> floodFillQueue = new Queue<int>(512);
        private bool[] frontierLookup;
        private bool[] visitedLookup;
        private OccupancyGrid2D subscribedGrid;
        private LidarOccupancyGridMapper subscribedMappingSource;
        private bool evaluationRequested = true;
        private double nextEvaluationTime;
        private double goalSelectedAt;
        private Vector3 lastEvaluationRobotPosition;
        private bool hasLastEvaluationRobotPosition;
        private Func<Vector2Int, bool> goalCellValidator;

        public event Action<FrontierGoalSelector> OnFrontierGoalChanged;

        public OccupancyGrid2D Grid { get { return occupancyGrid; } }
        public Transform RobotTransform { get { return robotTransform; } }
        public IReadOnlyList<Vector2Int> FrontierCells { get { return detector.FrontierCells; } }
        public IReadOnlyList<FrontierCluster> Clusters { get { return activeClusters; } }
        public bool HasValidGoal { get; private set; }
        public Vector3 SelectedGoalWorld { get; private set; }
        public Vector2Int SelectedGoalGrid { get; private set; } = new Vector2Int(-1, -1);
        public FrontierCluster SelectedCluster { get; private set; }
        public Vector3 LastRobotWorldPosition { get; private set; }
        public Vector2Int LastRobotGridPosition { get; private set; } = new Vector2Int(-1, -1);
        public int EvaluationCount { get; private set; }
        public double LastEvaluationMilliseconds { get; private set; }
        public int LastRejectedSmallClusterCount { get; private set; }

        public void Configure(
            OccupancyGrid2D grid,
            LidarOccupancyGridMapper scanSource,
            Transform trackedRobot)
        {
            occupancyGrid = grid;
            mappingSource = scanSource;
            robotTransform = trackedRobot;
            detector.SetGrid(grid);
            SubscribeToSources();
            RequestEvaluation();
        }

        public void ConfigureSettings(
            FrontierConnectivity configuredFrontierConnectivity,
            FrontierConnectivity configuredClusterConnectivity,
            int configuredMinClusterSize,
            float configuredInformationGainWeight,
            float configuredDistanceWeight,
            float configuredReplanInterval,
            float configuredMinimumGoalHoldTime,
            float configuredGoalSwitchScoreMargin)
        {
            frontierNeighborConnectivity = configuredFrontierConnectivity;
            clusterConnectivity = configuredClusterConnectivity;
            minClusterSize = Mathf.Max(1, configuredMinClusterSize);
            informationGainWeight = Mathf.Max(0f, configuredInformationGainWeight);
            distanceWeight = Mathf.Max(0f, configuredDistanceWeight);
            replanInterval = Mathf.Max(0.05f, configuredReplanInterval);
            minimumGoalHoldTime = Mathf.Max(0f, configuredMinimumGoalHoldTime);
            goalSwitchScoreMargin = Mathf.Max(0f, configuredGoalSwitchScoreMargin);
            RequestEvaluation();
        }

        public void RequestEvaluation()
        {
            evaluationRequested = true;
        }

        /// <summary>
        /// Installs an optional reachability check. Candidates are tested in score order
        /// and the first accepted candidate becomes the selected goal.
        /// </summary>
        public void SetGoalCellValidator(Func<Vector2Int, bool> validator)
        {
            goalCellValidator = validator;
            RequestEvaluation();
        }

        /// <summary>
        /// Rebuilds cached Frontier data immediately. forceGoalReselection bypasses
        /// goal hold and switch-margin rules, and is intended for tests or operator commands.
        /// </summary>
        public bool EvaluateNow(bool forceGoalReselection = false)
        {
            ResolveReferences();
            SubscribeToSources();
            var startedAt = Stopwatch.GetTimestamp();
            evaluationRequested = false;
            nextEvaluationTime = CurrentTime + replanInterval;

            ReleaseActiveClusters();
            detector.SetGrid(occupancyGrid);
            detector.Connectivity = frontierNeighborConnectivity;
            detector.DetectFrontierCells();

            if (occupancyGrid == null || !TryResolveRobotPosition(out var robotWorldPosition) ||
                !occupancyGrid.WorldToGrid(robotWorldPosition, out var robotGridX, out var robotGridY))
            {
                LastRejectedSmallClusterCount = 0;
                InvalidateGoal();
                FinishEvaluation(startedAt);
                return false;
            }

            LastRobotWorldPosition = robotWorldPosition;
            LastRobotGridPosition = new Vector2Int(robotGridX, robotGridY);
            lastEvaluationRobotPosition = robotWorldPosition;
            hasLastEvaluationRobotPosition = true;

            BuildClusters(robotWorldPosition);
            activeClusters.Sort(CompareCandidates);
            SelectStableGoal(forceGoalReselection);
            FinishEvaluation(startedAt);
            return HasValidGoal;
        }

        public bool TryGetRobotGridPosition(out Vector2Int gridPosition)
        {
            ResolveReferences();
            if (occupancyGrid != null && TryResolveRobotPosition(out var worldPosition) &&
                occupancyGrid.WorldToGrid(worldPosition, out var x, out var y))
            {
                gridPosition = new Vector2Int(x, y);
                return true;
            }

            gridPosition = new Vector2Int(-1, -1);
            return false;
        }

        private void Reset()
        {
            ResolveReferences();
        }

        private void Awake()
        {
            ResolveReferences();
            detector.SetGrid(occupancyGrid);
        }

        private void OnEnable()
        {
            ResolveReferences();
            SubscribeToSources();
            RequestEvaluation();
        }

        private void OnDisable()
        {
            UnsubscribeFromSources();
        }

        private void Update()
        {
            ResolveReferences();
            SubscribeToSources();

            if (TryResolveRobotPosition(out var currentRobotPosition) &&
                hasLastEvaluationRobotPosition)
            {
                var movementThresholdSquared =
                    robotMovementReevaluationDistance * robotMovementReevaluationDistance;
                var delta = currentRobotPosition - lastEvaluationRobotPosition;
                delta.y = 0f;
                if (delta.sqrMagnitude >= movementThresholdSquared)
                {
                    evaluationRequested = true;
                }
            }

            if (evaluationRequested && CurrentTime >= nextEvaluationTime)
            {
                EvaluateNow();
            }
        }

        private void OnValidate()
        {
            minClusterSize = Mathf.Max(1, minClusterSize);
            informationGainWeight = Mathf.Max(0f, informationGainWeight);
            distanceWeight = Mathf.Max(0f, distanceWeight);
            replanInterval = Mathf.Max(0.05f, replanInterval);
            minimumGoalHoldTime = Mathf.Max(0f, minimumGoalHoldTime);
            goalSwitchScoreMargin = Mathf.Max(0f, goalSwitchScoreMargin);
            robotMovementReevaluationDistance = Mathf.Max(0f, robotMovementReevaluationDistance);
            logEveryEvaluations = Mathf.Max(1, logEveryEvaluations);
            RequestEvaluation();
        }

        private void ResolveReferences()
        {
            if (occupancyGrid == null)
            {
                occupancyGrid = GetComponent<OccupancyGrid2D>();
            }

            if (mappingSource == null)
            {
                mappingSource = GetComponent<LidarOccupancyGridMapper>();
            }

            if (robotTransform == null)
            {
                var renderer = GetComponent<OccupancyGrid2DRenderer>();
                if (renderer != null)
                {
                    robotTransform = renderer.RobotTransform;
                }
            }

            detector.SetGrid(occupancyGrid);
        }

        private void SubscribeToSources()
        {
            if (!isActiveAndEnabled)
            {
                return;
            }

            if (subscribedGrid != occupancyGrid)
            {
                if (subscribedGrid != null)
                {
                    subscribedGrid.CellsChanged -= HandleGridChanged;
                    subscribedGrid.GridReset -= HandleGridReset;
                }

                subscribedGrid = occupancyGrid;
                if (subscribedGrid != null)
                {
                    subscribedGrid.CellsChanged += HandleGridChanged;
                    subscribedGrid.GridReset += HandleGridReset;
                }
            }

            if (subscribedMappingSource != mappingSource)
            {
                if (subscribedMappingSource != null)
                {
                    subscribedMappingSource.ScanIntegrated -= HandleScanIntegrated;
                }

                subscribedMappingSource = mappingSource;
                if (subscribedMappingSource != null)
                {
                    subscribedMappingSource.ScanIntegrated += HandleScanIntegrated;
                }
            }
        }

        private void UnsubscribeFromSources()
        {
            if (subscribedGrid != null)
            {
                subscribedGrid.CellsChanged -= HandleGridChanged;
                subscribedGrid.GridReset -= HandleGridReset;
                subscribedGrid = null;
            }

            if (subscribedMappingSource != null)
            {
                subscribedMappingSource.ScanIntegrated -= HandleScanIntegrated;
                subscribedMappingSource = null;
            }
        }

        private void HandleGridChanged(OccupancyGrid2D grid, IReadOnlyList<int> changedIndices)
        {
            RequestEvaluation();
        }

        private void HandleGridReset(OccupancyGrid2D grid)
        {
            RequestEvaluation();
        }

        private void HandleScanIntegrated(LidarOccupancyGridMapper mapper)
        {
            RequestEvaluation();
        }

        private bool TryResolveRobotPosition(out Vector3 worldPosition)
        {
            if (robotTransform != null)
            {
                worldPosition = robotTransform.position;
                return IsFinite(worldPosition);
            }

            if (mappingSource != null && mappingSource.HasMappingPose)
            {
                worldPosition = mappingSource.LastMappingPose.Position;
                return IsFinite(worldPosition);
            }

            worldPosition = Vector3.zero;
            return false;
        }

        private void BuildClusters(Vector3 robotWorldPosition)
        {
            EnsureLookupStorage();
            Array.Clear(frontierLookup, 0, frontierLookup.Length);
            Array.Clear(visitedLookup, 0, visitedLookup.Length);

            var frontierCells = detector.FrontierCells;
            for (var i = 0; i < frontierCells.Count; i++)
            {
                var cell = frontierCells[i];
                var index = occupancyGrid.GetFlatIndex(cell.x, cell.y);
                if (index >= 0)
                {
                    frontierLookup[index] = true;
                }
            }

            LastRejectedSmallClusterCount = 0;
            var offsets = FrontierDetector.GetNeighborOffsets(clusterConnectivity);
            for (var i = 0; i < frontierCells.Count; i++)
            {
                var seed = frontierCells[i];
                var seedIndex = occupancyGrid.GetFlatIndex(seed.x, seed.y);
                if (seedIndex < 0 || visitedLookup[seedIndex])
                {
                    continue;
                }

                var cluster = AcquireCluster();
                floodFillQueue.Clear();
                floodFillQueue.Enqueue(seedIndex);
                visitedLookup[seedIndex] = true;

                while (floodFillQueue.Count > 0)
                {
                    var currentIndex = floodFillQueue.Dequeue();
                    occupancyGrid.TryGetCoordinates(currentIndex, out var currentX, out var currentY);
                    cluster.AddCell(new Vector2Int(currentX, currentY));

                    for (var neighbor = 0; neighbor < offsets.Length; neighbor++)
                    {
                        var neighborX = currentX + offsets[neighbor].x;
                        var neighborY = currentY + offsets[neighbor].y;
                        var neighborIndex = occupancyGrid.GetFlatIndex(neighborX, neighborY);
                        if (neighborIndex < 0 || visitedLookup[neighborIndex] ||
                            !frontierLookup[neighborIndex])
                        {
                            continue;
                        }

                        visitedLookup[neighborIndex] = true;
                        floodFillQueue.Enqueue(neighborIndex);
                    }
                }

                if (cluster.CellCount < minClusterSize || !cluster.FinalizeCandidate(occupancyGrid))
                {
                    LastRejectedSmallClusterCount++;
                    ReleaseCluster(cluster);
                    continue;
                }

                var horizontalDelta = cluster.GoalWorld - robotWorldPosition;
                horizontalDelta.y = 0f;
                cluster.DistanceFromRobot = horizontalDelta.magnitude;
                cluster.Score = CalculateScore(cluster.CellCount, cluster.DistanceFromRobot);
                activeClusters.Add(cluster);
            }
        }

        private float CalculateScore(int clusterSize, float travelDistance)
        {
            // Kept in one method so the next stage can replace straight-line distance
            // with a path cost without changing detection, clustering, or goal stability.
            return informationGainWeight * clusterSize - distanceWeight * travelDistance;
        }

        private void SelectStableGoal(bool forceGoalReselection)
        {
            var bestCluster = FindBestCluster();
            if (bestCluster == null)
            {
                InvalidateGoal();
                return;
            }

            var currentCluster = HasValidGoal ? FindClusterContaining(SelectedGoalGrid) : null;
            var currentGoalIsValid = currentCluster != null &&
                                     occupancyGrid.GetCell(SelectedGoalGrid.x, SelectedGoalGrid.y) ==
                                     OccupancyCellState.Free &&
                                     IsGoalCellAccepted(SelectedGoalGrid);

            if (!forceGoalReselection && currentGoalIsValid)
            {
                SelectedCluster = currentCluster;
                SelectedGoalWorld = occupancyGrid.GridToWorld(SelectedGoalGrid.x, SelectedGoalGrid.y);

                var holdTimeHasElapsed = CurrentTime - goalSelectedAt >= minimumGoalHoldTime;
                var challengerIsBetter = bestCluster != currentCluster &&
                                         bestCluster.Score >=
                                         currentCluster.Score + goalSwitchScoreMargin;
                if (!holdTimeHasElapsed || !challengerIsBetter)
                {
                    return;
                }
            }

            SetSelectedGoal(bestCluster);
        }

        private FrontierCluster FindBestCluster()
        {
            for (var i = 0; i < activeClusters.Count; i++)
            {
                var candidate = activeClusters[i];
                if (IsGoalCellAccepted(candidate.GoalGrid))
                {
                    return candidate;
                }
            }

            return null;
        }

        private bool IsGoalCellAccepted(Vector2Int goalCell)
        {
            return goalCellValidator == null || goalCellValidator(goalCell);
        }

        private static int CompareCandidates(FrontierCluster first, FrontierCluster second)
        {
            if (ReferenceEquals(first, second))
            {
                return 0;
            }

            if (IsBetterCandidate(first, second))
            {
                return -1;
            }

            return IsBetterCandidate(second, first) ? 1 : 0;
        }

        private static bool IsBetterCandidate(FrontierCluster candidate, FrontierCluster currentBest)
        {
            const float epsilon = 0.0001f;
            if (candidate.Score > currentBest.Score + epsilon)
            {
                return true;
            }

            if (Mathf.Abs(candidate.Score - currentBest.Score) > epsilon)
            {
                return false;
            }

            if (candidate.CellCount != currentBest.CellCount)
            {
                return candidate.CellCount > currentBest.CellCount;
            }

            if (!Mathf.Approximately(candidate.DistanceFromRobot, currentBest.DistanceFromRobot))
            {
                return candidate.DistanceFromRobot < currentBest.DistanceFromRobot;
            }

            return candidate.GoalGrid.y < currentBest.GoalGrid.y ||
                   (candidate.GoalGrid.y == currentBest.GoalGrid.y &&
                    candidate.GoalGrid.x < currentBest.GoalGrid.x);
        }

        private FrontierCluster FindClusterContaining(Vector2Int cell)
        {
            for (var i = 0; i < activeClusters.Count; i++)
            {
                if (activeClusters[i].Contains(cell))
                {
                    return activeClusters[i];
                }
            }

            return null;
        }

        private void SetSelectedGoal(FrontierCluster cluster)
        {
            var newGoal = cluster.GoalGrid;
            var goalChanged = !HasValidGoal || SelectedGoalGrid != newGoal;
            HasValidGoal = true;
            SelectedCluster = cluster;
            SelectedGoalGrid = newGoal;
            SelectedGoalWorld = cluster.GoalWorld;

            if (goalChanged)
            {
                goalSelectedAt = CurrentTime;
                RaiseGoalChanged();
            }
        }

        private void InvalidateGoal()
        {
            var goalChanged = HasValidGoal;
            HasValidGoal = false;
            SelectedCluster = null;
            SelectedGoalGrid = new Vector2Int(-1, -1);
            SelectedGoalWorld = Vector3.zero;
            if (goalChanged)
            {
                RaiseGoalChanged();
            }
        }

        private void RaiseGoalChanged()
        {
            if (OnFrontierGoalChanged != null)
            {
                OnFrontierGoalChanged(this);
            }
        }

        private FrontierCluster AcquireCluster()
        {
            var cluster = clusterPool.Count > 0 ? clusterPool.Pop() : new FrontierCluster();
            cluster.Reset();
            return cluster;
        }

        private void ReleaseCluster(FrontierCluster cluster)
        {
            cluster.Reset();
            clusterPool.Push(cluster);
        }

        private void ReleaseActiveClusters()
        {
            for (var i = 0; i < activeClusters.Count; i++)
            {
                ReleaseCluster(activeClusters[i]);
            }

            activeClusters.Clear();
        }

        private void EnsureLookupStorage()
        {
            var requiredLength = occupancyGrid.CellCount;
            if (frontierLookup == null || frontierLookup.Length != requiredLength)
            {
                frontierLookup = new bool[requiredLength];
                visitedLookup = new bool[requiredLength];
            }
        }

        private void FinishEvaluation(long startedAt)
        {
            EvaluationCount++;
            LastEvaluationMilliseconds =
                (Stopwatch.GetTimestamp() - startedAt) * 1000.0 / Stopwatch.Frequency;

            if (logStatistics && EvaluationCount % logEveryEvaluations == 0)
            {
                Debug.LogFormat(
                    this,
                    "Frontier evaluation {0}: {1:0.###} ms, cells {2}, clusters {3}, " +
                    "small clusters rejected {4}, goal {5}.",
                    EvaluationCount,
                    LastEvaluationMilliseconds,
                    detector.FrontierCells.Count,
                    activeClusters.Count,
                    LastRejectedSmallClusterCount,
                    HasValidGoal ? SelectedGoalGrid.ToString() : "none");
            }
        }

        private static bool IsFinite(Vector3 value)
        {
            return !float.IsNaN(value.x) && !float.IsInfinity(value.x) &&
                   !float.IsNaN(value.y) && !float.IsInfinity(value.y) &&
                   !float.IsNaN(value.z) && !float.IsInfinity(value.z);
        }

        private static double CurrentTime
        {
            get { return Time.realtimeSinceStartupAsDouble; }
        }
    }
}
