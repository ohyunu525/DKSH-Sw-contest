using System.Collections.Generic;
using DKSH.Spiderbot.Mapping;
using UnityEngine;

namespace DKSH.Spiderbot.Exploration
{
    public enum FrontierExplorerState
    {
        Idle,
        Moving,
        Complete,
        Failed
    }

    /// <summary>
    /// Owns the single standalone exploration loop: select a reachable Frontier,
    /// plan a grid path, follow it, and request a new goal. ROS2/Nav2 replaces this
    /// component in the production robot stack.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class FrontierExplorerCoordinator : MonoBehaviour
    {
        [Header("Inputs")]
        [SerializeField]
        private OccupancyGrid2D occupancyGrid;

        [SerializeField]
        private FrontierGoalSelector goalSelector;

        [SerializeField]
        private FrontierPathFollower pathFollower;

        [SerializeField]
        private Transform robotTransform;

        [Header("Standalone planner")]
        [SerializeField, Min(0f)]
        private float clearanceMeters = 0.25f;

        [SerializeField, Min(0f)]
        private float scanSettleTimeSeconds = 0.35f;

        private readonly GridAStarPlanner planner = new GridAStarPlanner();
        private readonly List<Vector2Int> currentPath = new List<Vector2Int>(128);
        private readonly List<Vector2Int> planningScratch = new List<Vector2Int>(128);
        private readonly List<Vector2Int> reachabilityScratch = new List<Vector2Int>(128);
        private readonly HashSet<Vector2Int> failedGoalCells = new HashSet<Vector2Int>();
        private FrontierGoalSelector subscribedSelector;
        private FrontierPathFollower subscribedFollower;
        private bool planRequested = true;
        private bool forceGoalReselection;
        private bool isPlanning;
        private float nextPlanTime;
        private int observedGridVersion = -1;

        public OccupancyGrid2D Grid { get { return occupancyGrid; } }
        public FrontierGoalSelector GoalSelector { get { return goalSelector; } }
        public FrontierPathFollower PathFollower { get { return pathFollower; } }
        public IReadOnlyList<Vector2Int> CurrentPath { get { return currentPath; } }
        public FrontierExplorerState State { get; private set; } = FrontierExplorerState.Idle;
        public Vector2Int CurrentGoalGrid
        {
            get
            {
                return goalSelector != null && goalSelector.HasValidGoal
                    ? goalSelector.SelectedGoalGrid
                    : new Vector2Int(-1, -1);
            }
        }
        public string LastFailureReason { get; private set; } = string.Empty;
        public int FailedGoalCount { get { return failedGoalCells.Count; } }

        public void Configure(
            OccupancyGrid2D grid,
            FrontierGoalSelector selector,
            FrontierPathFollower follower,
            Transform trackedRobot)
        {
            occupancyGrid = grid;
            goalSelector = selector;
            pathFollower = follower;
            robotTransform = trackedRobot;
            SubscribeToSources();
            RequestPlan(true);
        }

        public void ConfigureSettings(float configuredClearanceMeters, float configuredScanSettleTimeSeconds)
        {
            clearanceMeters = Mathf.Max(0f, configuredClearanceMeters);
            scanSettleTimeSeconds = Mathf.Max(0f, configuredScanSettleTimeSeconds);
            RequestPlan(true);
        }

        public void RequestPlan(bool forceReselection = false)
        {
            planRequested = true;
            forceGoalReselection |= forceReselection;
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
            SubscribeToSources();
            observedGridVersion = occupancyGrid != null ? occupancyGrid.Version : -1;
            RequestPlan(true);
        }

        private void OnDisable()
        {
            if (subscribedSelector != null)
            {
                subscribedSelector.OnFrontierGoalChanged -= HandleFrontierGoalChanged;
                subscribedSelector.SetGoalCellValidator(null);
                subscribedSelector = null;
            }

            if (subscribedFollower != null)
            {
                subscribedFollower.PathCompleted -= HandlePathCompleted;
                subscribedFollower.PathBlocked -= HandlePathBlocked;
                subscribedFollower = null;
            }
        }

        private void Update()
        {
            ResolveReferences();
            SubscribeToSources();

            if (occupancyGrid != null && occupancyGrid.Version != observedGridVersion)
            {
                observedGridVersion = occupancyGrid.Version;
                failedGoalCells.Clear();
                if (State == FrontierExplorerState.Complete || State == FrontierExplorerState.Failed)
                {
                    RequestPlan(true);
                }
            }

            if (planRequested && !isPlanning && Time.time >= nextPlanTime)
            {
                EvaluateAndStartPath();
            }
        }

        private void EvaluateAndStartPath()
        {
            isPlanning = true;
            planRequested = false;
            var forceReselection = forceGoalReselection;
            forceGoalReselection = false;

            try
            {
                ResolveReferences();
                if (occupancyGrid == null || goalSelector == null || pathFollower == null ||
                    robotTransform == null)
                {
                    SetFailed("Standalone exploration references are incomplete.");
                    return;
                }

                if (!goalSelector.EvaluateNow(forceReselection))
                {
                    pathFollower.Stop();
                    currentPath.Clear();
                    if (goalSelector.Clusters.Count == 0)
                    {
                        State = FrontierExplorerState.Complete;
                        LastFailureReason = string.Empty;
                    }
                    else
                    {
                        SetFailed("Frontier candidates exist, but none is reachable with the current clearance.");
                    }

                    return;
                }

                if (!occupancyGrid.WorldToGrid(robotTransform.position, out var startX, out var startY))
                {
                    SetFailed("Robot position is outside the occupancy grid.");
                    return;
                }

                var start = new Vector2Int(startX, startY);
                var goal = goalSelector.SelectedGoalGrid;
                if (!planner.TryFindPath(occupancyGrid, start, goal, clearanceMeters, planningScratch))
                {
                    failedGoalCells.Add(goal);
                    SetFailed("The selected Frontier became unreachable before path execution.");
                    RequestPlan(true);
                    return;
                }

                currentPath.Clear();
                currentPath.AddRange(planningScratch);
                LastFailureReason = string.Empty;
                State = FrontierExplorerState.Moving;
                pathFollower.SetPath(occupancyGrid, currentPath);
            }
            finally
            {
                isPlanning = false;
            }
        }

        private bool CanReachGoal(Vector2Int goalCell)
        {
            if (failedGoalCells.Contains(goalCell) || occupancyGrid == null || robotTransform == null ||
                !occupancyGrid.WorldToGrid(robotTransform.position, out var startX, out var startY))
            {
                return false;
            }

            return planner.TryFindPath(
                occupancyGrid,
                new Vector2Int(startX, startY),
                goalCell,
                clearanceMeters,
                reachabilityScratch);
        }

        private void HandleFrontierGoalChanged(FrontierGoalSelector selector)
        {
            if (!isPlanning)
            {
                RequestPlan();
            }
        }

        private void HandlePathCompleted(FrontierPathFollower follower)
        {
            State = FrontierExplorerState.Idle;
            nextPlanTime = Time.time + scanSettleTimeSeconds;
            RequestPlan(true);
        }

        private void HandlePathBlocked(FrontierPathFollower follower)
        {
            if (CurrentGoalGrid.x >= 0)
            {
                failedGoalCells.Add(CurrentGoalGrid);
            }

            currentPath.Clear();
            SetFailed("Path following stopped because the robot made no progress.");
            RequestPlan(true);
        }

        private void SetFailed(string reason)
        {
            if (pathFollower != null)
            {
                pathFollower.Stop();
            }

            State = FrontierExplorerState.Failed;
            LastFailureReason = reason;
        }

        private void ResolveReferences()
        {
            if (occupancyGrid == null)
            {
                occupancyGrid = GetComponent<OccupancyGrid2D>();
            }

            if (goalSelector == null)
            {
                goalSelector = GetComponent<FrontierGoalSelector>();
            }

            if (robotTransform == null && goalSelector != null)
            {
                robotTransform = goalSelector.RobotTransform;
            }

            if (pathFollower == null && robotTransform != null)
            {
                pathFollower = robotTransform.GetComponent<FrontierPathFollower>();
            }
        }

        private void SubscribeToSources()
        {
            if (!isActiveAndEnabled)
            {
                return;
            }

            if (subscribedSelector != goalSelector)
            {
                if (subscribedSelector != null)
                {
                    subscribedSelector.OnFrontierGoalChanged -= HandleFrontierGoalChanged;
                    subscribedSelector.SetGoalCellValidator(null);
                }

                subscribedSelector = goalSelector;
                if (subscribedSelector != null)
                {
                    subscribedSelector.SetGoalCellValidator(CanReachGoal);
                    subscribedSelector.OnFrontierGoalChanged += HandleFrontierGoalChanged;
                }
            }

            if (subscribedFollower != pathFollower)
            {
                if (subscribedFollower != null)
                {
                    subscribedFollower.PathCompleted -= HandlePathCompleted;
                    subscribedFollower.PathBlocked -= HandlePathBlocked;
                }

                subscribedFollower = pathFollower;
                if (subscribedFollower != null)
                {
                    subscribedFollower.PathCompleted += HandlePathCompleted;
                    subscribedFollower.PathBlocked += HandlePathBlocked;
                }
            }
        }

        private void OnValidate()
        {
            clearanceMeters = Mathf.Max(0f, clearanceMeters);
            scanSettleTimeSeconds = Mathf.Max(0f, scanSettleTimeSeconds);
            ResolveReferences();
        }
    }
}
