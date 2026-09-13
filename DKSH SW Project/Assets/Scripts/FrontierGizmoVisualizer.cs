using UnityEngine;

namespace DKSH.Spiderbot.Exploration
{
    [DisallowMultipleComponent]
    public sealed class FrontierGizmoVisualizer : MonoBehaviour
    {
        [SerializeField]
        private FrontierGoalSelector goalSelector;

        [SerializeField]
        private FrontierExplorerCoordinator explorerCoordinator;

        [Header("Visibility")]
        [SerializeField]
        private bool drawFrontierCells = true;

        [SerializeField]
        private bool drawClusterCenters = true;

        [SerializeField]
        private bool drawSelectedGoal = true;

        [SerializeField]
        private bool drawRobotPosition = true;

        [SerializeField]
        private bool drawCurrentPath = true;

        [Header("Colors")]
        [SerializeField]
        private Color frontierCellColor = new Color(0f, 0.95f, 0.9f, 0.85f);

        [SerializeField]
        private Color clusterCenterColor = new Color(1f, 0.85f, 0f, 0.95f);

        [SerializeField]
        private Color selectedGoalColor = new Color(1f, 0.05f, 0.55f, 1f);

        [SerializeField]
        private Color robotPositionColor = new Color(0.1f, 0.45f, 1f, 1f);

        [SerializeField]
        private Color currentPathColor = new Color(0.25f, 1f, 0.25f, 1f);

        [Header("Marker sizes")]
        [SerializeField, Range(0.1f, 1f)]
        private float frontierCellScale = 0.65f;

        [SerializeField, Min(0.02f)]
        private float clusterCenterRadius = 0.08f;

        [SerializeField, Min(0.02f)]
        private float selectedGoalRadius = 0.18f;

        [SerializeField, Min(0f)]
        private float heightOffset = 0.08f;

        public FrontierGoalSelector GoalSelector { get { return goalSelector; } }
        public FrontierExplorerCoordinator ExplorerCoordinator { get { return explorerCoordinator; } }

        public void Configure(FrontierGoalSelector selector)
        {
            goalSelector = selector;
        }

        public void Configure(FrontierGoalSelector selector, FrontierExplorerCoordinator coordinator)
        {
            goalSelector = selector;
            explorerCoordinator = coordinator;
        }

        private void Reset()
        {
            ResolveReference();
        }

        private void OnValidate()
        {
            frontierCellScale = Mathf.Clamp(frontierCellScale, 0.1f, 1f);
            clusterCenterRadius = Mathf.Max(0.02f, clusterCenterRadius);
            selectedGoalRadius = Mathf.Max(0.02f, selectedGoalRadius);
            heightOffset = Mathf.Max(0f, heightOffset);
            ResolveReference();
        }

        private void OnDrawGizmos()
        {
            ResolveReference();
            if (goalSelector == null || goalSelector.Grid == null)
            {
                return;
            }

            var grid = goalSelector.Grid;
            var markerY = grid.MapOrigin.y + heightOffset;

            if (drawFrontierCells)
            {
                Gizmos.color = frontierCellColor;
                var cells = goalSelector.FrontierCells;
                var size = new Vector3(
                    grid.Resolution * frontierCellScale,
                    0.012f,
                    grid.Resolution * frontierCellScale);
                for (var i = 0; i < cells.Count; i++)
                {
                    var position = grid.GridToWorld(cells[i].x, cells[i].y);
                    position.y = markerY;
                    Gizmos.DrawCube(position, size);
                }
            }

            if (drawClusterCenters)
            {
                Gizmos.color = clusterCenterColor;
                var clusters = goalSelector.Clusters;
                for (var i = 0; i < clusters.Count; i++)
                {
                    var center = clusters[i].CentroidWorld;
                    center.y = markerY + 0.02f;
                    Gizmos.DrawSphere(center, clusterCenterRadius);
                }
            }

            if (drawSelectedGoal && goalSelector.HasValidGoal)
            {
                Gizmos.color = selectedGoalColor;
                var goal = goalSelector.SelectedGoalWorld;
                goal.y = markerY + 0.04f;
                Gizmos.DrawSphere(goal, selectedGoalRadius);
                Gizmos.DrawWireSphere(goal, selectedGoalRadius * 1.5f);
            }

            if (drawRobotPosition && goalSelector.TryGetRobotGridPosition(out var robotGrid))
            {
                Gizmos.color = robotPositionColor;
                var robot = grid.GridToWorld(robotGrid.x, robotGrid.y);
                robot.y = markerY + 0.06f;
                Gizmos.DrawWireSphere(robot, selectedGoalRadius * 0.8f);
            }

            if (drawCurrentPath && explorerCoordinator != null)
            {
                var path = explorerCoordinator.CurrentPath;
                Gizmos.color = currentPathColor;
                for (var i = 1; i < path.Count; i++)
                {
                    var from = grid.GridToWorld(path[i - 1].x, path[i - 1].y);
                    var to = grid.GridToWorld(path[i].x, path[i].y);
                    from.y = markerY + 0.1f;
                    to.y = markerY + 0.1f;
                    Gizmos.DrawLine(from, to);
                    Gizmos.DrawWireSphere(to, grid.Resolution * 0.2f);
                }
            }
        }

        private void ResolveReference()
        {
            if (goalSelector == null)
            {
                goalSelector = GetComponent<FrontierGoalSelector>();
            }

            if (explorerCoordinator == null)
            {
                explorerCoordinator = GetComponent<FrontierExplorerCoordinator>();
            }
        }
    }
}
