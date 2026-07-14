using System.Collections.Generic;
using UnityEngine;

namespace DKSH.Spiderbot.Training
{
    [DisallowMultipleComponent]
    public sealed class GeneratedTrainingEnvironment : MonoBehaviour
    {
        [SerializeField]
        private RLMapGenerator generator;

        [SerializeField]
        private int environmentIndex;

        [SerializeField]
        private int seed;

        [SerializeField]
        private RLMapLevel level;

        [SerializeField]
        private Transform environmentRoot;

        [SerializeField]
        private Transform geometryRoot;

        [SerializeField]
        private Transform obstaclesRoot;

        [SerializeField]
        private Transform hazardsRoot;

        [SerializeField]
        private Transform startPoint;

        [SerializeField]
        private Transform targetPoint;

        [SerializeField]
        private Vector2Int mapSize;

        [SerializeField]
        private float cellSize = 1f;

        [SerializeField]
        private int floorCount = 1;

        [SerializeField]
        private float floorHeight = 3f;

        [SerializeField]
        private Vector2Int startCell;

        [SerializeField]
        private Vector2Int targetCell;

        [SerializeField]
        private Vector3Int startNode;

        [SerializeField]
        private Vector3Int targetNode;

        [SerializeField]
        private Vector2Int[] obstacleCells = new Vector2Int[0];

        [SerializeField]
        private Vector2Int[] hazardCells = new Vector2Int[0];

        [SerializeField]
        private Vector2Int[] collapseCells = new Vector2Int[0];

        [SerializeField]
        private Vector3Int[] walkableNodes = new Vector3Int[0];

        [SerializeField]
        private Vector3Int[] stairNodes = new Vector3Int[0];

        private static readonly Vector2Int[] PathDirections =
        {
            new Vector2Int(1, 0),
            new Vector2Int(-1, 0),
            new Vector2Int(0, 1),
            new Vector2Int(0, -1)
        };

        public RLMapGenerator Generator { get { return generator; } }
        public int EnvironmentIndex { get { return environmentIndex; } }
        public int Seed { get { return seed; } }
        public RLMapLevel Level { get { return level; } }
        public Transform EnvironmentRoot { get { return environmentRoot; } }
        public Transform GeometryRoot { get { return geometryRoot; } }
        public Transform ObstaclesRoot { get { return obstaclesRoot; } }
        public Transform HazardsRoot { get { return hazardsRoot; } }
        public Transform StartPoint { get { return startPoint; } }
        public Transform TargetPoint { get { return targetPoint; } }
        public Vector2Int MapSize { get { return mapSize; } }
        public float CellSize { get { return cellSize; } }
        public int FloorCount { get { return floorCount; } }
        public float FloorHeight { get { return floorHeight; } }
        public Vector2Int StartCell { get { return startCell; } }
        public Vector2Int TargetCell { get { return targetCell; } }
        public Vector3Int StartNode { get { return startNode; } }
        public Vector3Int TargetNode { get { return targetNode; } }
        public IReadOnlyList<Vector2Int> ObstacleCells { get { return obstacleCells; } }
        public IReadOnlyList<Vector2Int> HazardCells { get { return hazardCells; } }
        public IReadOnlyList<Vector2Int> CollapseCells { get { return collapseCells; } }
        public IReadOnlyList<Vector3Int> WalkableNodes { get { return walkableNodes; } }
        public IReadOnlyList<Vector3Int> StairNodes { get { return stairNodes; } }
        public bool UsesLayeredNavigation { get { return walkableNodes != null && walkableNodes.Length > 0; } }

        internal void Initialize(
            RLMapGenerator owner,
            int generatedEnvironmentIndex,
            int generatedSeed,
            RLMapLevel generatedLevel,
            Transform generatedEnvironmentRoot,
            Transform generatedGeometryRoot,
            Transform generatedObstaclesRoot,
            Transform generatedHazardsRoot,
            Transform generatedStartPoint,
            Transform generatedTargetPoint,
            Vector2Int generatedMapSize,
            float generatedCellSize,
            int generatedFloorCount,
            float generatedFloorHeight,
            Vector2Int generatedStartCell,
            Vector2Int generatedTargetCell,
            Vector3Int generatedStartNode,
            Vector3Int generatedTargetNode,
            Vector2Int[] generatedObstacleCells,
            Vector2Int[] generatedHazardCells,
            Vector2Int[] generatedCollapseCells,
            Vector3Int[] generatedWalkableNodes,
            Vector3Int[] generatedStairNodes)
        {
            generator = owner;
            environmentIndex = generatedEnvironmentIndex;
            seed = generatedSeed;
            level = generatedLevel;
            environmentRoot = generatedEnvironmentRoot;
            geometryRoot = generatedGeometryRoot;
            obstaclesRoot = generatedObstaclesRoot;
            hazardsRoot = generatedHazardsRoot;
            startPoint = generatedStartPoint;
            targetPoint = generatedTargetPoint;
            mapSize = generatedMapSize;
            cellSize = generatedCellSize;
            floorCount = Mathf.Max(1, generatedFloorCount);
            floorHeight = Mathf.Max(0.1f, generatedFloorHeight);
            startCell = generatedStartCell;
            targetCell = generatedTargetCell;
            startNode = generatedStartNode;
            targetNode = generatedTargetNode;
            obstacleCells = generatedObstacleCells ?? new Vector2Int[0];
            hazardCells = generatedHazardCells ?? new Vector2Int[0];
            collapseCells = generatedCollapseCells ?? new Vector2Int[0];
            walkableNodes = generatedWalkableNodes ?? new Vector3Int[0];
            stairNodes = generatedStairNodes ?? new Vector3Int[0];
        }

        public bool IsInsideMap(Vector2Int cell)
        {
            return cell.x >= 0 && cell.y >= 0 && cell.x < mapSize.x && cell.y < mapSize.y;
        }

        public bool IsCellBlocked(Vector2Int cell)
        {
            return RLTrainingGenerationUtility.ContainsCell(obstacleCells, cell);
        }

        public bool IsCellHazard(Vector2Int cell)
        {
            return RLTrainingGenerationUtility.ContainsCell(hazardCells, cell);
        }

        public bool IsCellCollapse(Vector2Int cell)
        {
            return RLTrainingGenerationUtility.ContainsCell(collapseCells, cell);
        }

        public bool IsCellUnsafe(Vector2Int cell)
        {
            return IsCellBlocked(cell) || IsCellHazard(cell) || IsCellCollapse(cell);
        }

        public Vector3 GetWorldPositionForCell(Vector2Int cell, float y)
        {
            var root = environmentRoot != null ? environmentRoot : transform;
            return root.TransformPoint(RLTrainingGenerationUtility.CellToLocalPosition(cell, mapSize, cellSize, y));
        }

        public bool HasPathFromStartToTarget()
        {
            if (UsesLayeredNavigation)
            {
                return HasLayeredPathFromStartToTarget();
            }

            if (!IsInsideMap(startCell) || !IsInsideMap(targetCell))
            {
                return false;
            }

            if (IsCellUnsafe(startCell) || IsCellUnsafe(targetCell))
            {
                return false;
            }

            var visited = new bool[mapSize.x, mapSize.y];
            var queue = new Queue<Vector2Int>();
            visited[startCell.x, startCell.y] = true;
            queue.Enqueue(startCell);

            while (queue.Count > 0)
            {
                var current = queue.Dequeue();
                if (current == targetCell)
                {
                    return true;
                }

                for (var i = 0; i < PathDirections.Length; i++)
                {
                    var next = current + PathDirections[i];
                    if (!IsInsideMap(next) || visited[next.x, next.y] || IsCellUnsafe(next))
                    {
                        continue;
                    }

                    visited[next.x, next.y] = true;
                    queue.Enqueue(next);
                }
            }

            return false;
        }

        private bool HasLayeredPathFromStartToTarget()
        {
            var walkable = new HashSet<Vector3Int>(walkableNodes);
            if (!walkable.Contains(startNode) || !walkable.Contains(targetNode))
            {
                return false;
            }

            var stairs = new HashSet<Vector3Int>(stairNodes);
            var visited = new HashSet<Vector3Int>();
            var queue = new Queue<Vector3Int>();
            visited.Add(startNode);
            queue.Enqueue(startNode);

            while (queue.Count > 0)
            {
                var current = queue.Dequeue();
                if (current == targetNode)
                {
                    return true;
                }

                EnqueueLayeredNeighbor(current + new Vector3Int(1, 0, 0), walkable, visited, queue);
                EnqueueLayeredNeighbor(current + new Vector3Int(-1, 0, 0), walkable, visited, queue);
                EnqueueLayeredNeighbor(current + new Vector3Int(0, 1, 0), walkable, visited, queue);
                EnqueueLayeredNeighbor(current + new Vector3Int(0, -1, 0), walkable, visited, queue);

                if (!stairs.Contains(current))
                {
                    continue;
                }

                var up = current + new Vector3Int(0, 0, 1);
                var down = current + new Vector3Int(0, 0, -1);
                if (stairs.Contains(up))
                {
                    EnqueueLayeredNeighbor(up, walkable, visited, queue);
                }

                if (stairs.Contains(down))
                {
                    EnqueueLayeredNeighbor(down, walkable, visited, queue);
                }
            }

            return false;
        }

        private static void EnqueueLayeredNeighbor(
            Vector3Int node,
            HashSet<Vector3Int> walkable,
            HashSet<Vector3Int> visited,
            Queue<Vector3Int> queue)
        {
            if (!walkable.Contains(node) || visited.Contains(node))
            {
                return;
            }

            visited.Add(node);
            queue.Enqueue(node);
        }
    }
}
