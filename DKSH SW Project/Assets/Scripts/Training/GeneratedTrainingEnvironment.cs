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
        private Vector2Int startCell;

        [SerializeField]
        private Vector2Int targetCell;

        [SerializeField]
        private Vector2Int[] obstacleCells = new Vector2Int[0];

        [SerializeField]
        private Vector2Int[] hazardCells = new Vector2Int[0];

        [SerializeField]
        private Vector2Int[] collapseCells = new Vector2Int[0];

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
        public Vector2Int StartCell { get { return startCell; } }
        public Vector2Int TargetCell { get { return targetCell; } }
        public IReadOnlyList<Vector2Int> ObstacleCells { get { return obstacleCells; } }
        public IReadOnlyList<Vector2Int> HazardCells { get { return hazardCells; } }
        public IReadOnlyList<Vector2Int> CollapseCells { get { return collapseCells; } }

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
            Vector2Int generatedStartCell,
            Vector2Int generatedTargetCell,
            Vector2Int[] generatedObstacleCells,
            Vector2Int[] generatedHazardCells,
            Vector2Int[] generatedCollapseCells)
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
            startCell = generatedStartCell;
            targetCell = generatedTargetCell;
            obstacleCells = generatedObstacleCells ?? new Vector2Int[0];
            hazardCells = generatedHazardCells ?? new Vector2Int[0];
            collapseCells = generatedCollapseCells ?? new Vector2Int[0];
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
    }
}
