using System;
using System.Collections.Generic;
using DKSH.Spiderbot.Mapping;
using UnityEngine;

namespace DKSH.Spiderbot.Exploration
{
    [Serializable]
    public sealed class FrontierCluster
    {
        private readonly List<Vector2Int> cells = new List<Vector2Int>(64);
        private float gridCoordinateSumX;
        private float gridCoordinateSumY;

        public IReadOnlyList<Vector2Int> Cells { get { return cells; } }
        public int CellCount { get { return cells.Count; } }
        public Vector2 CentroidGrid { get; private set; }
        public Vector3 CentroidWorld { get; private set; }
        public Vector2Int GoalGrid { get; private set; }
        public Vector3 GoalWorld { get; private set; }
        public float DistanceFromRobot { get; internal set; }
        public float Score { get; internal set; }

        internal void Reset()
        {
            cells.Clear();
            gridCoordinateSumX = 0f;
            gridCoordinateSumY = 0f;
            CentroidGrid = Vector2.zero;
            CentroidWorld = Vector3.zero;
            GoalGrid = new Vector2Int(-1, -1);
            GoalWorld = Vector3.zero;
            DistanceFromRobot = float.PositiveInfinity;
            Score = float.NegativeInfinity;
        }

        internal void AddCell(Vector2Int cell)
        {
            cells.Add(cell);
            gridCoordinateSumX += cell.x;
            gridCoordinateSumY += cell.y;
        }

        internal bool FinalizeCandidate(OccupancyGrid2D grid)
        {
            if (grid == null || cells.Count == 0)
            {
                return false;
            }

            CentroidGrid = new Vector2(
                gridCoordinateSumX / cells.Count,
                gridCoordinateSumY / cells.Count);

            var worldSum = Vector3.zero;
            var bestDistanceSquared = float.PositiveInfinity;
            var representative = new Vector2Int(-1, -1);

            for (var i = 0; i < cells.Count; i++)
            {
                var cell = cells[i];
                worldSum += grid.GridToWorld(cell.x, cell.y);

                var deltaX = cell.x - CentroidGrid.x;
                var deltaY = cell.y - CentroidGrid.y;
                var distanceSquared = deltaX * deltaX + deltaY * deltaY;
                if (distanceSquared < bestDistanceSquared ||
                    (Mathf.Approximately(distanceSquared, bestDistanceSquared) &&
                     IsLexicographicallyEarlier(cell, representative)))
                {
                    bestDistanceSquared = distanceSquared;
                    representative = cell;
                }
            }

            CentroidWorld = worldSum / cells.Count;
            if (!grid.IsInsideGrid(representative.x, representative.y) ||
                grid.GetCell(representative.x, representative.y) != OccupancyCellState.Free)
            {
                return false;
            }

            GoalGrid = representative;
            GoalWorld = grid.GridToWorld(representative.x, representative.y);
            return true;
        }

        public bool Contains(Vector2Int cell)
        {
            for (var i = 0; i < cells.Count; i++)
            {
                if (cells[i] == cell)
                {
                    return true;
                }
            }

            return false;
        }

        private static bool IsLexicographicallyEarlier(Vector2Int candidate, Vector2Int current)
        {
            return current.x < 0 || candidate.y < current.y ||
                   (candidate.y == current.y && candidate.x < current.x);
        }
    }
}
