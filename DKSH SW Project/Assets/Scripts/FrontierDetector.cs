using System.Collections.Generic;
using DKSH.Spiderbot.Mapping;
using UnityEngine;

namespace DKSH.Spiderbot.Exploration
{
    public enum FrontierConnectivity
    {
        Four = 4,
        Eight = 8
    }

    /// <summary>
    /// Detects Free cells that border at least one in-bounds Unknown cell.
    /// The returned list is reused and remains valid until the next detection.
    /// </summary>
    public sealed class FrontierDetector
    {
        private static readonly Vector2Int[] FourNeighborOffsets =
        {
            new Vector2Int(1, 0),
            new Vector2Int(-1, 0),
            new Vector2Int(0, 1),
            new Vector2Int(0, -1)
        };

        private static readonly Vector2Int[] EightNeighborOffsets =
        {
            new Vector2Int(1, 0),
            new Vector2Int(-1, 0),
            new Vector2Int(0, 1),
            new Vector2Int(0, -1),
            new Vector2Int(1, 1),
            new Vector2Int(1, -1),
            new Vector2Int(-1, 1),
            new Vector2Int(-1, -1)
        };

        private readonly List<Vector2Int> frontierCells = new List<Vector2Int>(512);
        private OccupancyGrid2D occupancyGrid;

        public FrontierDetector()
        {
        }

        public FrontierDetector(OccupancyGrid2D grid)
        {
            occupancyGrid = grid;
        }

        public FrontierConnectivity Connectivity { get; set; } = FrontierConnectivity.Eight;
        public OccupancyGrid2D Grid { get { return occupancyGrid; } }
        public IReadOnlyList<Vector2Int> FrontierCells { get { return frontierCells; } }

        public void SetGrid(OccupancyGrid2D grid)
        {
            occupancyGrid = grid;
        }

        public List<Vector2Int> DetectFrontierCells()
        {
            frontierCells.Clear();
            if (occupancyGrid == null)
            {
                return frontierCells;
            }

            var width = occupancyGrid.Width;
            var height = occupancyGrid.Height;
            var cells = occupancyGrid.Cells;
            var offsets = GetNeighborOffsets(Connectivity);
            for (var y = 0; y < height; y++)
            {
                for (var x = 0; x < width; x++)
                {
                    if (cells[y * width + x] != OccupancyCellState.Free)
                    {
                        continue;
                    }

                    for (var neighbor = 0; neighbor < offsets.Length; neighbor++)
                    {
                        var neighborX = x + offsets[neighbor].x;
                        var neighborY = y + offsets[neighbor].y;
                        if (neighborX < 0 || neighborX >= width ||
                            neighborY < 0 || neighborY >= height ||
                            cells[neighborY * width + neighborX] != OccupancyCellState.Unknown)
                        {
                            continue;
                        }

                        frontierCells.Add(new Vector2Int(x, y));
                        break;
                    }
                }
            }

            return frontierCells;
        }

        public static bool IsFrontierCell(
            OccupancyGrid2D grid,
            int x,
            int y,
            FrontierConnectivity connectivity = FrontierConnectivity.Eight)
        {
            if (grid == null || !grid.IsInsideGrid(x, y) ||
                grid.GetCell(x, y) != OccupancyCellState.Free)
            {
                return false;
            }

            var offsets = GetNeighborOffsets(connectivity);
            for (var i = 0; i < offsets.Length; i++)
            {
                var neighborX = x + offsets[i].x;
                var neighborY = y + offsets[i].y;
                if (grid.IsInsideGrid(neighborX, neighborY) &&
                    grid.GetCell(neighborX, neighborY) == OccupancyCellState.Unknown)
                {
                    return true;
                }
            }

            return false;
        }

        internal static Vector2Int[] GetNeighborOffsets(FrontierConnectivity connectivity)
        {
            return connectivity == FrontierConnectivity.Four
                ? FourNeighborOffsets
                : EightNeighborOffsets;
        }
    }
}
