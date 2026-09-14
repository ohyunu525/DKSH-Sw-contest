using System;
using System.Collections.Generic;
using DKSH.Spiderbot.Mapping;
using UnityEngine;

namespace DKSH.Spiderbot.Exploration
{
    /// <summary>
    /// Small Unity-only reference planner. Unknown and Occupied cells are blocked,
    /// Occupied cells receive a binary clearance radius, and diagonal corner cutting
    /// is forbidden. The returned path is greedily compressed with grid line of sight.
    /// Production navigation is intentionally delegated to ROS2/Nav2.
    /// </summary>
    public sealed class GridAStarPlanner
    {
        private static readonly Vector2Int[] NeighborOffsets =
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

        private readonly List<Vector2Int> rawPath = new List<Vector2Int>(256);
        private readonly List<int> openCells = new List<int>(512);
        private bool[] blockedCells;
        private bool[] closedCells;
        private bool[] openLookup;
        private int[] gScores;
        private int[] cameFrom;
        private OccupancyGrid2D cachedGrid;
        private int cachedGridVersion = -1;
        private float cachedClearanceMeters = -1f;
        private int cachedWidth;
        private int cachedHeight;
        private int searchGoalX;
        private int searchGoalY;

        public bool TryFindPath(
            OccupancyGrid2D grid,
            Vector2Int start,
            Vector2Int goal,
            float clearanceMeters,
            List<Vector2Int> path)
        {
            if (path == null)
            {
                throw new ArgumentNullException(nameof(path));
            }

            path.Clear();
            if (grid == null || !grid.IsInsideGrid(start.x, start.y) ||
                !grid.IsInsideGrid(goal.x, goal.y) ||
                grid.GetCell(start.x, start.y) != OccupancyCellState.Free ||
                grid.GetCell(goal.x, goal.y) != OccupancyCellState.Free)
            {
                return false;
            }

            EnsureBlockedCells(grid, Mathf.Max(0f, clearanceMeters));
            ResetSearchStorage(grid.CellCount);

            var startIndex = grid.GetFlatIndex(start.x, start.y);
            var goalIndex = grid.GetFlatIndex(goal.x, goal.y);
            if (startIndex == goalIndex)
            {
                path.Add(start);
                return true;
            }

            // The robot may already be inside newly inflated clearance. Its current
            // Free cell remains a legal start, but every other inflated cell is blocked.
            if (IsBlocked(goalIndex, startIndex))
            {
                return false;
            }

            searchGoalX = goal.x;
            searchGoalY = goal.y;
            gScores[startIndex] = 0;
            openCells.Add(startIndex);
            openLookup[startIndex] = true;

            while (openCells.Count > 0)
            {
                var currentIndex = PopBestOpenCell();

                if (currentIndex == goalIndex)
                {
                    ReconstructAndCompress(grid, startIndex, goalIndex, path);
                    return true;
                }

                closedCells[currentIndex] = true;
                grid.TryGetCoordinates(currentIndex, out var currentX, out var currentY);

                for (var neighbor = 0; neighbor < NeighborOffsets.Length; neighbor++)
                {
                    var offset = NeighborOffsets[neighbor];
                    var neighborX = currentX + offset.x;
                    var neighborY = currentY + offset.y;
                    var neighborIndex = grid.GetFlatIndex(neighborX, neighborY);
                    if (neighborIndex < 0 || closedCells[neighborIndex] ||
                        IsBlocked(neighborIndex, startIndex))
                    {
                        continue;
                    }

                    var diagonal = offset.x != 0 && offset.y != 0;
                    if (diagonal)
                    {
                        var horizontalIndex = grid.GetFlatIndex(currentX + offset.x, currentY);
                        var verticalIndex = grid.GetFlatIndex(currentX, currentY + offset.y);
                        if (horizontalIndex < 0 || verticalIndex < 0 ||
                            IsBlocked(horizontalIndex, startIndex) ||
                            IsBlocked(verticalIndex, startIndex))
                        {
                            continue;
                        }
                    }

                    var tentativeScore = gScores[currentIndex] + (diagonal ? 14 : 10);
                    if (tentativeScore >= gScores[neighborIndex])
                    {
                        continue;
                    }

                    cameFrom[neighborIndex] = currentIndex;
                    gScores[neighborIndex] = tentativeScore;
                    if (!openLookup[neighborIndex])
                    {
                        openCells.Add(neighborIndex);
                        openLookup[neighborIndex] = true;
                    }
                }
            }

            return false;
        }

        private void EnsureBlockedCells(OccupancyGrid2D grid, float clearanceMeters)
        {
            var cacheIsValid = cachedGrid == grid && cachedGridVersion == grid.Version &&
                               cachedWidth == grid.Width && cachedHeight == grid.Height &&
                               Mathf.Approximately(cachedClearanceMeters, clearanceMeters) &&
                               blockedCells != null && blockedCells.Length == grid.CellCount;
            if (cacheIsValid)
            {
                return;
            }

            EnsureArrayStorage(grid.CellCount);
            var cells = grid.Cells;
            for (var index = 0; index < grid.CellCount; index++)
            {
                blockedCells[index] = cells[index] != OccupancyCellState.Free;
            }

            if (clearanceMeters > 0f)
            {
                var maximumOffset = Mathf.CeilToInt(clearanceMeters / grid.Resolution);
                var clearanceSquared = clearanceMeters * clearanceMeters + 0.000001f;
                for (var y = 0; y < grid.Height; y++)
                {
                    for (var x = 0; x < grid.Width; x++)
                    {
                        if (grid.GetCell(x, y) != OccupancyCellState.Occupied)
                        {
                            continue;
                        }

                        for (var offsetY = -maximumOffset; offsetY <= maximumOffset; offsetY++)
                        {
                            for (var offsetX = -maximumOffset; offsetX <= maximumOffset; offsetX++)
                            {
                                var distanceX = offsetX * grid.Resolution;
                                var distanceY = offsetY * grid.Resolution;
                                if (distanceX * distanceX + distanceY * distanceY > clearanceSquared)
                                {
                                    continue;
                                }

                                var inflatedIndex = grid.GetFlatIndex(x + offsetX, y + offsetY);
                                if (inflatedIndex >= 0)
                                {
                                    blockedCells[inflatedIndex] = true;
                                }
                            }
                        }
                    }
                }
            }

            cachedGrid = grid;
            cachedGridVersion = grid.Version;
            cachedClearanceMeters = clearanceMeters;
            cachedWidth = grid.Width;
            cachedHeight = grid.Height;
        }

        private void EnsureArrayStorage(int cellCount)
        {
            if (blockedCells != null && blockedCells.Length == cellCount)
            {
                return;
            }

            blockedCells = new bool[cellCount];
            closedCells = new bool[cellCount];
            openLookup = new bool[cellCount];
            gScores = new int[cellCount];
            cameFrom = new int[cellCount];
            cachedGridVersion = -1;
        }

        private void ResetSearchStorage(int cellCount)
        {
            openCells.Clear();
            for (var index = 0; index < cellCount; index++)
            {
                closedCells[index] = false;
                openLookup[index] = false;
                gScores[index] = int.MaxValue;
                cameFrom[index] = -1;
            }
        }

        private bool IsBlocked(int index, int startIndex)
        {
            return index != startIndex && blockedCells[index];
        }

        private int PopBestOpenCell()
        {
            var bestPosition = 0;
            for (var position = 1; position < openCells.Count; position++)
            {
                if (IsHigherPriority(openCells[position], openCells[bestPosition]))
                {
                    bestPosition = position;
                }
            }

            var best = openCells[bestPosition];
            var lastPosition = openCells.Count - 1;
            openCells[bestPosition] = openCells[lastPosition];
            openCells.RemoveAt(lastPosition);
            openLookup[best] = false;
            return best;
        }

        private bool IsHigherPriority(int firstIndex, int secondIndex)
        {
            var firstHeuristic = Heuristic(firstIndex);
            var secondHeuristic = Heuristic(secondIndex);
            var firstTotal = gScores[firstIndex] + firstHeuristic;
            var secondTotal = gScores[secondIndex] + secondHeuristic;
            if (firstTotal != secondTotal)
            {
                return firstTotal < secondTotal;
            }

            if (firstHeuristic != secondHeuristic)
            {
                return firstHeuristic < secondHeuristic;
            }

            return firstIndex < secondIndex;
        }

        private int Heuristic(int cellIndex)
        {
            var x = cellIndex % cachedWidth;
            var y = cellIndex / cachedWidth;
            var deltaX = Mathf.Abs(searchGoalX - x);
            var deltaY = Mathf.Abs(searchGoalY - y);
            var diagonal = Mathf.Min(deltaX, deltaY);
            var straight = Mathf.Max(deltaX, deltaY) - diagonal;
            return diagonal * 14 + straight * 10;
        }

        private void ReconstructAndCompress(
            OccupancyGrid2D grid,
            int startIndex,
            int goalIndex,
            List<Vector2Int> path)
        {
            rawPath.Clear();
            var current = goalIndex;
            while (current >= 0)
            {
                grid.TryGetCoordinates(current, out var x, out var y);
                rawPath.Add(new Vector2Int(x, y));
                if (current == startIndex)
                {
                    break;
                }

                current = cameFrom[current];
            }

            rawPath.Reverse();
            CompressPath(grid, startIndex, path);
        }

        private void CompressPath(OccupancyGrid2D grid, int startIndex, List<Vector2Int> path)
        {
            if (rawPath.Count == 0)
            {
                return;
            }

            path.Add(rawPath[0]);
            var anchor = 0;
            while (anchor < rawPath.Count - 1)
            {
                var furthestVisible = anchor + 1;
                for (var candidate = rawPath.Count - 1; candidate > anchor + 1; candidate--)
                {
                    if (!HasLineOfSight(grid, rawPath[anchor], rawPath[candidate], startIndex))
                    {
                        continue;
                    }

                    furthestVisible = candidate;
                    break;
                }

                path.Add(rawPath[furthestVisible]);
                anchor = furthestVisible;
            }
        }

        private bool HasLineOfSight(
            OccupancyGrid2D grid,
            Vector2Int start,
            Vector2Int goal,
            int startIndex)
        {
            var x = start.x;
            var y = start.y;
            var deltaX = goal.x - start.x;
            var deltaY = goal.y - start.y;
            var stepX = Math.Sign(deltaX);
            var stepY = Math.Sign(deltaY);
            var absoluteX = Mathf.Abs(deltaX);
            var absoluteY = Mathf.Abs(deltaY);
            var deltaTimeX = absoluteX > 0 ? 1f / absoluteX : float.PositiveInfinity;
            var deltaTimeY = absoluteY > 0 ? 1f / absoluteY : float.PositiveInfinity;
            var maximumTimeX = absoluteX > 0 ? 0.5f / absoluteX : float.PositiveInfinity;
            var maximumTimeY = absoluteY > 0 ? 0.5f / absoluteY : float.PositiveInfinity;

            while (x != goal.x || y != goal.y)
            {
                if (maximumTimeX < maximumTimeY)
                {
                    x += stepX;
                    maximumTimeX += deltaTimeX;
                }
                else if (maximumTimeY < maximumTimeX)
                {
                    y += stepY;
                    maximumTimeY += deltaTimeY;
                }
                else
                {
                    var horizontalIndex = grid.GetFlatIndex(x + stepX, y);
                    var verticalIndex = grid.GetFlatIndex(x, y + stepY);
                    if (horizontalIndex < 0 || verticalIndex < 0 ||
                        IsBlocked(horizontalIndex, startIndex) ||
                        IsBlocked(verticalIndex, startIndex))
                    {
                        return false;
                    }

                    x += stepX;
                    y += stepY;
                    maximumTimeX += deltaTimeX;
                    maximumTimeY += deltaTimeY;
                }

                var cellIndex = grid.GetFlatIndex(x, y);
                if (cellIndex < 0 || IsBlocked(cellIndex, startIndex))
                {
                    return false;
                }
            }

            return true;
        }
    }
}
