using System.Collections.Generic;
using UnityEngine;

namespace DKSH.Spiderbot.Training
{
    internal static class RLTrainingGenerationUtility
    {
        public static Vector3 CellToLocalPosition(Vector2Int cell, Vector2Int mapSize, float cellSize, float y)
        {
            var x = (cell.x - (mapSize.x - 1) * 0.5f) * cellSize;
            var z = (cell.y - (mapSize.y - 1) * 0.5f) * cellSize;
            return new Vector3(x, y, z);
        }

        public static void DestroyUnityObject(Object unityObject)
        {
            if (unityObject == null)
            {
                return;
            }

            if (Application.isPlaying)
            {
                Object.Destroy(unityObject);
            }
            else
            {
                Object.DestroyImmediate(unityObject);
            }
        }

        public static bool ContainsCell(IReadOnlyList<Vector2Int> cells, Vector2Int cell)
        {
            if (cells == null)
            {
                return false;
            }

            for (var i = 0; i < cells.Count; i++)
            {
                if (cells[i] == cell)
                {
                    return true;
                }
            }

            return false;
        }

        public static void AddIfMissing(List<Vector2Int> cells, Vector2Int cell)
        {
            if (!ContainsCell(cells, cell))
            {
                cells.Add(cell);
            }
        }

        public static Vector2Int[] CopySortedCells(IEnumerable<Vector2Int> cells)
        {
            var sortedCells = new List<Vector2Int>();
            if (cells != null)
            {
                sortedCells.AddRange(cells);
            }

            sortedCells.Sort(CompareCells);
            return sortedCells.ToArray();
        }

        public static int CompareCells(Vector2Int a, Vector2Int b)
        {
            var yCompare = a.y.CompareTo(b.y);
            return yCompare != 0 ? yCompare : a.x.CompareTo(b.x);
        }
    }
}
