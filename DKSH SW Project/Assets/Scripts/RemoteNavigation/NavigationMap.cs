using System;
using System.Collections.Generic;
using UnityEngine;

namespace DKSH.Spiderbot.RemoteNavigation
{
    // Coordinates in this assembly are ROS map metres: x right, y up, yaw CCW.
    // Unity's X/Z axes and a sensor's local frame must never be passed here directly.
    public sealed class NavigationMap
    {
        public const int MaximumCells = 262144;
        public readonly int Width, Height;
        public readonly float Resolution, OriginYaw;
        public readonly Vector2 Origin;
        public readonly string Revision;
        private readonly int[] cells;

        public NavigationMap(int width, int height, float resolution, Vector2 origin,
            float originYaw, int[] data, string revision)
        {
            if (width < 1 || height < 1 || (long)width * height > MaximumCells ||
                !Finite(resolution) || resolution < .01f || !Finite(origin.x) ||
                !Finite(origin.y) || !Finite(originYaw) || data == null ||
                data.Length != width * height || string.IsNullOrWhiteSpace(revision))
                throw new ArgumentException("Invalid occupancy map metadata.");
            foreach (int value in data)
                if (value < -1 || value > 100) throw new ArgumentException("Invalid occupancy value.");
            Width = width; Height = height; Resolution = resolution;
            Origin = origin; OriginYaw = originYaw; Revision = revision;
            cells = (int[])data.Clone();
        }

        public int Cell(int x, int y) => Inside(x, y) ? cells[y * Width + x] : -1;
        public bool Inside(int x, int y) => x >= 0 && y >= 0 && x < Width && y < Height;
        public int[] CopyCells() => (int[])cells.Clone();
        public Vector2 LocalToMap(Vector2 local) => Origin + Rotate(local, OriginYaw);
        public Vector2 MapToLocal(Vector2 point) => Rotate(point - Origin, -OriginYaw);
        public Vector2 CellCentre(int x, int y) =>
            LocalToMap(new Vector2((x + .5f) * Resolution, (y + .5f) * Resolution));
        public Vector2Int MapToCell(Vector2 point)
        {
            Vector2 local = MapToLocal(point) / Resolution;
            return new Vector2Int(Mathf.FloorToInt(local.x), Mathf.FloorToInt(local.y));
        }

        public bool Validate(Vector2 point, float radius, out string reason)
        {
            reason = "";
            if (!Finite(point.x) || !Finite(point.y) || !Finite(radius) || radius < 0)
            { reason = "Invalid coordinates."; return false; }
            Vector2 p = MapToLocal(point);
            Vector2Int cell = MapToCell(point);
            if (!Inside(cell.x, cell.y)) { reason = "Outside the map."; return false; }
            if (Cell(cell.x, cell.y) < 0) { reason = "Unmapped area."; return false; }
            if (Cell(cell.x, cell.y) >= 50) { reason = "Obstacle."; return false; }
            if (p.x < radius || p.y < radius || p.x > Width * Resolution - radius ||
                p.y > Height * Resolution - radius)
            { reason = "Too close to the map boundary."; return false; }
            int extent = Mathf.CeilToInt(radius / Resolution) + 1;
            for (int y = cell.y - extent; y <= cell.y + extent; y++)
            for (int x = cell.x - extent; x <= cell.x + extent; x++)
            {
                int value = Cell(x, y);
                if (value >= 0 && value < 50) continue;
                float dx = Mathf.Max(x * Resolution - p.x, 0, p.x - (x + 1) * Resolution);
                float dy = Mathf.Max(y * Resolution - p.y, 0, p.y - (y + 1) * Resolution);
                if (dx * dx + dy * dy <= radius * radius)
                { reason = "Insufficient clearance from obstacles or unmapped cells."; return false; }
            }
            return true;
        }

        public bool SegmentFree(Vector2 a, Vector2 b, float radius)
        {
            int steps = Mathf.Max(1, Mathf.CeilToInt(Vector2.Distance(a, b) / (Resolution * .2f)));
            for (int i = 0; i <= steps; i++)
                if (!Validate(Vector2.Lerp(a, b, i / (float)steps), radius, out _)) return false;
            return true;
        }

        public static bool Finite(float value) => !float.IsNaN(value) && !float.IsInfinity(value);
        public static Vector2 Rotate(Vector2 p, float yaw) => new Vector2(
            Mathf.Cos(yaw) * p.x - Mathf.Sin(yaw) * p.y,
            Mathf.Sin(yaw) * p.x + Mathf.Cos(yaw) * p.y);
        public static float WrapYaw(float yaw) => Mathf.Repeat(yaw + Mathf.PI, 2 * Mathf.PI) - Mathf.PI;

        public static NavigationMap CreateDemo()
        {
            const int w = 80, h = 60;
            var data = new int[w * h];
            for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                bool wall = x < 2 || y < 2 || x >= w - 2 || y >= h - 2 ||
                    (x >= 36 && x <= 38 && (y < 24 || y > 34)) ||
                    (x >= 13 && x <= 23 && y >= 25 && y <= 29) ||
                    (x >= 51 && x <= 62 && y >= 14 && y <= 22) ||
                    (x >= 54 && x <= 64 && y >= 42 && y <= 47);
                data[y * w + x] = wall ? 100 : (x >= 67 && y >= 46 ? -1 : 0);
            }
            return new NavigationMap(w, h, .2f, new Vector2(-8, -6), 0, data, "demo-room-v1");
        }
    }

    public sealed class WaypointPlanner
    {
        private readonly struct Entry
        {
            public readonly int Index;
            public readonly float Cost;
            public Entry(int index, float cost) { Index = index; Cost = cost; }
        }
        private readonly List<Entry> heap = new List<Entry>();
        private void Push(Entry value)
        {
            int i = heap.Count; heap.Add(value);
            while (i > 0)
            {
                int parent = (i - 1) / 2;
                if (heap[parent].Cost <= value.Cost) break;
                heap[i] = heap[parent]; i = parent;
            }
            heap[i] = value;
        }
        private Entry Pop()
        {
            Entry result = heap[0], last = heap[heap.Count - 1];
            heap.RemoveAt(heap.Count - 1);
            if (heap.Count == 0) return result;
            int i = 0;
            while (i * 2 + 1 < heap.Count)
            {
                int child = i * 2 + 1;
                if (child + 1 < heap.Count && heap[child + 1].Cost < heap[child].Cost) child++;
                if (heap[child].Cost >= last.Cost) break;
                heap[i] = heap[child]; i = child;
            }
            heap[i] = last; return result;
        }

        public bool Plan(NavigationMap map, Vector2 start, Vector2 goal, float radius,
            List<Vector2> path)
        {
            path.Clear(); heap.Clear();
            if (!map.Validate(start, radius, out _) || !map.Validate(goal, radius, out _)) return false;
            var s = map.MapToCell(start); var g = map.MapToCell(goal);
            int count = map.Width * map.Height, first = s.y * map.Width + s.x, target = g.y * map.Width + g.x;
            var cost = new float[count]; var parent = new int[count];
            var closed = new bool[count]; var free = new bool[count];
            for (int i = 0; i < count; i++)
            {
                cost[i] = float.PositiveInfinity; parent[i] = -1;
                free[i] = map.Validate(map.CellCentre(i % map.Width, i / map.Width), radius, out _);
            }
            if (!free[first] || !free[target]) return false;
            cost[first] = 0; Push(new Entry(first, 0));
            while (heap.Count > 0)
            {
                int current = Pop().Index;
                if (closed[current]) continue;
                if (current == target)
                {
                    var reverse = new List<Vector2>();
                    for (int i = target; i != -1; i = parent[i])
                        reverse.Add(map.CellCentre(i % map.Width, i / map.Width));
                    reverse.Reverse(); reverse.Insert(0, start); reverse.Add(goal);
                    int anchor = 0; path.Add(start);
                    while (anchor < reverse.Count - 1)
                    {
                        int next = reverse.Count - 1;
                        while (next > anchor + 1 && !map.SegmentFree(reverse[anchor], reverse[next], radius)) next--;
                        if (!map.SegmentFree(reverse[anchor], reverse[next], radius)) { path.Clear(); return false; }
                        path.Add(reverse[next]); anchor = next;
                    }
                    return true;
                }
                closed[current] = true;
                int cx = current % map.Width, cy = current / map.Width;
                for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++)
                {
                    if (dx == 0 && dy == 0) continue;
                    int x = cx + dx, y = cy + dy;
                    if (!map.Inside(x, y)) continue;
                    int index = y * map.Width + x;
                    if (!free[index] || closed[index]) continue;
                    if (dx != 0 && dy != 0 && (!free[cy * map.Width + x] || !free[y * map.Width + cx])) continue;
                    float candidate = cost[current] + (dx != 0 && dy != 0 ? 1.414214f : 1);
                    if (candidate >= cost[index]) continue;
                    cost[index] = candidate; parent[index] = current;
                    Push(new Entry(index, candidate + Vector2.Distance(new Vector2(x, y), g)));
                }
            }
            return false;
        }
    }
}
