using System;
using System.Collections.Generic;

namespace DKSH.Spiderbot.RemoteNavigation
{
    // A draft retains the map identity on which the operator selected its coordinates.
    public sealed class WaypointEditor
    {
        private readonly List<Waypoint> points = new List<Waypoint>();
        public IReadOnlyList<Waypoint> Points => points;
        public string MapRevision { get; private set; }
        public bool AppendMode { get; set; }
        public bool Add(Waypoint point, string revision)
        {
            if (point == null || string.IsNullOrEmpty(revision)) return false;
            if (MapRevision != revision) Clear();
            if (!AppendMode) points.Clear();
            if (points.Count >= 10) return false;
            MapRevision = revision; points.Add(new Waypoint(point.Position, point.yaw)); return true;
        }
        public void Remove(int index) { if (index >= 0 && index < points.Count) points.RemoveAt(index); }
        public void Move(int index, int offset)
        {
            int target = index + offset;
            if (index < 0 || target < 0 || index >= points.Count || target >= points.Count) return;
            var point = points[index]; points.RemoveAt(index); points.Insert(target, point);
        }
        public void Clear() { points.Clear(); MapRevision = null; }
        public Waypoint[] Copy() => points.ConvertAll(p => new Waypoint(p.Position, p.yaw)).ToArray();
    }
}
