using System;
using System.Collections.Generic;
using UnityEngine;

namespace DKSH.Spiderbot.RemoteNavigation
{
    [Serializable]
    public sealed class Waypoint
    {
        public float x, y, yaw;
        public Vector2 Position => new Vector2(x, y);
        public Waypoint() { }
        public Waypoint(Vector2 position, float heading) { x = position.x; y = position.y; yaw = NavigationMap.WrapYaw(heading); }
    }

    [Serializable]
    public sealed class GatewayPacket
    {
        public int v = 1;
        public string type, command, token, session, challenge, request_id, mission_id, map_revision;
        public long seq;
        public Waypoint[] waypoints;
        public int width, height;
        public float resolution, origin_x, origin_y, origin_yaw;
        public int[] cells;
        public string state, detail, backend;
        public bool ready, stop_latched;
        public int waypoint_index, waypoint_count;
        public float x, y, yaw, distance_remaining;
        public Waypoint[] path;
        public float footprint_radius;
    }

    public interface INavigationClient : IDisposable
    {
        NavigationMap Map { get; }
        GatewayPacket Status { get; }
        bool Connected { get; }
        bool CanSubmit { get; }
        void Tick(float deltaTime);
        bool Submit(IReadOnlyList<Waypoint> waypoints, string mapRevision);
        void Cancel();
        void Stop();
        void ResetStop();
    }

    public sealed class SimulatedNavigationClient : INavigationClient
    {
        private readonly WaypointPlanner planner = new WaypointPlanner();
        private readonly List<Vector2> path = new List<Vector2>();
        private Waypoint[] goals = Array.Empty<Waypoint>();
        private int pathIndex;
        public NavigationMap Map { get; private set; } = NavigationMap.CreateDemo();
        public GatewayPacket Status { get; private set; } = new GatewayPacket
        {
            type = "status", state = "Idle", detail = "Select a waypoint on the map.",
            backend = "unity_simulation", ready = true, x = -5, y = -3, footprint_radius = .35f,
            map_revision = "demo-room-v1", path = Array.Empty<Waypoint>()
        };
        public bool Connected { get; private set; } = true;
        public bool CanSubmit => Connected && Status.ready && !Status.stop_latched && !Busy;
        private bool Busy => Status.state == "Accepted" || Status.state == "Navigating";

        public bool Submit(IReadOnlyList<Waypoint> waypoints, string mapRevision)
        {
            if (!CanSubmit || mapRevision != Map.Revision || waypoints == null || waypoints.Count < 1 || waypoints.Count > 10)
                return Reject("Cannot execute: check connection, map and mission state.");
            var preview = new List<Vector2>(); Vector2 from = new Vector2(Status.x, Status.y);
            foreach (var goal in waypoints)
            {
                if (goal == null || !NavigationMap.Finite(goal.yaw) ||
                    !Map.Validate(goal.Position, Status.footprint_radius, out _) ||
                    !planner.Plan(Map, from, goal.Position, Status.footprint_radius, preview))
                    return Reject("A waypoint is invalid or unreachable.");
                from = goal.Position;
            }
            goals = new Waypoint[waypoints.Count];
            for (int i = 0; i < goals.Length; i++) goals[i] = new Waypoint(waypoints[i].Position, waypoints[i].yaw);
            Status.mission_id = Guid.NewGuid().ToString("N");
            Status.waypoint_index = 0; Status.waypoint_count = goals.Length;
            Status.state = "Accepted"; Status.detail = "Mission accepted by the virtual robot.";
            PrepareLeg(); return true;
        }

        private bool Reject(string detail) { Status.detail = detail; return false; }
        private void PrepareLeg()
        {
            if (!planner.Plan(Map, new Vector2(Status.x, Status.y), goals[Status.waypoint_index].Position,
                Status.footprint_radius, path))
            { Status.state = "Failed"; Status.detail = "Route is blocked."; ClearMotion(); return; }
            pathIndex = 1;
            Status.path = path.ConvertAll(p => new Waypoint(p, 0)).ToArray();
        }

        public void Tick(float deltaTime)
        {
            if (!Connected || !Busy) return;
            if (Status.state == "Accepted") { Status.state = "Navigating"; return; }
            deltaTime = Mathf.Clamp(deltaTime, 0, .1f);
            float budget = deltaTime * .65f;
            Vector2 position = new Vector2(Status.x, Status.y);
            while (pathIndex < path.Count && budget > 0)
            {
                Vector2 next = path[pathIndex]; float distance = Vector2.Distance(position, next);
                Vector2 step = Vector2.MoveTowards(position, next, budget);
                if (distance > .0001f)
                {
                    Vector2 direction = next - position;
                    Status.yaw = Mathf.Atan2(direction.y, direction.x);
                    if (!Map.SegmentFree(position, step, Status.footprint_radius))
                    { Stop(); Status.detail = "Route became unsafe."; return; }
                }
                if (distance <= budget) { position = next; budget -= distance; pathIndex++; }
                else { position = step; budget = 0; }
            }
            Status.x = position.x; Status.y = position.y;
            Status.distance_remaining = 0;
            Vector2 previous = position;
            for (int i = pathIndex; i < path.Count; i++)
            { Status.distance_remaining += Vector2.Distance(previous, path[i]); previous = path[i]; }
            if (pathIndex < path.Count) return;
            float targetYaw = goals[Status.waypoint_index].yaw;
            float error = NavigationMap.WrapYaw(targetYaw - Status.yaw);
            Status.yaw = NavigationMap.WrapYaw(Status.yaw + Mathf.Clamp(error, -deltaTime, deltaTime));
            if (Mathf.Abs(error) > .025f) { Status.detail = "Aligning to the waypoint heading."; return; }
            Status.yaw = targetYaw;
            Status.waypoint_index++;
            if (Status.waypoint_index < goals.Length) PrepareLeg();
            else { Status.state = "Succeeded"; Status.detail = "All waypoints reached."; ClearMotion(); }
        }

        private void ClearMotion() { path.Clear(); Status.path = Array.Empty<Waypoint>(); Status.distance_remaining = 0; }
        public void Cancel()
        {
            if (!Connected || !Busy) return;
            Status.state = "Canceled"; Status.detail = "Mission canceled. Virtual robot stopped."; ClearMotion();
        }
        public void Stop()
        {
            Status.stop_latched = true; Status.state = "Stopped";
            Status.detail = "Stop is latched. Reset it before executing another mission."; ClearMotion();
        }
        public void ResetStop()
        {
            if (!Connected || !Status.stop_latched || Busy) return;
            Status.stop_latched = false; Status.state = "Idle"; Status.detail = "Stop reset. Select and execute a new mission.";
        }
        public void SetLinkAvailable(bool available)
        {
            if (Connected == available) return;
            Stop(); Connected = available; Status.ready = available;
            Status.detail = available ? "Link restored. Reset stop and submit a new mission." : "Link lost. Virtual robot stopped.";
        }
        public void Dispose() { Stop(); Connected = false; }
    }
}
