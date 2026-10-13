using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace DKSH.Spiderbot.RemoteNavigation
{
    // NDJSON v1 is deliberately a narrow gateway, rather than a general ROS topic tunnel.
    // Only simulation gateways are accepted in this sensor-free client.
    public sealed class TcpNavigationClient : INavigationClient
    {
        private readonly struct Received
        {
            public readonly string Json;
            public readonly double At;
            public Received(string json) { Json = json; At = Now; }
        }
        private static double Now => System.Diagnostics.Stopwatch.GetTimestamp() /
            (double)System.Diagnostics.Stopwatch.Frequency;
        private const int MaximumLineBytes = 2 * 1024 * 1024;
        private readonly ConcurrentQueue<Received> incoming = new ConcurrentQueue<Received>();
        private readonly ConcurrentQueue<byte[]> outgoing = new ConcurrentQueue<byte[]>();
        private readonly CancellationTokenSource cancellation = new CancellationTokenSource();
        private TcpClient socket;
        private readonly string token;
        private string session = "", challenge = "", pendingMission = "";
        private long commandSequence, statusSequence = -1;
        private double lastStatus, nextHeartbeat, pendingSince;
        private readonly double connectedAt;
        private volatile bool disposed;
        private bool timedOut;
        public NavigationMap Map { get; private set; }
        public GatewayPacket Status { get; private set; } = new GatewayPacket
        { type = "status", state = "Connecting", detail = "Connecting to the virtual robot gateway.", path = Array.Empty<Waypoint>() };
        public bool Connected => !disposed && !timedOut && session.Length > 0 &&
            Now - lastStatus < 1.5;
        public bool CanSubmit => Connected && Status.ready && Map != null && !Status.stop_latched &&
            string.IsNullOrEmpty(pendingMission) && (Status.state == "Idle" ||
            Status.state == "Succeeded" || Status.state == "Canceled" || Status.state == "Rejected");

        public TcpNavigationClient(string host, int port, string sharedToken)
        {
            token = sharedToken ?? "";
            connectedAt = Now;
            string hello = JsonUtility.ToJson(new GatewayPacket { type = "hello", token = token }) + "\n";
            _ = Run(host, port, hello);
        }

        private async Task Run(string host, int port, string hello)
        {
            try
            {
                socket = new TcpClient { NoDelay = true };
                Task connect = socket.ConnectAsync(host, port);
                if (await Task.WhenAny(connect, Task.Delay(5000, cancellation.Token)).ConfigureAwait(false) != connect)
                    throw new IOException("Connection timed out.");
                await connect.ConfigureAwait(false);
                using (NetworkStream stream = socket.GetStream())
                {
                    byte[] bytes = Encoding.UTF8.GetBytes(hello);
                    await stream.WriteAsync(bytes, 0, bytes.Length, cancellation.Token).ConfigureAwait(false);
                    Task reader = Read(stream), writer = Write(stream);
                    await Task.WhenAny(reader, writer).ConfigureAwait(false);
                    socket.Close(); cancellation.Cancel();
                    await Task.WhenAll(reader, writer).ConfigureAwait(false);
                }
            }
            catch (Exception exception) when (exception is IOException || exception is SocketException ||
                exception is OperationCanceledException || exception is ObjectDisposedException || exception is ArgumentException)
            {
                if (!disposed) incoming.Enqueue(new Received("{\"v\":1,\"type\":\"error\",\"detail\":\"Gateway connection closed. Reconnect explicitly.\"}"));
            }
            finally { socket?.Close(); }
        }

        private async Task Read(NetworkStream stream)
        {
            var buffer = new byte[8192];
            using (var line = new MemoryStream())
            {
                while (!cancellation.IsCancellationRequested)
                {
                    int count = await stream.ReadAsync(buffer, 0, buffer.Length, cancellation.Token).ConfigureAwait(false);
                    if (count == 0) throw new IOException("Gateway disconnected.");
                    for (int i = 0; i < count; i++)
                    {
                        if (buffer[i] == 10)
                        {
                            if (incoming.Count >= 16) throw new IOException("Incoming queue overflow.");
                            incoming.Enqueue(new Received(Encoding.UTF8.GetString(line.GetBuffer(), 0, (int)line.Length)));
                            line.SetLength(0);
                        }
                        else
                        {
                            if (line.Length >= MaximumLineBytes) throw new IOException("Oversized gateway packet.");
                            line.WriteByte(buffer[i]);
                        }
                    }
                }
            }
        }

        private async Task Write(NetworkStream stream)
        {
            while (!cancellation.IsCancellationRequested)
            {
                if (outgoing.TryDequeue(out var bytes))
                    await stream.WriteAsync(bytes, 0, bytes.Length, cancellation.Token).ConfigureAwait(false);
                else await Task.Delay(20, cancellation.Token).ConfigureAwait(false);
            }
        }

        public void Tick(float deltaTime)
        {
            if (disposed || timedOut) return;
            while (incoming.TryDequeue(out Received received))
            {
                if (Now - received.At >= 1.5)
                { Fail("Queued telemetry is stale. Reconnect explicitly to confirm the stop state."); return; }
                try
                {
                    GatewayPacket packet = JsonUtility.FromJson<GatewayPacket>(received.Json);
                    if (packet == null || packet.v != 1) throw new ArgumentException("Protocol mismatch.");
                    if (packet.type == "error") { Fail(packet.detail); return; }
                    if (packet.type == "welcome")
                    {
                        if (!string.IsNullOrEmpty(session) || string.IsNullOrEmpty(packet.session) ||
                            (packet.backend != "mock" && packet.backend != "nav2_simulation"))
                            throw new ArgumentException("This client requires a simulation gateway.");
                        session = packet.session; lastStatus = received.At;
                    }
                    else if (packet.type == "map")
                    {
                        if (packet.session != session || session.Length == 0) continue;
                        Map = new NavigationMap(packet.width, packet.height, packet.resolution,
                            new Vector2(packet.origin_x, packet.origin_y), packet.origin_yaw, packet.cells, packet.map_revision);
                    }
                    else if (packet.type == "status")
                    {
                        if (packet.session != session || session.Length == 0 || packet.seq <= statusSequence) continue;
                        if (!NavigationMap.Finite(packet.x) || !NavigationMap.Finite(packet.y) ||
                            !NavigationMap.Finite(packet.yaw) || !NavigationMap.Finite(packet.footprint_radius) ||
                            packet.footprint_radius <= 0 || string.IsNullOrEmpty(packet.challenge))
                            throw new ArgumentException("Invalid robot state.");
                        statusSequence = packet.seq; challenge = packet.challenge;
                        lastStatus = received.At;
                        Status = packet;
                        if (packet.mission_id == pendingMission && packet.state != "Idle") pendingMission = "";
                        if (packet.state == "Rejected") pendingMission = "";
                    }
                }
                catch (Exception e) when (e is ArgumentException || e is OverflowException)
                { Fail("Invalid gateway message. Reconnect explicitly."); return; }
            }
            double now = Now;
            if (session.Length == 0 && now - connectedAt > 5)
            { Fail("Gateway handshake timed out. Reconnect explicitly."); return; }
            if (session.Length > 0 && now - lastStatus >= 1.5)
            { Fail("Telemetry is stale. Stop confirmation unavailable; reconnect explicitly."); return; }
            if (pendingMission.Length > 0 && now - pendingSince > 2)
            { Send("stop"); Fail("Goal acknowledgement timed out. Reconnect to check the stop state."); return; }
            if (pendingMission.Length > 0)
            { Status.state = "Sent"; Status.detail = "Awaiting mission acknowledgement."; }
            if (Connected && now >= nextHeartbeat && challenge.Length > 0)
            { Send("heartbeat"); nextHeartbeat = now + .2; }
        }

        public bool Submit(IReadOnlyList<Waypoint> waypoints, string mapRevision)
        {
            if (!CanSubmit || mapRevision != Map.Revision || waypoints == null || waypoints.Count < 1 || waypoints.Count > 10) return false;
            var copied = new Waypoint[waypoints.Count];
            for (int i = 0; i < copied.Length; i++)
            {
                var point = waypoints[i];
                if (point == null || !NavigationMap.Finite(point.yaw) || !Map.Validate(point.Position, Status.footprint_radius, out _)) return false;
                copied[i] = new Waypoint(point.Position, point.yaw);
            }
            pendingMission = Guid.NewGuid().ToString("N"); pendingSince = Now;
            Send("submit", copied, pendingMission, mapRevision);
            return true;
        }

        private void Send(string command, Waypoint[] points = null, string mission = null, string revision = null)
        {
            if (!Connected || challenge.Length == 0) return;
            if (outgoing.Count >= 16) { Fail("Outgoing queue overflow."); return; }
            var packet = new GatewayPacket { type = "command", command = command, session = session,
                challenge = challenge, seq = ++commandSequence, request_id = Guid.NewGuid().ToString("N"),
                mission_id = mission ?? Status.mission_id, map_revision = revision ?? Map?.Revision, waypoints = points };
            outgoing.Enqueue(Encoding.UTF8.GetBytes(JsonUtility.ToJson(packet) + "\n"));
        }

        public void Cancel() { Send("cancel", mission: pendingMission.Length > 0 ? pendingMission : Status.mission_id); }
        public void Stop() { Send("stop"); }
        public void ResetStop() { Send("reset_stop"); }
        private void Fail(string detail)
        {
            timedOut = true; pendingMission = "";
            Status = new GatewayPacket { state = "Disconnected", detail = detail, ready = false,
                x = Status.x, y = Status.y, yaw = Status.yaw, footprint_radius = Status.footprint_radius,
                path = Array.Empty<Waypoint>() };
            cancellation.Cancel(); socket?.Close();
        }
        public void Dispose()
        {
            if (disposed) return;
            disposed = true; cancellation.Cancel(); socket?.Close();
        }
    }
}
