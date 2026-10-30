using System;
using System.Collections;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace DKSH.Spiderbot.RemoteNavigation.Tests
{
    // Explicitly run this category after starting ros2/run_waypoint_demo.ps1.
    // It exercises the real C# transport against the real Python server.
    [Category("GatewayIntegration")]
    public sealed class GatewayIntegrationTests
    {
        private TcpNavigationClient client;
        private IEnumerator Until(Func<bool> condition, float timeout = 5)
        {
            double end = Time.realtimeSinceStartupAsDouble + timeout;
            while (Time.realtimeSinceStartupAsDouble < end)
            {
                client.Tick(Time.unscaledDeltaTime);
                if (condition()) yield break;
                yield return null;
            }
            Assert.Fail("Gateway condition timed out: " + client.Status.state + " / " + client.Status.detail);
        }
        private IEnumerator Connect()
        {
            client = new TcpNavigationClient("127.0.0.1", 8765, "");
            yield return Until(() => client.Connected && client.Map != null && client.Status.ready);
            Assert.That(client.Map.Width, Is.EqualTo(80));
            if (client.Status.stop_latched)
            {
                client.ResetStop();
                yield return Until(() => !client.Status.stop_latched);
            }
        }
        [TearDown]
        public void Cleanup() { client?.Dispose(); client = null; }

        [UnityTest]
        public IEnumerator CompletionCancellationAndReconnectUseConfirmedServerState()
        {
            yield return Connect();
            Vector2 from = new Vector2(client.Status.x, client.Status.y);
            Assert.That(client.Submit(new[] { new Waypoint(from + new Vector2(.3f, 0), .3f) }, client.Map.Revision), Is.True);
            yield return Until(() => client.Status.state == "Succeeded");
            Assert.That(client.Status.x, Is.EqualTo(from.x + .3f).Within(.01));
            Assert.That(client.Status.yaw, Is.EqualTo(.3f).Within(.02));

            from = new Vector2(client.Status.x, client.Status.y);
            Assert.That(client.Submit(new[] { new Waypoint(from + new Vector2(1, 0), 0) }, client.Map.Revision), Is.True);
            client.Cancel(); // cancel before the first acknowledgement reaches Unity
            yield return Until(() => client.Status.state == "Canceled");
            float stopped = client.Status.x;
            double end = Time.realtimeSinceStartupAsDouble + .4;
            while (Time.realtimeSinceStartupAsDouble < end) { client.Tick(Time.unscaledDeltaTime); yield return null; }
            Assert.That(client.Status.x, Is.EqualTo(stopped).Within(.001));

            Assert.That(client.Submit(new[] { new Waypoint(new Vector2(stopped + 1, client.Status.y), 0) }, client.Map.Revision), Is.True);
            yield return Until(() => client.Status.state == "Navigating");
            client.Dispose();
            yield return new WaitForSecondsRealtime(.3f);
            client = new TcpNavigationClient("127.0.0.1", 8765, "");
            yield return Until(() => client.Connected && client.Map != null && client.Status.ready);
            Assert.That(client.Status.stop_latched, Is.True);
            Assert.That(client.Status.state, Is.EqualTo("Stopped"));
            Assert.That(client.CanSubmit, Is.False);
            stopped = client.Status.x;
            end = Time.realtimeSinceStartupAsDouble + .3;
            while (Time.realtimeSinceStartupAsDouble < end) { client.Tick(Time.unscaledDeltaTime); yield return null; }
            Assert.That(client.Status.x, Is.EqualTo(stopped).Within(.001));
        }

        [UnityTest]
        public IEnumerator FrozenUiDoesNotTurnOldTelemetryIntoFreshReadiness()
        {
            yield return Connect();
            Assert.That(client.Submit(new[] { new Waypoint(new Vector2(client.Status.x + 1, client.Status.y), 0) }, client.Map.Revision), Is.True);
            yield return Until(() => client.Status.state == "Navigating");
            yield return new WaitForSecondsRealtime(1.8f); // no Tick, therefore no fresh heartbeat
            client.Tick(.01f);
            Assert.That(client.Connected, Is.False);
            Assert.That(client.Status.state, Is.EqualTo("Disconnected"));
            Assert.That(client.CanSubmit, Is.False);
        }
    }

    // Self-contained protocol regression tests; no Python process or fixed port is required.
    [Category("GatewayTransport")]
    public sealed class GatewayTransportTests
    {
        private TcpNavigationClient client;
        private TcpListener listener;
        private CancellationTokenSource serverCancellation;
        private Task serverTask;
        private const string Session = "transport-test-session";

        private static string MapPacket(string revision)
        {
            NavigationMap map = NavigationMap.CreateDemo();
            return JsonUtility.ToJson(new GatewayPacket
            {
                type = "map", session = Session, map_revision = revision,
                width = map.Width, height = map.Height, resolution = map.Resolution,
                origin_x = map.Origin.x, origin_y = map.Origin.y, origin_yaw = map.OriginYaw,
                cells = map.CopyCells()
            });
        }

        private static GatewayPacket RobotStatus(long sequence = 1, string revision = "first-map") => new GatewayPacket
        {
            type = "status", session = Session, seq = sequence, challenge = "fresh-challenge",
            state = "Idle", ready = true, map_revision = revision, mission_id = "previous-mission",
            x = -5, y = -3, footprint_radius = .35f, path = Array.Empty<Waypoint>()
        };

        private void StartGateway(Func<StreamReader, StreamWriter, Task> conversation)
        {
            string welcome = JsonUtility.ToJson(new GatewayPacket { type = "welcome", session = Session, backend = "mock" });
            string map = MapPacket("first-map"), status = JsonUtility.ToJson(RobotStatus());
            serverCancellation = new CancellationTokenSource();
            listener = new TcpListener(IPAddress.Loopback, 0); listener.Start();
            int port = ((IPEndPoint)listener.LocalEndpoint).Port;
            serverTask = Task.Run(async () =>
            {
                try
                {
                    using TcpClient peer = await listener.AcceptTcpClientAsync();
                    peer.NoDelay = true;
                    using NetworkStream stream = peer.GetStream();
                    using var reader = new StreamReader(stream, Encoding.UTF8, false, 1024, true);
                    using var writer = new StreamWriter(stream, new UTF8Encoding(false), 1024, true) { AutoFlush = true };
                    await writer.WriteLineAsync(welcome);
                    await writer.WriteLineAsync(map);
                    await writer.WriteLineAsync(status);
                    Task conversationTask = conversation(reader, writer);
                    Task holdConnection = Task.Delay(Timeout.Infinite, serverCancellation.Token);
                    if (await Task.WhenAny(conversationTask, holdConnection) == conversationTask)
                        await conversationTask;
                    await holdConnection;
                }
                catch (Exception exception) when (exception is IOException || exception is SocketException ||
                    exception is ObjectDisposedException || exception is OperationCanceledException) { }
            });
            client = new TcpNavigationClient("127.0.0.1", port, "");
        }

        private IEnumerator Until(Func<bool> condition)
        {
            double end = Time.realtimeSinceStartupAsDouble + 3;
            while (Time.realtimeSinceStartupAsDouble < end)
            {
                Assert.That(serverTask.IsFaulted, Is.False, serverTask.Exception?.ToString());
                client.Tick(Time.unscaledDeltaTime);
                if (condition()) yield break;
                yield return null;
            }
            Assert.Fail("Transport condition timed out: " + client.Status.state + " / " + client.Status.detail);
        }

        [TearDown]
        public void Cleanup()
        {
            client?.Dispose(); serverCancellation?.Cancel(); listener?.Stop();
        }

        [UnityTest]
        public IEnumerator MapReplacementWaitsForTelemetryFromTheSameMap()
        {
            var replaceMap = new TaskCompletionSource<bool>();
            var updateStatus = new TaskCompletionSource<bool>();
            string replacement = MapPacket("second-map");
            string matching = JsonUtility.ToJson(RobotStatus(2, "second-map"));
            StartGateway(async (_, writer) =>
            {
                await replaceMap.Task;
                await writer.WriteLineAsync(replacement);
                await updateStatus.Task;
                await writer.WriteLineAsync(matching);
            });
            yield return Until(() => client.CanSubmit);
            replaceMap.SetResult(true);
            yield return Until(() => client.Map?.Revision == "second-map");
            Assert.That(client.Connected, Is.True);
            Assert.That(client.CanSubmit, Is.False, "The old pose/readiness belongs to a different map.");
            updateStatus.SetResult(true);
            yield return Until(() => client.CanSubmit);
        }

        [UnityTest]
        public IEnumerator EarlierRejectionCannotAcknowledgeANewMission()
        {
            GatewayPacket rejected = RobotStatus(2);
            rejected.state = "Rejected";
            string delayedRejection = JsonUtility.ToJson(rejected);
            StartGateway(async (reader, writer) =>
            {
                string line;
                while ((line = await reader.ReadLineAsync()) != null)
                {
                    if (!line.Contains("\"command\":\"submit\"")) continue;
                    await writer.WriteLineAsync(delayedRejection);
                    return;
                }
            });
            yield return Until(() => client.CanSubmit);
            Assert.That(client.Submit(new[] { new Waypoint(new Vector2(-4, -3), 0) }, client.Map.Revision), Is.True);
            yield return Until(() => client.Status.seq == 2);
            Assert.That(client.Status.state, Is.EqualTo("Sent"));
            Assert.That(client.CanSubmit, Is.False, "Only the matching mission ID can acknowledge this submission.");
        }

        [UnityTest]
        public IEnumerator InvalidRouteDisconnectsBeforeItReachesTheMapRenderer()
        {
            var sendInvalid = new TaskCompletionSource<bool>();
            GatewayPacket malformed = RobotStatus(2);
            malformed.path = new[] { new Waypoint(new Vector2(-4, -3), 0) };
            // The JSON number is finite as a double, but overflows the client's float coordinates.
            string invalid = JsonUtility.ToJson(malformed);
            int pathStart = invalid.IndexOf("\"path\":", StringComparison.Ordinal);
            int pathEnd = invalid.IndexOf(']', pathStart);
            invalid = invalid.Substring(0, pathStart) + "\"path\":[{\"x\":1e100,\"y\":-3,\"yaw\":0}]" + invalid.Substring(pathEnd + 1);
            StartGateway(async (_, writer) =>
            {
                await sendInvalid.Task;
                await writer.WriteLineAsync(invalid);
            });
            yield return Until(() => client.CanSubmit);
            sendInvalid.SetResult(true);
            yield return Until(() => client.Status.state == "Disconnected");
            Assert.That(client.CanSubmit, Is.False);
            Assert.That(client.Status.path, Is.Empty);
            Assert.That(client.Status.detail, Is.EqualTo("Invalid gateway message. Reconnect explicitly."));
        }
    }
}
