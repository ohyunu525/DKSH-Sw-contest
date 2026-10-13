using System;
using System.Collections;
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
}
