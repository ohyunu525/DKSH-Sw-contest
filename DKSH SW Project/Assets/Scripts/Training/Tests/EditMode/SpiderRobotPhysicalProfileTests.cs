using NUnit.Framework;
using UnityEngine;

namespace DKSH.Spiderbot.Training.Tests
{
    public sealed class SpiderRobotPhysicalProfileTests
    {
        [Test]
        public void RuntimeDefault_MatchesProvidedLegMassBreakdown()
        {
            var profile = SpiderRobotPhysicalProfile.CreateRuntimeDefault();
            try
            {
                Assert.That(profile.LegMassKg, Is.EqualTo(0.30227f).Within(0.000001f));
                Assert.That(profile.TotalKnownLegMassKg, Is.EqualTo(2.41816f).Within(0.00001f));
            }
            finally
            {
                Object.DestroyImmediate(profile);
            }
        }

        [Test]
        public void RuntimeDefault_MatchesProvidedGeometryAndServoLimits()
        {
            var profile = SpiderRobotPhysicalProfile.CreateRuntimeDefault();
            try
            {
                Assert.That(profile.HipLengthM, Is.EqualTo(0.08617f).Within(0.000001f));
                Assert.That(profile.FemurLengthM, Is.EqualTo(0.1f).Within(0.000001f));
                Assert.That(profile.TibiaLengthM, Is.EqualTo(0.12f).Within(0.000001f));
                Assert.That(profile.FemurTibiaReachM, Is.EqualTo(0.22f).Within(0.000001f));
                Assert.That(profile.TotalLinkReachM, Is.EqualTo(0.30617f).Within(0.000001f));
                Assert.That(profile.StallTorqueNewtonMeters, Is.EqualTo(0.980665f).Within(0.000001f));
                Assert.That(profile.CommandPeriodSeconds, Is.EqualTo(0.02f).Within(0.000001f));
                Assert.That(profile.JointRangeDegrees, Is.EqualTo(180f));
                Assert.That(profile.SupplyVoltageV, Is.EqualTo(6f));
            }
            finally
            {
                Object.DestroyImmediate(profile);
            }
        }
    }
}
