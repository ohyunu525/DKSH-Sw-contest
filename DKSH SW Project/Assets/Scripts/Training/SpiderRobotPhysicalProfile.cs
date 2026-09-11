using UnityEngine;

namespace DKSH.Spiderbot.Training
{
    [CreateAssetMenu(
        fileName = "SpiderRobotPhysicalProfile",
        menuName = "DKSH/Spiderbot/Physical Profile")]
    public sealed class SpiderRobotPhysicalProfile : ScriptableObject
    {
        private const float KilogramCentimeterToNewtonMeter = 0.0980665f;

        [Header("Robot Layout")]
        [SerializeField, Min(1)] private int legCount = 8;
        [SerializeField, Min(1)] private int servosPerLeg = 3;
        [SerializeField, Min(0)] private int bearingsPerLeg = 3;

        [Header("Known Masses (kg)")]
        [SerializeField, Min(0f)] private float servoMassKg = 0.055f;
        [SerializeField, Min(0f)] private float bearingMassKg = 0.0024f;
        [SerializeField, Min(0f)] private float printedPartsMassPerLegKg = 0.13007f;

        [Header("Link Geometry (m, axis to axis)")]
        [SerializeField, Min(0f)] private float hipLengthM = 0.08617f;
        [SerializeField, Min(0f)] private float femurLengthM = 0.100f;
        [SerializeField, Min(0f)] private float tibiaLengthM = 0.120f;

        [Header("MG996R at 6 V")]
        [SerializeField, Min(0f)] private float supplyVoltageV = 6f;
        [SerializeField, Min(0f)] private float stallTorqueKgCm = 10f;
        [SerializeField, Range(0f, 180f)] private float jointRangeDegrees = 180f;
        [SerializeField, Min(1f)] private float commandFrequencyHz = 50f;
        [Tooltip("Nominal no-load time at 6 V. Treat this as a tunable estimate, not a loaded-joint guarantee.")]
        [SerializeField, Min(0.01f)] private float secondsPer60Degrees = 0.15f;

        [Header("Navigation Proxy")]
        [Tooltip("Conservative stair riser used before an articulated leg model is available.")]
        [SerializeField, Min(0.01f)] private float recommendedStepHeightM = 0.16f;
        [SerializeField, Min(0.01f)] private float proxyMaxSpeedMps = 1.0f;
        [SerializeField, Min(0.01f)] private float proxyAccelerationMps2 = 3.5f;
        [SerializeField, Min(1f)] private float proxyYawSpeedDegreesPerSecond = 150f;

        public int LegCount { get { return legCount; } }
        public int ServosPerLeg { get { return servosPerLeg; } }
        public int BearingsPerLeg { get { return bearingsPerLeg; } }
        public float ServoMassKg { get { return servoMassKg; } }
        public float BearingMassKg { get { return bearingMassKg; } }
        public float PrintedPartsMassPerLegKg { get { return printedPartsMassPerLegKg; } }
        public float HipLengthM { get { return hipLengthM; } }
        public float FemurLengthM { get { return femurLengthM; } }
        public float TibiaLengthM { get { return tibiaLengthM; } }
        public float SupplyVoltageV { get { return supplyVoltageV; } }
        public float StallTorqueKgCm { get { return stallTorqueKgCm; } }
        public float JointRangeDegrees { get { return jointRangeDegrees; } }
        public float CommandFrequencyHz { get { return commandFrequencyHz; } }
        public float SecondsPer60Degrees { get { return secondsPer60Degrees; } }
        public float RecommendedStepHeightM { get { return recommendedStepHeightM; } }
        public float ProxyMaxSpeedMps { get { return proxyMaxSpeedMps; } }
        public float ProxyAccelerationMps2 { get { return proxyAccelerationMps2; } }
        public float ProxyYawSpeedDegreesPerSecond { get { return proxyYawSpeedDegreesPerSecond; } }

        public float LegMassKg
        {
            get
            {
                return servosPerLeg * servoMassKg
                    + bearingsPerLeg * bearingMassKg
                    + printedPartsMassPerLegKg;
            }
        }

        public float TotalKnownLegMassKg { get { return legCount * LegMassKg; } }
        public float StallTorqueNewtonMeters { get { return stallTorqueKgCm * KilogramCentimeterToNewtonMeter; } }
        public float CommandPeriodSeconds { get { return 1f / Mathf.Max(1f, commandFrequencyHz); } }
        public float FemurTibiaReachM { get { return femurLengthM + tibiaLengthM; } }
        public float TotalLinkReachM { get { return hipLengthM + femurLengthM + tibiaLengthM; } }
        public float NoLoadAngularSpeedRadiansPerSecond
        {
            get { return Mathf.Deg2Rad * 60f / Mathf.Max(0.01f, secondsPer60Degrees); }
        }

        public static SpiderRobotPhysicalProfile CreateRuntimeDefault()
        {
            var profile = CreateInstance<SpiderRobotPhysicalProfile>();
            profile.name = "Spider Robot Physical Profile (Runtime Default)";
            profile.hideFlags = HideFlags.DontSave;
            return profile;
        }

        private void OnValidate()
        {
            legCount = Mathf.Max(1, legCount);
            servosPerLeg = Mathf.Max(1, servosPerLeg);
            bearingsPerLeg = Mathf.Max(0, bearingsPerLeg);
            servoMassKg = Mathf.Max(0f, servoMassKg);
            bearingMassKg = Mathf.Max(0f, bearingMassKg);
            printedPartsMassPerLegKg = Mathf.Max(0f, printedPartsMassPerLegKg);
            hipLengthM = Mathf.Max(0f, hipLengthM);
            femurLengthM = Mathf.Max(0f, femurLengthM);
            tibiaLengthM = Mathf.Max(0f, tibiaLengthM);
            supplyVoltageV = Mathf.Max(0f, supplyVoltageV);
            stallTorqueKgCm = Mathf.Max(0f, stallTorqueKgCm);
            commandFrequencyHz = Mathf.Max(1f, commandFrequencyHz);
            secondsPer60Degrees = Mathf.Max(0.01f, secondsPer60Degrees);
            recommendedStepHeightM = Mathf.Max(0.01f, recommendedStepHeightM);
            proxyMaxSpeedMps = Mathf.Max(0.01f, proxyMaxSpeedMps);
            proxyAccelerationMps2 = Mathf.Max(0.01f, proxyAccelerationMps2);
            proxyYawSpeedDegreesPerSecond = Mathf.Max(1f, proxyYawSpeedDegreesPerSecond);
        }
    }
}
