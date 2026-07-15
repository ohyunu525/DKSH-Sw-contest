using System;
using UnityEngine;

namespace DKSH.Spiderbot.Sensors
{
    [DisallowMultipleComponent]
    public sealed class SimulatedLidarSensor : MonoBehaviour
    {
        [SerializeField]
        private LidarScanProfile profile = null;

        [SerializeField]
        private LidarScanSettings settings = LidarScanSettings.Default;

        [SerializeField]
        private bool scanAutomatically = true;

        [SerializeField]
        private bool scanInFixedUpdate = true;

        [SerializeField]
        private bool scanOnEnable = true;

        [Header("Debug")]
        [SerializeField]
        private bool drawDebugRays = true;

        [SerializeField]
        private Color hitRayColor = new Color(0.05f, 0.85f, 1f, 0.85f);

        [SerializeField]
        private Color missRayColor = new Color(0.35f, 0.35f, 0.35f, 0.25f);

        [SerializeField, Min(0f)]
        private float hitPointRadius = 0.035f;

        [SerializeField, Min(1)]
        private int maxDebugRays = 256;

        private float scanTimer;
        private bool initialScanPending;
        private LidarScanFrame latestFrame;

        public event Action<LidarScanFrame> ScanCompleted;

        public LidarScanFrame LatestFrame { get { return latestFrame; } }
        public bool HasFrame { get { return latestFrame != null; } }

        public LidarScanSettings ActiveSettings
        {
            get
            {
                var activeSettings = profile != null ? profile.Settings : settings;
                activeSettings.Clamp();
                return activeSettings;
            }
        }

        public void Configure(LidarScanSettings configuredSettings, bool automatic)
        {
            profile = null;
            settings = configuredSettings;
            settings.Clamp();
            scanAutomatically = automatic;
            scanTimer = 0f;
            initialScanPending = scanOnEnable;
        }

        public bool TryGetLatestFrame(out LidarScanFrame frame)
        {
            frame = latestFrame;
            return frame != null;
        }

        public LidarScanFrame ScanNow()
        {
            var activeSettings = ActiveSettings;
            var samples = new LidarSample[activeSettings.SampleCount];
            var origin = transform.position;
            var sensorRotation = transform.rotation;
            var scanRotation = activeSettings.rotateWithSensor ? sensorRotation : Quaternion.identity;

            var sampleCount = PopulateSamples(activeSettings, origin, scanRotation, samples, out var hitCount);

            if (sampleCount < samples.Length)
            {
                Array.Resize(ref samples, sampleCount);
            }

            latestFrame = CreateFrame(activeSettings, origin, sensorRotation, hitCount, samples);
            PublishScanCompleted(latestFrame);

            return latestFrame;
        }

        private void Reset()
        {
            settings = LidarScanSettings.Default;
        }

        private void OnEnable()
        {
            scanTimer = 0f;
            initialScanPending = scanOnEnable;
        }

        private void OnDisable()
        {
            initialScanPending = false;
        }

        private void Update()
        {
            if (!scanInFixedUpdate)
            {
                ScanPendingInitialFrame();
                Tick(Time.deltaTime);
            }
        }

        private void FixedUpdate()
        {
            if (scanInFixedUpdate)
            {
                ScanPendingInitialFrame();
                Tick(Time.fixedDeltaTime);
            }
        }

        private void Tick(float deltaTime)
        {
            if (!scanAutomatically)
            {
                return;
            }

            var activeSettings = ActiveSettings;
            var scanPeriod = 1f / activeSettings.scanFrequencyHz;
            scanTimer += deltaTime;

            var scansThisTick = 0;
            while (scanTimer >= scanPeriod && scansThisTick < 4)
            {
                scanTimer -= scanPeriod;
                ScanNow();
                scansThisTick++;
            }
        }

        private int PopulateSamples(
            LidarScanSettings activeSettings,
            Vector3 origin,
            Quaternion scanRotation,
            LidarSample[] samples,
            out int hitCount)
        {
            var sampleCount = 0;
            hitCount = 0;

            for (var vertical = 0; vertical < activeSettings.verticalResolution; vertical++)
            {
                var verticalAngle = GetVerticalAngle(activeSettings, vertical);

                for (var horizontal = 0; horizontal < activeSettings.horizontalResolution; horizontal++)
                {
                    var direction = GetRayDirection(activeSettings, scanRotation, horizontal, verticalAngle);
                    var didHit = CastRay(activeSettings, origin, direction, out var hitInfo);

                    if (didHit)
                    {
                        hitCount++;
                    }

                    if (!didHit && !activeSettings.includeMisses)
                    {
                        continue;
                    }

                    samples[sampleCount] = CreateSample(
                        horizontal,
                        vertical,
                        origin,
                        direction,
                        activeSettings.maxDistance,
                        didHit,
                        hitInfo);
                    sampleCount++;
                }
            }

            return sampleCount;
        }

        private static bool CastRay(
            LidarScanSettings activeSettings,
            Vector3 origin,
            Vector3 direction,
            out RaycastHit hitInfo)
        {
            return Physics.Raycast(
                origin,
                direction,
                out hitInfo,
                activeSettings.maxDistance,
                activeSettings.detectionMask.value,
                activeSettings.triggerInteraction);
        }

        private static LidarSample CreateSample(
            int horizontal,
            int vertical,
            Vector3 origin,
            Vector3 direction,
            float maxDistance,
            bool didHit,
            RaycastHit hitInfo)
        {
            var point = didHit ? hitInfo.point : origin + direction * maxDistance;
            var normal = didHit ? hitInfo.normal : Vector3.zero;
            var distance = didHit ? hitInfo.distance : maxDistance;
            var colliderInstanceId = didHit && hitInfo.collider != null ? hitInfo.collider.GetHashCode() : 0;

            return new LidarSample(
                horizontal,
                vertical,
                origin,
                direction,
                point,
                normal,
                distance,
                didHit,
                colliderInstanceId);
        }

        private static LidarScanFrame CreateFrame(
            LidarScanSettings activeSettings,
            Vector3 origin,
            Quaternion sensorRotation,
            int hitCount,
            LidarSample[] samples)
        {
            return new LidarScanFrame(
                Time.timeAsDouble,
                origin,
                sensorRotation,
                activeSettings.horizontalResolution,
                activeSettings.verticalResolution,
                hitCount,
                samples);
        }

        private void PublishScanCompleted(LidarScanFrame frame)
        {
            if (ScanCompleted != null)
            {
                ScanCompleted(frame);
            }
        }

        private void ScanPendingInitialFrame()
        {
            if (!initialScanPending)
            {
                return;
            }

            initialScanPending = false;
            ScanNow();
        }

        private static Vector3 GetRayDirection(
            LidarScanSettings activeSettings,
            Quaternion scanRotation,
            int horizontalIndex,
            float verticalAngle)
        {
            var horizontalAngle = GetHorizontalAngle(activeSettings, horizontalIndex);
            var localRotation =
                Quaternion.AngleAxis(horizontalAngle, Vector3.up) *
                Quaternion.AngleAxis(-verticalAngle, Vector3.right);

            return scanRotation * (localRotation * Vector3.forward);
        }

        private static float GetVerticalAngle(LidarScanSettings activeSettings, int verticalIndex)
        {
            var verticalT = activeSettings.verticalResolution <= 1
                ? 0.5f
                : verticalIndex / (activeSettings.verticalResolution - 1f);

            return Mathf.Lerp(
                -activeSettings.verticalFovDegrees * 0.5f,
                activeSettings.verticalFovDegrees * 0.5f,
                verticalT);
        }

        private static float GetHorizontalAngle(LidarScanSettings activeSettings, int horizontalIndex)
        {
            if (activeSettings.horizontalResolution <= 1)
            {
                return 0f;
            }

            if (activeSettings.horizontalFovDegrees >= 359.999f)
            {
                return -180f + 360f * horizontalIndex / activeSettings.horizontalResolution;
            }

            var horizontalT = horizontalIndex / (activeSettings.horizontalResolution - 1f);
            return Mathf.Lerp(
                -activeSettings.horizontalFovDegrees * 0.5f,
                activeSettings.horizontalFovDegrees * 0.5f,
                horizontalT);
        }

        private void OnValidate()
        {
            settings.Clamp();
            hitPointRadius = Mathf.Max(0f, hitPointRadius);
            maxDebugRays = Mathf.Max(1, maxDebugRays);
        }

        private void OnDrawGizmosSelected()
        {
            if (!drawDebugRays || latestFrame == null)
            {
                return;
            }

            var samples = latestFrame.Samples;
            var stride = Mathf.Max(1, samples.Count / maxDebugRays);

            for (var i = 0; i < samples.Count; i += stride)
            {
                var sample = samples[i];
                Gizmos.color = sample.hit ? hitRayColor : missRayColor;
                Gizmos.DrawLine(sample.origin, sample.point);

                if (sample.hit && hitPointRadius > 0f)
                {
                    Gizmos.DrawSphere(sample.point, hitPointRadius);
                }
            }
        }
    }
}
