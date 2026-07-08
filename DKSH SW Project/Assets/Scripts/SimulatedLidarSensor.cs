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

            var index = 0;
            var hitCount = 0;

            for (var vertical = 0; vertical < activeSettings.verticalResolution; vertical++)
            {
                var verticalT = activeSettings.verticalResolution <= 1
                    ? 0.5f
                    : vertical / (activeSettings.verticalResolution - 1f);
                var verticalAngle = Mathf.Lerp(
                    -activeSettings.verticalFovDegrees * 0.5f,
                    activeSettings.verticalFovDegrees * 0.5f,
                    verticalT);

                for (var horizontal = 0; horizontal < activeSettings.horizontalResolution; horizontal++)
                {
                    var horizontalAngle = GetHorizontalAngle(activeSettings, horizontal);

                    var localRotation =
                        Quaternion.AngleAxis(horizontalAngle, Vector3.up) *
                        Quaternion.AngleAxis(-verticalAngle, Vector3.right);
                    var direction = scanRotation * (localRotation * Vector3.forward);

                    RaycastHit hitInfo;
                    var didHit = Physics.Raycast(
                        origin,
                        direction,
                        out hitInfo,
                        activeSettings.maxDistance,
                        activeSettings.detectionMask.value,
                        activeSettings.triggerInteraction);

                    if (didHit)
                    {
                        hitCount++;
                    }

                    var point = didHit ? hitInfo.point : origin + direction * activeSettings.maxDistance;
                    var normal = didHit ? hitInfo.normal : Vector3.zero;
                    var distance = didHit ? hitInfo.distance : activeSettings.maxDistance;
                    var colliderInstanceId = didHit && hitInfo.collider != null ? hitInfo.collider.GetInstanceID() : 0;

                    if (!didHit && !activeSettings.includeMisses)
                    {
                        continue;
                    }

                    samples[index] = new LidarSample(
                        horizontal,
                        vertical,
                        origin,
                        direction,
                        point,
                        normal,
                        distance,
                        didHit,
                        colliderInstanceId);
                    index++;
                }
            }

            if (index < samples.Length)
            {
                Array.Resize(ref samples, index);
            }

            latestFrame = new LidarScanFrame(
                Time.timeAsDouble,
                origin,
                sensorRotation,
                activeSettings.horizontalResolution,
                activeSettings.verticalResolution,
                hitCount,
                samples);

            if (ScanCompleted != null)
            {
                ScanCompleted(latestFrame);
            }

            return latestFrame;
        }

        private void Reset()
        {
            settings = LidarScanSettings.Default;
        }

        private void OnEnable()
        {
            scanTimer = 0f;

            if (scanOnEnable)
            {
                ScanNow();
            }
        }

        private void Update()
        {
            if (!scanInFixedUpdate)
            {
                Tick(Time.deltaTime);
            }
        }

        private void FixedUpdate()
        {
            if (scanInFixedUpdate)
            {
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
