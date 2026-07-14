using UnityEngine;

namespace DKSH.Spiderbot.Mapping
{
    [DisallowMultipleComponent]
    public sealed class LidarVirtualSpaceOutput : MonoBehaviour
    {
        [SerializeField]
        private LidarPointCloudMap pointCloudMap = null;

        [SerializeField]
        private Transform virtualSpaceFloor = null;

        [SerializeField]
        private string virtualSpaceFloorName = "VirtualSpaceFloor";

        [SerializeField]
        private string rendererObjectName = "LiDAR Point Cloud";

        [SerializeField, Min(1)]
        private int maxVisiblePoints = 12000;

        [SerializeField, Min(0.001f)]
        private float pointSize = 0.07f;

        [SerializeField]
        private Color pointColor = new Color(0.1f, 1f, 0.55f, 1f);

        [SerializeField, Min(0.001f)]
        private float outputScale = 1f;

        [SerializeField]
        private Vector3 outputOffset = Vector3.zero;

        [SerializeField]
        private Material particleMaterial = null;

        [SerializeField]
        private ParticleSystem pointRenderer = null;

        [SerializeField]
        private bool updateEveryFrame = false;

        private ParticleSystem.Particle[] particles;
        private Material runtimeMaterial;
        private int lastRenderedVersion = -1;
        private bool renderDirty = true;
        private LidarPointCloudMap subscribedMap;
        private Vector3 lastRealFloorScale;
        private Vector3 lastVirtualFloorScale;
        private bool hasLastFloorScales;

        public Transform VirtualSpaceFloor { get { return virtualSpaceFloor; } }
        public ParticleSystem PointRenderer { get { return pointRenderer; } }

        public void RenderNow()
        {
            ResolveReferences();

            if (pointCloudMap == null || pointRenderer == null || virtualSpaceFloor == null)
            {
                return;
            }

            ConfigureParticleSystem();
            EnsureParticleBuffer();

            var points = pointCloudMap.StoredPoints;
            var pointCount = points.Count;
            if (pointCount == 0)
            {
                pointRenderer.SetParticles(particles, 0);
                lastRenderedVersion = pointCloudMap.Version;
                renderDirty = false;
                return;
            }

            var stride = Mathf.Max(1, Mathf.CeilToInt(pointCount / (float)maxVisiblePoints));
            var particleCount = 0;

            for (var i = 0; i < pointCount && particleCount < maxVisiblePoints; i += stride)
            {
                var virtualPoint = ToVirtualSpaceFloorPoint(points[i]);
                virtualPoint = virtualPoint * outputScale + outputOffset;

                particles[particleCount] = new ParticleSystem.Particle
                {
                    position = virtualPoint,
                    startColor = pointColor,
                    startSize = pointSize,
                    startLifetime = 1f,
                    remainingLifetime = 1f
                };
                particleCount++;
            }

            pointRenderer.SetParticles(particles, particleCount);
            lastRenderedVersion = pointCloudMap.Version;
            renderDirty = false;
        }

        private void Reset()
        {
            ResolveReferences();
        }

        private void Awake()
        {
            ResolveReferences();
            ConfigureParticleSystem();
        }

        private void OnEnable()
        {
            ResolveReferences();
            ConfigureParticleSystem();
            SubscribeToMap();
            renderDirty = true;
        }

        private void OnDisable()
        {
            UnsubscribeFromMap();
        }

        private void OnDestroy()
        {
            UnsubscribeFromMap();

            if (runtimeMaterial == null)
            {
                return;
            }

            if (Application.isPlaying)
            {
                Destroy(runtimeMaterial);
            }
            else
            {
                DestroyImmediate(runtimeMaterial);
            }
        }

        private void LateUpdate()
        {
            ResolveReferences();
            SubscribeToMap();

            if (HaveFloorScalesChanged())
            {
                renderDirty = true;
            }

            if (updateEveryFrame || renderDirty || (pointCloudMap != null && pointCloudMap.Version != lastRenderedVersion))
            {
                RenderNow();
            }
        }

        private void OnValidate()
        {
            maxVisiblePoints = Mathf.Max(1, maxVisiblePoints);
            pointSize = Mathf.Max(0.001f, pointSize);
            outputScale = Mathf.Max(0.001f, outputScale);
        }

        private void HandlePointsChanged(LidarPointCloudMap map)
        {
            renderDirty = true;
        }

        private void SubscribeToMap()
        {
            if (subscribedMap == pointCloudMap)
            {
                return;
            }

            UnsubscribeFromMap();
            subscribedMap = pointCloudMap;

            if (subscribedMap != null)
            {
                subscribedMap.PointsChanged += HandlePointsChanged;
                renderDirty = true;
            }
        }

        private void UnsubscribeFromMap()
        {
            if (subscribedMap == null)
            {
                return;
            }

            subscribedMap.PointsChanged -= HandlePointsChanged;
            subscribedMap = null;
        }

        private bool HaveFloorScalesChanged()
        {
            if (pointCloudMap == null || virtualSpaceFloor == null)
            {
                hasLastFloorScales = false;
                return false;
            }

            var realFloor = pointCloudMap.RealSpaceFloor;
            var realFloorScale = realFloor != null ? realFloor.lossyScale : Vector3.one;
            var virtualFloorScale = virtualSpaceFloor.lossyScale;

            if (!hasLastFloorScales)
            {
                StoreFloorScales(realFloorScale, virtualFloorScale);
                return false;
            }

            if (Approximately(lastRealFloorScale, realFloorScale) &&
                Approximately(lastVirtualFloorScale, virtualFloorScale))
            {
                return false;
            }

            StoreFloorScales(realFloorScale, virtualFloorScale);
            return true;
        }

        private void StoreFloorScales(Vector3 realFloorScale, Vector3 virtualFloorScale)
        {
            lastRealFloorScale = realFloorScale;
            lastVirtualFloorScale = virtualFloorScale;
            hasLastFloorScales = true;
        }

        private static bool Approximately(Vector3 a, Vector3 b)
        {
            return Mathf.Approximately(a.x, b.x) &&
                Mathf.Approximately(a.y, b.y) &&
                Mathf.Approximately(a.z, b.z);
        }

        private void ResolveReferences()
        {
            if (pointCloudMap == null)
            {
                pointCloudMap = GetComponent<LidarPointCloudMap>();
            }

            if (pointCloudMap == null)
            {
                pointCloudMap = GetComponentInParent<LidarPointCloudMap>();
            }

            if (virtualSpaceFloor == null)
            {
                var virtualSpaceFloorObject = GameObject.Find(virtualSpaceFloorName);
                if (virtualSpaceFloorObject != null)
                {
                    virtualSpaceFloor = virtualSpaceFloorObject.transform;
                }
            }

            if (pointRenderer == null && virtualSpaceFloor != null)
            {
                var rendererTransform = virtualSpaceFloor.Find(rendererObjectName);
                if (rendererTransform != null)
                {
                    pointRenderer = rendererTransform.GetComponent<ParticleSystem>();
                }
            }

            if (pointRenderer == null && virtualSpaceFloor != null)
            {
                var rendererObject = new GameObject(rendererObjectName);
                rendererObject.transform.SetParent(virtualSpaceFloor, false);
                pointRenderer = rendererObject.AddComponent<ParticleSystem>();
            }
        }

        private void ConfigureParticleSystem()
        {
            if (pointRenderer == null)
            {
                return;
            }

            AlignRendererToVirtualSpaceFloor();

            var main = pointRenderer.main;
            main.loop = false;
            main.playOnAwake = false;
            main.maxParticles = maxVisiblePoints;
            main.simulationSpace = ParticleSystemSimulationSpace.Local;
            main.simulationSpeed = 0f;
            main.startSpeed = 0f;
            main.startLifetime = 1f;

            var emission = pointRenderer.emission;
            emission.enabled = false;

            var shape = pointRenderer.shape;
            shape.enabled = false;

            var renderer = pointRenderer.GetComponent<ParticleSystemRenderer>();
            renderer.renderMode = ParticleSystemRenderMode.Billboard;
            renderer.alignment = ParticleSystemRenderSpace.View;
            renderer.minParticleSize = 0f;
            renderer.maxParticleSize = 1f;

            var resolvedMaterial = particleMaterial != null ? particleMaterial : GetRuntimeMaterial();
            if (resolvedMaterial != null)
            {
                renderer.sharedMaterial = resolvedMaterial;
            }

            if (!pointRenderer.isPlaying)
            {
                pointRenderer.Play();
            }
        }

        private void AlignRendererToVirtualSpaceFloor()
        {
            if (virtualSpaceFloor == null)
            {
                return;
            }

            var rendererTransform = pointRenderer.transform;
            if (rendererTransform.parent != virtualSpaceFloor)
            {
                rendererTransform.SetParent(virtualSpaceFloor, false);
            }

            rendererTransform.localPosition = Vector3.zero;
            rendererTransform.localRotation = Quaternion.identity;
            rendererTransform.localScale = Vector3.one;
        }

        private Vector3 ToVirtualSpaceFloorPoint(Vector3 storedPoint)
        {
            if (pointCloudMap != null)
            {
                return pointCloudMap.StoredPointToVirtualFloorPoint(storedPoint, virtualSpaceFloor);
            }

            return virtualSpaceFloor.InverseTransformPoint(storedPoint);
        }

        private Material GetRuntimeMaterial()
        {
            if (runtimeMaterial != null)
            {
                return runtimeMaterial;
            }

            var shader = Shader.Find("Universal Render Pipeline/Particles/Unlit");
            if (shader == null)
            {
                shader = Shader.Find("Particles/Standard Unlit");
            }

            if (shader == null)
            {
                shader = Shader.Find("Sprites/Default");
            }

            if (shader == null)
            {
                return null;
            }

            runtimeMaterial = new Material(shader)
            {
                name = "Runtime LiDAR Point Material"
            };

            if (runtimeMaterial.HasProperty("_BaseColor"))
            {
                runtimeMaterial.SetColor("_BaseColor", Color.white);
            }

            if (runtimeMaterial.HasProperty("_Color"))
            {
                runtimeMaterial.SetColor("_Color", Color.white);
            }

            return runtimeMaterial;
        }

        private void EnsureParticleBuffer()
        {
            if (particles == null || particles.Length != maxVisiblePoints)
            {
                particles = new ParticleSystem.Particle[maxVisiblePoints];
            }
        }
    }
}
