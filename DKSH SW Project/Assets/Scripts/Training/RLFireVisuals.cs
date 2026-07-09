using UnityEngine;

namespace DKSH.Spiderbot.Training
{
    internal static class RLFireVisuals
    {
        public static void Apply(GameObject fireObject, float cellSize, int seed)
        {
            if (fireObject == null)
            {
                return;
            }

            ApplyEmissiveMaterial(fireObject);
            CreateParticleFlame(fireObject.transform, cellSize);
            CreateFlickerLight(fireObject.transform, cellSize, seed);
        }

        private static void ApplyEmissiveMaterial(GameObject fireObject)
        {
            var renderer = fireObject.GetComponent<Renderer>();
            if (renderer == null)
            {
                return;
            }

            var shader = Shader.Find("Universal Render Pipeline/Lit");
            if (shader == null)
            {
                shader = Shader.Find("Standard");
            }

            if (shader == null)
            {
                return;
            }

            var material = new Material(shader)
            {
                name = "Runtime Fire Emissive"
            };

            var baseColor = new Color(1f, 0.26f, 0.03f, 1f);
            var emissionColor = new Color(2.2f, 0.45f, 0.05f, 1f);
            if (material.HasProperty("_BaseColor"))
            {
                material.SetColor("_BaseColor", baseColor);
            }

            if (material.HasProperty("_Color"))
            {
                material.SetColor("_Color", baseColor);
            }

            if (material.HasProperty("_EmissionColor"))
            {
                material.EnableKeyword("_EMISSION");
                material.SetColor("_EmissionColor", emissionColor);
            }

            renderer.sharedMaterial = material;
        }

        private static void CreateParticleFlame(Transform parent, float cellSize)
        {
            var particleObject = new GameObject("FireParticles");
            particleObject.transform.SetParent(parent, false);
            particleObject.transform.localPosition = new Vector3(0f, 0.15f, 0f);
            particleObject.transform.localScale = Vector3.one;

            var particles = particleObject.AddComponent<ParticleSystem>();
            var main = particles.main;
            main.loop = true;
            main.playOnAwake = true;
            main.startLifetime = new ParticleSystem.MinMaxCurve(0.35f, 0.8f);
            main.startSpeed = new ParticleSystem.MinMaxCurve(0.8f, 1.8f);
            main.startSize = new ParticleSystem.MinMaxCurve(cellSize * 0.18f, cellSize * 0.42f);
            main.startColor = new ParticleSystem.MinMaxGradient(
                new Color(1f, 0.2f, 0.02f, 0.92f),
                new Color(1f, 0.8f, 0.08f, 0.75f));
            main.simulationSpace = ParticleSystemSimulationSpace.Local;
            main.maxParticles = 64;

            var emission = particles.emission;
            emission.rateOverTime = 18f;

            var shape = particles.shape;
            shape.enabled = true;
            shape.shapeType = ParticleSystemShapeType.Cone;
            shape.angle = 18f;
            shape.radius = cellSize * 0.28f;

            var renderer = particles.GetComponent<ParticleSystemRenderer>();
            renderer.renderMode = ParticleSystemRenderMode.Billboard;
        }

        private static void CreateFlickerLight(Transform parent, float cellSize, int seed)
        {
            var lightObject = new GameObject("FireLight");
            lightObject.transform.SetParent(parent, false);
            lightObject.transform.localPosition = new Vector3(0f, cellSize * 0.75f, 0f);

            var pointLight = lightObject.AddComponent<Light>();
            pointLight.type = LightType.Point;
            pointLight.color = new Color(1f, 0.35f, 0.06f, 1f);
            pointLight.range = Mathf.Max(2.5f, cellSize * 3f);
            pointLight.intensity = 1.25f;

            var flicker = lightObject.AddComponent<RLFireLightFlicker>();
            flicker.Configure(seed, 0.8f, 1.8f, 8f);
        }
    }
}
