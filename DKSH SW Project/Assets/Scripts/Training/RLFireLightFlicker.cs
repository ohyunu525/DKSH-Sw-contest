using UnityEngine;

namespace DKSH.Spiderbot.Training
{
    [DisallowMultipleComponent]
    public sealed class RLFireLightFlicker : MonoBehaviour
    {
        [SerializeField]
        private int seed;

        [SerializeField, Min(0f)]
        private float minIntensity = 0.8f;

        [SerializeField, Min(0f)]
        private float maxIntensity = 1.8f;

        [SerializeField, Min(0.1f)]
        private float frequency = 8f;

        private Light targetLight;
        private float phase;

        public void Configure(int deterministicSeed, float minimumIntensity, float maximumIntensity, float flickerFrequency)
        {
            seed = deterministicSeed;
            minIntensity = Mathf.Max(0f, minimumIntensity);
            maxIntensity = Mathf.Max(minIntensity, maximumIntensity);
            frequency = Mathf.Max(0.1f, flickerFrequency);
            phase = Mathf.Abs(seed % 997) * 0.013f;
            targetLight = GetComponent<Light>();
        }

        private void Awake()
        {
            targetLight = GetComponent<Light>();
            phase = Mathf.Abs(seed % 997) * 0.013f;
        }

        private void Update()
        {
            if (targetLight == null)
            {
                return;
            }

            var wave = Mathf.PerlinNoise(phase, Time.time * frequency);
            targetLight.intensity = Mathf.Lerp(minIntensity, maxIntensity, wave);
        }

        private void OnValidate()
        {
            minIntensity = Mathf.Max(0f, minIntensity);
            maxIntensity = Mathf.Max(minIntensity, maxIntensity);
            frequency = Mathf.Max(0.1f, frequency);
        }
    }
}
