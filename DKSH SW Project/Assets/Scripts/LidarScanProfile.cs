using UnityEngine;

namespace DKSH.Spiderbot.Sensors
{
    [CreateAssetMenu(fileName = "LidarScanProfile", menuName = "Spiderbot/Sensors/LiDAR Scan Profile")]
    public sealed class LidarScanProfile : ScriptableObject
    {
        [SerializeField]
        private LidarScanSettings settings = LidarScanSettings.Default;

        public LidarScanSettings Settings
        {
            get
            {
                var resolvedSettings = settings;
                resolvedSettings.Clamp();
                return resolvedSettings;
            }
        }

        private void Reset()
        {
            settings = LidarScanSettings.Default;
        }

        private void OnValidate()
        {
            settings.Clamp();
        }
    }
}
