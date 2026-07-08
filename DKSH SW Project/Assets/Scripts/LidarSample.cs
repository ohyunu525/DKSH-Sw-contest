using System;
using UnityEngine;

namespace DKSH.Spiderbot.Sensors
{
    [Serializable]
    public struct LidarSample
    {
        public int horizontalIndex;
        public int verticalIndex;
        public Vector3 origin;
        public Vector3 direction;
        public Vector3 point;
        public Vector3 normal;
        public float distance;
        public bool hit;
        public int colliderInstanceId;

        public LidarSample(
            int horizontalIndex,
            int verticalIndex,
            Vector3 origin,
            Vector3 direction,
            Vector3 point,
            Vector3 normal,
            float distance,
            bool hit,
            int colliderInstanceId)
        {
            this.horizontalIndex = horizontalIndex;
            this.verticalIndex = verticalIndex;
            this.origin = origin;
            this.direction = direction;
            this.point = point;
            this.normal = normal;
            this.distance = distance;
            this.hit = hit;
            this.colliderInstanceId = colliderInstanceId;
        }
    }
}
