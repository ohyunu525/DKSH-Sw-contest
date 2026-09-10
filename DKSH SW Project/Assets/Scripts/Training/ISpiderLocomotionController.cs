using UnityEngine;

namespace DKSH.Spiderbot.Navigation
{
    public interface ISpiderLocomotionController
    {
        Vector3 LocalVelocity { get; }
        float YawRateRadians { get; }

        void SetCommand(Vector3 normalizedLocalPlanarCommand, float normalizedYawCommand);
        void ResetMotion(Vector3 worldPosition, Quaternion worldRotation);
    }
}
