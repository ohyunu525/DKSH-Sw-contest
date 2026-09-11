using System;
using DKSH.Spiderbot.Training;
using UnityEngine;

namespace DKSH.Spiderbot.Navigation
{
    [DisallowMultipleComponent]
    [RequireComponent(typeof(CharacterController))]
    public sealed class SpiderProxyLocomotion : MonoBehaviour, ISpiderLocomotionController
    {
        [SerializeField] private SpiderRobotPhysicalProfile physicalProfile;
        [SerializeField, Min(0f)] private float gravityMps2 = 18f;

        private CharacterController characterController;
        private Vector3 desiredLocalVelocity;
        private Vector3 currentLocalVelocity;
        private float desiredYawRateDegrees;
        private float currentYawRateDegrees;
        private float verticalVelocity;

        public event Action<float> CollisionOccurred;

        public Vector3 LocalVelocity { get { return currentLocalVelocity; } }
        public float YawRateRadians { get { return currentYawRateDegrees * Mathf.Deg2Rad; } }
        public SpiderRobotPhysicalProfile PhysicalProfile { get { return physicalProfile; } }

        public void Configure(SpiderRobotPhysicalProfile profile)
        {
            physicalProfile = profile != null ? profile : SpiderRobotPhysicalProfile.CreateRuntimeDefault();
            EnsureReferences();
            ConfigureCharacterController();
        }

        public void SetCommand(Vector3 normalizedLocalPlanarCommand, float normalizedYawCommand)
        {
            EnsureProfile();
            var planar = new Vector3(normalizedLocalPlanarCommand.x, 0f, normalizedLocalPlanarCommand.z);
            planar = Vector3.ClampMagnitude(planar, 1f);
            desiredLocalVelocity = planar * physicalProfile.ProxyMaxSpeedMps;
            desiredYawRateDegrees = Mathf.Clamp(normalizedYawCommand, -1f, 1f)
                * physicalProfile.ProxyYawSpeedDegreesPerSecond;
        }

        public void ResetMotion(Vector3 worldPosition, Quaternion worldRotation)
        {
            EnsureReferences();
            var wasEnabled = characterController.enabled;
            characterController.enabled = false;
            transform.SetPositionAndRotation(worldPosition, worldRotation);
            characterController.enabled = wasEnabled;
            desiredLocalVelocity = Vector3.zero;
            currentLocalVelocity = Vector3.zero;
            desiredYawRateDegrees = 0f;
            currentYawRateDegrees = 0f;
            verticalVelocity = 0f;
        }

        private void Awake()
        {
            EnsureProfile();
            EnsureReferences();
            ConfigureCharacterController();
        }

        private void FixedUpdate()
        {
            EnsureProfile();
            EnsureReferences();

            var deltaTime = Time.fixedDeltaTime;
            var acceleration = physicalProfile.ProxyAccelerationMps2;
            currentLocalVelocity = Vector3.MoveTowards(
                currentLocalVelocity,
                desiredLocalVelocity,
                acceleration * deltaTime);

            currentYawRateDegrees = Mathf.MoveTowards(
                currentYawRateDegrees,
                desiredYawRateDegrees,
                physicalProfile.ProxyYawSpeedDegreesPerSecond * 3f * deltaTime);
            transform.Rotate(0f, currentYawRateDegrees * deltaTime, 0f, Space.Self);

            if (characterController.isGrounded && verticalVelocity < 0f)
            {
                verticalVelocity = -1f;
            }
            else
            {
                verticalVelocity -= gravityMps2 * deltaTime;
            }

            var worldPlanarVelocity = transform.TransformDirection(currentLocalVelocity);
            var motion = (worldPlanarVelocity + Vector3.up * verticalVelocity) * deltaTime;
            var flags = characterController.Move(motion);
            if ((flags & CollisionFlags.Above) != 0 && verticalVelocity > 0f)
            {
                verticalVelocity = 0f;
            }
        }

        private void OnControllerColliderHit(ControllerColliderHit hit)
        {
            if (Vector3.Dot(hit.normal, Vector3.up) > 0.5f)
            {
                return;
            }

            if (CollisionOccurred != null)
            {
                CollisionOccurred(currentLocalVelocity.magnitude);
            }
        }

        private void EnsureProfile()
        {
            if (physicalProfile == null)
            {
                physicalProfile = SpiderRobotPhysicalProfile.CreateRuntimeDefault();
            }
        }

        private void EnsureReferences()
        {
            if (characterController == null)
            {
                characterController = GetComponent<CharacterController>();
            }
        }

        private void ConfigureCharacterController()
        {
            if (characterController == null || physicalProfile == null)
            {
                return;
            }

            characterController.radius = 0.18f;
            characterController.height = 0.28f;
            characterController.center = Vector3.zero;
            characterController.slopeLimit = 50f;
            characterController.stepOffset = Mathf.Min(
                physicalProfile.RecommendedStepHeightM,
                characterController.height - 0.01f);
            characterController.skinWidth = 0.02f;
            characterController.minMoveDistance = 0f;
        }

        private void OnValidate()
        {
            gravityMps2 = Mathf.Max(0f, gravityMps2);
            EnsureReferences();
            if (physicalProfile != null)
            {
                ConfigureCharacterController();
            }
        }
    }
}
