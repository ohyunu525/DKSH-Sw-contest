using DKSH.Spiderbot.Exploration;
using UnityEngine;

namespace DKSH.Spiderbot.ROS
{
    public enum LocomotionAvailability
    {
        Normal,
        SlowdownRequired,
        Paused,
        Unavailable
    }

    /// <summary>
    /// Project-owned boundary between a navigation Twist and hexapod locomotion.
    /// The CharacterController implementation is a Unity proxy; gait and RL policy
    /// controllers can later replace its internals without changing the ROS bridge.
    /// </summary>
    [DisallowMultipleComponent]
    [RequireComponent(typeof(CharacterController))]
    public sealed class HexapodVelocityCommandAdapter : MonoBehaviour
    {
        [Header("Standalone mode components")]
        [SerializeField] private FrontierPathFollower standaloneFollower;
        [SerializeField] private FrontierExplorerCoordinator standaloneCoordinator;

        [Header("Hexapod command limits")]
        [SerializeField, Min(0f)] private float maximumForwardMetersPerSecond = 0.35f;
        [SerializeField, Min(0f)] private float maximumBackwardMetersPerSecond = 0.2f;
        [SerializeField, Min(0f)] private float maximumLateralMetersPerSecond = 0.25f;
        [SerializeField, Min(0f)] private float maximumYawRadiansPerSecond = 0.8f;
        [SerializeField, Min(0f)] private float linearAccelerationMetersPerSecondSquared = 0.8f;
        [SerializeField, Min(0f)] private float yawAccelerationRadiansPerSecondSquared = 2f;
        [SerializeField, Min(0.05f)] private float commandTimeoutSeconds = 0.5f;
        [SerializeField, Min(0f)] private float gravityMetersPerSecondSquared = 18f;

        private CharacterController characterController;
        private bool externalControlEnabled;
        private bool restoreFollower;
        private bool restoreCoordinator;
        private float requestedForward;
        private float requestedLeft;
        private float requestedYaw;
        private Vector3 currentLocalVelocity;
        private float currentYawRadians;
        private float verticalVelocity;
        private float lastCommandRealTime = float.NegativeInfinity;
        private LocomotionAvailability requestedAvailability = LocomotionAvailability.Normal;

        public bool ExternalControlEnabled { get { return externalControlEnabled; } }
        public bool HasFreshCommand
        {
            get
            {
                return externalControlEnabled &&
                       Time.realtimeSinceStartup - lastCommandRealTime <= commandTimeoutSeconds;
            }
        }

        public LocomotionAvailability EffectiveAvailability
        {
            get
            {
                if (!externalControlEnabled || !HasFreshCommand)
                {
                    return LocomotionAvailability.Paused;
                }

                return requestedAvailability;
            }
        }

        public Vector3 CurrentLocalVelocity { get { return currentLocalVelocity; } }
        public float CurrentYawRadiansPerSecond { get { return currentYawRadians; } }

        public void ConfigureStandaloneMode(
            FrontierPathFollower follower,
            FrontierExplorerCoordinator coordinator)
        {
            standaloneFollower = follower;
            standaloneCoordinator = coordinator;
        }

        public void ConfigureLimits(
            float forwardMetersPerSecond,
            float backwardMetersPerSecond,
            float lateralMetersPerSecond,
            float yawRadiansPerSecond,
            float commandTimeout)
        {
            maximumForwardMetersPerSecond = Mathf.Max(0f, forwardMetersPerSecond);
            maximumBackwardMetersPerSecond = Mathf.Max(0f, backwardMetersPerSecond);
            maximumLateralMetersPerSecond = Mathf.Max(0f, lateralMetersPerSecond);
            maximumYawRadiansPerSecond = Mathf.Max(0f, yawRadiansPerSecond);
            commandTimeoutSeconds = Mathf.Max(0.05f, commandTimeout);
        }

        public void SetExternalControlEnabled(bool enabled)
        {
            if (externalControlEnabled == enabled)
            {
                return;
            }

            externalControlEnabled = enabled;
            StopImmediately();

            if (enabled)
            {
                restoreFollower = standaloneFollower != null && standaloneFollower.enabled;
                restoreCoordinator = standaloneCoordinator != null && standaloneCoordinator.enabled;

                if (standaloneFollower != null)
                {
                    standaloneFollower.enabled = false;
                }

                if (standaloneCoordinator != null)
                {
                    standaloneCoordinator.enabled = false;
                }
            }
            else
            {
                if (standaloneFollower != null)
                {
                    standaloneFollower.enabled = restoreFollower;
                }

                if (standaloneCoordinator != null)
                {
                    standaloneCoordinator.enabled = restoreCoordinator;
                }

                restoreFollower = false;
                restoreCoordinator = false;
            }
        }

        public void SetNavigationAvailability(LocomotionAvailability availability)
        {
            requestedAvailability = availability;
            if (availability == LocomotionAvailability.Paused ||
                availability == LocomotionAvailability.Unavailable)
            {
                StopImmediately();
            }
        }

        public void SetVelocityCommand(
            float forwardMetersPerSecond,
            float leftMetersPerSecond,
            float yawRadiansPerSecond)
        {
            if (!externalControlEnabled)
            {
                return;
            }

            requestedForward = Mathf.Clamp(
                forwardMetersPerSecond,
                -maximumBackwardMetersPerSecond,
                maximumForwardMetersPerSecond);
            requestedLeft = Mathf.Clamp(
                leftMetersPerSecond,
                -maximumLateralMetersPerSecond,
                maximumLateralMetersPerSecond);
            requestedYaw = Mathf.Clamp(
                yawRadiansPerSecond,
                -maximumYawRadiansPerSecond,
                maximumYawRadiansPerSecond);
            lastCommandRealTime = Time.realtimeSinceStartup;
        }

        public void EmergencyStop()
        {
            requestedAvailability = LocomotionAvailability.Paused;
            StopImmediately();
        }

        private void Awake()
        {
            characterController = GetComponent<CharacterController>();
        }

        private void OnDisable()
        {
            SetExternalControlEnabled(false);
        }

        private void FixedUpdate()
        {
            if (!externalControlEnabled)
            {
                return;
            }

            if (!HasFreshCommand ||
                requestedAvailability == LocomotionAvailability.Paused ||
                requestedAvailability == LocomotionAvailability.Unavailable)
            {
                StopImmediately();
                ApplyGravity(Time.fixedDeltaTime);
                return;
            }

            var scale = requestedAvailability == LocomotionAvailability.SlowdownRequired
                ? 0.5f
                : 1f;
            var desiredLocalVelocity = new Vector3(
                -requestedLeft * scale,
                0f,
                requestedForward * scale);
            var desiredYaw = requestedYaw * scale;
            var deltaTime = Time.fixedDeltaTime;

            currentLocalVelocity = Vector3.MoveTowards(
                currentLocalVelocity,
                desiredLocalVelocity,
                linearAccelerationMetersPerSecondSquared * deltaTime);
            currentYawRadians = Mathf.MoveTowards(
                currentYawRadians,
                desiredYaw,
                yawAccelerationRadiansPerSecondSquared * deltaTime);

            // Unity positive yaw turns right in this project; ROS REP-103 positive yaw turns left.
            transform.Rotate(
                0f,
                -currentYawRadians * Mathf.Rad2Deg * deltaTime,
                0f,
                Space.Self);
            ApplyMotion(deltaTime);
        }

        private void ApplyGravity(float deltaTime)
        {
            ApplyMotion(deltaTime);
        }

        private void ApplyMotion(float deltaTime)
        {
            if (characterController == null)
            {
                characterController = GetComponent<CharacterController>();
            }

            if (characterController.isGrounded && verticalVelocity < 0f)
            {
                verticalVelocity = -1f;
            }
            else
            {
                verticalVelocity -= gravityMetersPerSecondSquared * deltaTime;
            }

            var worldPlanarVelocity = transform.TransformDirection(currentLocalVelocity);
            characterController.Move(
                (worldPlanarVelocity + Vector3.up * verticalVelocity) * deltaTime);
        }

        private void StopImmediately()
        {
            requestedForward = 0f;
            requestedLeft = 0f;
            requestedYaw = 0f;
            currentLocalVelocity = Vector3.zero;
            currentYawRadians = 0f;
        }

        private void OnValidate()
        {
            maximumForwardMetersPerSecond = Mathf.Max(0f, maximumForwardMetersPerSecond);
            maximumBackwardMetersPerSecond = Mathf.Max(0f, maximumBackwardMetersPerSecond);
            maximumLateralMetersPerSecond = Mathf.Max(0f, maximumLateralMetersPerSecond);
            maximumYawRadiansPerSecond = Mathf.Max(0f, maximumYawRadiansPerSecond);
            linearAccelerationMetersPerSecondSquared = Mathf.Max(
                0f,
                linearAccelerationMetersPerSecondSquared);
            yawAccelerationRadiansPerSecondSquared = Mathf.Max(
                0f,
                yawAccelerationRadiansPerSecondSquared);
            commandTimeoutSeconds = Mathf.Max(0.05f, commandTimeoutSeconds);
            gravityMetersPerSecondSquared = Mathf.Max(0f, gravityMetersPerSecondSquared);
        }
    }
}
