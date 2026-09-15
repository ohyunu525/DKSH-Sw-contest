using System;
using System.Collections.Generic;
using DKSH.Spiderbot.Mapping;
using UnityEngine;

namespace DKSH.Spiderbot.Exploration
{
    /// <summary>
    /// Minimal CharacterController follower for the standalone Unity exploration demo.
    /// It is intentionally not a reusable robot navigation controller.
    /// </summary>
    [DisallowMultipleComponent]
    [RequireComponent(typeof(CharacterController))]
    public sealed class FrontierPathFollower : MonoBehaviour
    {
        [SerializeField, Min(0.01f)]
        private float moveSpeedMetersPerSecond = 0.6f;

        [SerializeField, Min(1f)]
        private float turnSpeedDegreesPerSecond = 240f;

        [SerializeField, Min(0.01f)]
        private float waypointToleranceMeters = 0.08f;

        [SerializeField, Min(0.1f)]
        private float stuckTimeoutSeconds = 2.5f;

        [SerializeField, Min(0.001f)]
        private float minimumProgressMeters = 0.025f;

        [SerializeField, Min(0f)]
        private float gravityMetersPerSecondSquared = 18f;

        private readonly List<Vector3> waypoints = new List<Vector3>(64);
        private CharacterController characterController;
        private int waypointIndex;
        private Vector3 lastProgressPosition;
        private float lastProgressTime;
        private float verticalVelocity;

        public event Action<FrontierPathFollower> PathCompleted;
        public event Action<FrontierPathFollower> PathBlocked;

        public bool IsFollowing { get; private set; }

        public void SetPath(OccupancyGrid2D grid, IReadOnlyList<Vector2Int> gridPath)
        {
            ResolveController();
            waypoints.Clear();
            waypointIndex = 0;
            verticalVelocity = 0f;

            if (grid == null || gridPath == null || gridPath.Count == 0)
            {
                IsFollowing = false;
                return;
            }

            for (var i = 0; i < gridPath.Count; i++)
            {
                waypoints.Add(grid.GridToWorld(gridPath[i].x, gridPath[i].y));
            }

            IsFollowing = true;
            lastProgressPosition = transform.position;
            lastProgressTime = Time.time;
            AdvanceReachedWaypoints();
        }

        public void Stop()
        {
            IsFollowing = false;
            waypoints.Clear();
            waypointIndex = 0;
            verticalVelocity = 0f;
        }

        private void Awake()
        {
            ResolveController();
        }

        private void OnDisable()
        {
            Stop();
        }

        private void Update()
        {
            if (!IsFollowing)
            {
                return;
            }

            ResolveController();
            if (characterController == null || !characterController.enabled)
            {
                ReportBlocked();
                return;
            }

            if (AdvanceReachedWaypoints())
            {
                return;
            }

            var target = waypoints[waypointIndex];
            var planarTarget = new Vector3(target.x, transform.position.y, target.z);
            var direction = planarTarget - transform.position;
            direction.y = 0f;
            if (direction.sqrMagnitude > 0.000001f)
            {
                direction.Normalize();
                var targetRotation = Quaternion.LookRotation(direction, Vector3.up);
                transform.rotation = Quaternion.RotateTowards(
                    transform.rotation,
                    targetRotation,
                    turnSpeedDegreesPerSecond * Time.deltaTime);
            }

            if (characterController.isGrounded && verticalVelocity < 0f)
            {
                verticalVelocity = -1f;
            }
            else
            {
                verticalVelocity -= gravityMetersPerSecondSquared * Time.deltaTime;
            }

            var movement = direction * (moveSpeedMetersPerSecond * Time.deltaTime) +
                           Vector3.up * (verticalVelocity * Time.deltaTime);
            characterController.Move(movement);

            var planarProgress = transform.position - lastProgressPosition;
            planarProgress.y = 0f;
            if (planarProgress.sqrMagnitude >= minimumProgressMeters * minimumProgressMeters)
            {
                lastProgressPosition = transform.position;
                lastProgressTime = Time.time;
            }
            else if (Time.time - lastProgressTime >= stuckTimeoutSeconds)
            {
                ReportBlocked();
            }
        }

        private bool AdvanceReachedWaypoints()
        {
            while (waypointIndex < waypoints.Count)
            {
                var delta = waypoints[waypointIndex] - transform.position;
                delta.y = 0f;
                if (delta.sqrMagnitude > waypointToleranceMeters * waypointToleranceMeters)
                {
                    return false;
                }

                waypointIndex++;
                lastProgressPosition = transform.position;
                lastProgressTime = Time.time;
            }

            IsFollowing = false;
            if (PathCompleted != null)
            {
                PathCompleted(this);
            }

            return true;
        }

        private void ReportBlocked()
        {
            IsFollowing = false;
            if (PathBlocked != null)
            {
                PathBlocked(this);
            }
        }

        private void ResolveController()
        {
            if (characterController == null)
            {
                characterController = GetComponent<CharacterController>();
            }
        }

        private void OnValidate()
        {
            moveSpeedMetersPerSecond = Mathf.Max(0.01f, moveSpeedMetersPerSecond);
            turnSpeedDegreesPerSecond = Mathf.Max(1f, turnSpeedDegreesPerSecond);
            waypointToleranceMeters = Mathf.Max(0.01f, waypointToleranceMeters);
            stuckTimeoutSeconds = Mathf.Max(0.1f, stuckTimeoutSeconds);
            minimumProgressMeters = Mathf.Max(0.001f, minimumProgressMeters);
            gravityMetersPerSecondSquared = Mathf.Max(0f, gravityMetersPerSecondSquared);
            ResolveController();
        }
    }
}
