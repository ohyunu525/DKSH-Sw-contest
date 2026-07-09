using UnityEngine;

namespace DKSH.Spiderbot.Training
{
    [DisallowMultipleComponent]
    [RequireComponent(typeof(Rigidbody))]
    public sealed class RLDebrisHazard : MonoBehaviour
    {
        [SerializeField]
        private Vector2Int cell;

        [SerializeField, Min(0f)]
        private float releaseTime;

        [SerializeField]
        private Vector3 initialLocalPosition;

        [SerializeField]
        private Quaternion initialLocalRotation = Quaternion.identity;

        [SerializeField]
        private Vector3 initialLocalScale = Vector3.one;

        [SerializeField]
        private bool released;

        private Rigidbody body;

        public Vector2Int Cell { get { return cell; } }
        public float ReleaseTime { get { return releaseTime; } }
        public bool Released { get { return released; } }

        public void Configure(
            Vector2Int targetCell,
            float deterministicReleaseTime,
            Vector3 localPosition,
            Quaternion localRotation,
            Vector3 localScale)
        {
            cell = targetCell;
            releaseTime = Mathf.Max(0f, deterministicReleaseTime);
            initialLocalPosition = localPosition;
            initialLocalRotation = localRotation;
            initialLocalScale = localScale;
            body = GetComponent<Rigidbody>();
            ResetDebris();
        }

        public void ResetDebris()
        {
            if (body == null)
            {
                body = GetComponent<Rigidbody>();
            }

            transform.localPosition = initialLocalPosition;
            transform.localRotation = initialLocalRotation;
            transform.localScale = initialLocalScale;
            released = false;

            if (body == null)
            {
                return;
            }

            body.isKinematic = true;
            body.useGravity = false;
            body.linearVelocity = Vector3.zero;
            body.angularVelocity = Vector3.zero;
        }

        public void TickRelease(float elapsedTime)
        {
            if (released || elapsedTime < releaseTime)
            {
                return;
            }

            Release();
        }

        private void Release()
        {
            released = true;
            if (body == null)
            {
                body = GetComponent<Rigidbody>();
            }

            if (body == null)
            {
                return;
            }

            body.isKinematic = false;
            body.useGravity = true;
        }

        private void OnValidate()
        {
            releaseTime = Mathf.Max(0f, releaseTime);
        }
    }
}
