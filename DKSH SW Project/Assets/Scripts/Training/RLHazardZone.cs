using UnityEngine;

namespace DKSH.Spiderbot.Training
{
    [DisallowMultipleComponent]
    public sealed class RLHazardZone : MonoBehaviour
    {
        [SerializeField]
        private RLHazardKind hazardKind = RLHazardKind.Fire;

        [SerializeField]
        private Vector2Int cell;

        [SerializeField, Min(0f)]
        private float radius = 0.45f;

        [SerializeField]
        private bool activeOnReset = true;

        [SerializeField]
        private bool isActive = true;

        public RLHazardKind HazardKind { get { return hazardKind; } }
        public Vector2Int Cell { get { return cell; } }
        public float Radius { get { return radius; } }
        public bool ActiveOnReset { get { return activeOnReset; } }
        public bool IsActive { get { return isActive; } }

        public void Initialize(RLHazardKind kind, Vector2Int gridCell, float zoneRadius, bool startsActive)
        {
            hazardKind = kind;
            cell = gridCell;
            radius = Mathf.Max(0f, zoneRadius);
            activeOnReset = startsActive;
            SetActive(startsActive);
        }

        public void ResetZone()
        {
            SetActive(activeOnReset);
        }

        public void SetActive(bool active)
        {
            isActive = active;
            gameObject.SetActive(active);
        }

        private void OnValidate()
        {
            radius = Mathf.Max(0f, radius);
        }
    }
}
