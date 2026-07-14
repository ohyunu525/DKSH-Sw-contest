using System;
using System.Collections.Generic;
using UnityEngine;

namespace DKSH.Spiderbot.Training
{
    [DisallowMultipleComponent]
    public sealed class RLSpreadingFireController : MonoBehaviour
    {
        [SerializeField]
        private int seed;

        [SerializeField]
        private Vector2Int mapSize;

        [SerializeField, Min(0.25f)]
        private float cellSize = 1f;

        [SerializeField, Min(0.1f)]
        private float spreadInterval = 8f;

        [SerializeField, Min(0.1f)]
        private float minIgnitionDelay = 6f;

        [SerializeField, Min(0.1f)]
        private float maxIgnitionDelay = 12f;

        [SerializeField]
        private bool spreadOnTimer = true;

        [SerializeField]
        private Transform fireRoot;

        [SerializeField]
        private Vector2Int[] initialFireCells = new Vector2Int[0];

        [SerializeField]
        private Vector2Int[] blockedCells = new Vector2Int[0];

        [SerializeField]
        private Vector2Int[] protectedCells = new Vector2Int[0];

        [SerializeField]
        private List<RLHazardZone> activeFireZones = new List<RLHazardZone>();

        [SerializeField]
        private List<ScheduledIgnition> pendingIgnitions = new List<ScheduledIgnition>();

        private readonly List<Vector2Int> activeFireCells = new List<Vector2Int>();
        private readonly Vector2Int[] directions =
        {
            new Vector2Int(1, 0),
            new Vector2Int(-1, 0),
            new Vector2Int(0, 1),
            new Vector2Int(0, -1)
        };

        private System.Random random;
        private float elapsedSpreadTime;

        public int Seed { get { return seed; } }
        public IReadOnlyList<RLHazardZone> ActiveFireZones { get { return activeFireZones; } }
        public IReadOnlyList<Vector2Int> ActiveFireCells { get { return activeFireCells; } }
        public IReadOnlyList<ScheduledIgnition> PendingIgnitions { get { return pendingIgnitions; } }

        [Serializable]
        public struct ScheduledIgnition
        {
            [SerializeField]
            private Vector2Int cell;

            [SerializeField]
            private float ignitionTime;

            public ScheduledIgnition(Vector2Int scheduledCell, float scheduledIgnitionTime)
            {
                cell = scheduledCell;
                ignitionTime = scheduledIgnitionTime;
            }

            public Vector2Int Cell { get { return cell; } }
            public float IgnitionTime { get { return ignitionTime; } }
        }

        public void Configure(
            int deterministicSeed,
            Vector2Int generatedMapSize,
            float generatedCellSize,
            Transform generatedFireRoot,
            IEnumerable<Vector2Int> generatedInitialFireCells,
            IEnumerable<Vector2Int> generatedBlockedCells,
            IEnumerable<Vector2Int> generatedProtectedCells)
        {
            seed = deterministicSeed;
            mapSize = generatedMapSize;
            cellSize = Mathf.Max(0.25f, generatedCellSize);
            fireRoot = generatedFireRoot != null ? generatedFireRoot : transform;
            initialFireCells = RLTrainingGenerationUtility.CopySortedCells(generatedInitialFireCells);
            blockedCells = RLTrainingGenerationUtility.CopySortedCells(generatedBlockedCells);
            protectedCells = RLTrainingGenerationUtility.CopySortedCells(generatedProtectedCells);
            ResetDeterministic();
        }

        public void ResetDeterministic()
        {
            random = new System.Random(seed);
            elapsedSpreadTime = 0f;
            ClearGeneratedFireZones();
            activeFireZones.Clear();
            activeFireCells.Clear();
            pendingIgnitions.Clear();

            for (var i = 0; i < initialFireCells.Length; i++)
            {
                if (IsSpreadCandidate(initialFireCells[i]))
                {
                    AddFireZone(initialFireCells[i]);
                    ScheduleNeighborIgnitions(initialFireCells[i]);
                }
            }
        }

        public bool StepSpread()
        {
            while (pendingIgnitions.Count > 0)
            {
                var nextIgnition = pendingIgnitions[0];
                pendingIgnitions.RemoveAt(0);
                elapsedSpreadTime = Mathf.Max(elapsedSpreadTime, nextIgnition.IgnitionTime);

                if (!IsSpreadCandidate(nextIgnition.Cell))
                {
                    continue;
                }

                AddFireZone(nextIgnition.Cell);
                ScheduleNeighborIgnitions(nextIgnition.Cell);
                return true;
            }

            return false;
        }

        public int AdvanceSpread(float deltaTime)
        {
            if (deltaTime <= 0f)
            {
                return 0;
            }

            elapsedSpreadTime += deltaTime;
            var ignitedCount = 0;
            while (pendingIgnitions.Count > 0 && pendingIgnitions[0].IgnitionTime <= elapsedSpreadTime)
            {
                var nextIgnition = pendingIgnitions[0];
                pendingIgnitions.RemoveAt(0);
                if (!IsSpreadCandidate(nextIgnition.Cell))
                {
                    continue;
                }

                AddFireZone(nextIgnition.Cell);
                ScheduleNeighborIgnitions(nextIgnition.Cell);
                ignitedCount++;
            }

            return ignitedCount;
        }

        private void Update()
        {
            if (!spreadOnTimer || spreadInterval <= 0f)
            {
                return;
            }

            AdvanceSpread(Time.deltaTime);
        }

        private void ScheduleNeighborIgnitions(Vector2Int sourceCell)
        {
            if (random == null)
            {
                random = new System.Random(seed);
            }

            for (var directionIndex = 0; directionIndex < directions.Length; directionIndex++)
            {
                var candidate = sourceCell + directions[directionIndex];
                if (!IsSpreadCandidate(candidate) || IsPendingIgnition(candidate))
                {
                    continue;
                }

                var delay = Mathf.Lerp(minIgnitionDelay, maxIgnitionDelay, (float)random.NextDouble());
                InsertPendingIgnition(new ScheduledIgnition(candidate, elapsedSpreadTime + delay));
            }
        }

        private bool IsPendingIgnition(Vector2Int cell)
        {
            for (var i = 0; i < pendingIgnitions.Count; i++)
            {
                if (pendingIgnitions[i].Cell == cell)
                {
                    return true;
                }
            }

            return false;
        }

        private void InsertPendingIgnition(ScheduledIgnition ignition)
        {
            var insertIndex = pendingIgnitions.Count;
            for (var i = 0; i < pendingIgnitions.Count; i++)
            {
                var scheduled = pendingIgnitions[i];
                if (scheduled.IgnitionTime > ignition.IgnitionTime ||
                    (Mathf.Approximately(scheduled.IgnitionTime, ignition.IgnitionTime) &&
                    RLTrainingGenerationUtility.CompareCells(scheduled.Cell, ignition.Cell) > 0))
                {
                    insertIndex = i;
                    break;
                }
            }

            pendingIgnitions.Insert(insertIndex, ignition);
        }

        private bool IsSpreadCandidate(Vector2Int cell)
        {
            return IsInsideMap(cell) &&
                !RLTrainingGenerationUtility.ContainsCell(activeFireCells, cell) &&
                !RLTrainingGenerationUtility.ContainsCell(blockedCells, cell) &&
                !RLTrainingGenerationUtility.ContainsCell(protectedCells, cell);
        }

        private bool IsInsideMap(Vector2Int cell)
        {
            return cell.x >= 0 && cell.y >= 0 && cell.x < mapSize.x && cell.y < mapSize.y;
        }

        private void AddFireZone(Vector2Int cell)
        {
            var zoneObject = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            zoneObject.name = string.Format("SpreadingFire_{0}_{1}", cell.x, cell.y);
            zoneObject.transform.SetParent(fireRoot != null ? fireRoot : transform, false);
            zoneObject.transform.localPosition = RLTrainingGenerationUtility.CellToLocalPosition(cell, mapSize, cellSize, 0.06f);
            zoneObject.transform.localScale = new Vector3(cellSize * 0.7f, 0.06f, cellSize * 0.7f);

            var collider = zoneObject.GetComponent<Collider>();
            if (collider != null)
            {
                collider.isTrigger = true;
            }

            var hazardZone = zoneObject.AddComponent<RLHazardZone>();
            hazardZone.Initialize(RLHazardKind.Fire, cell, cellSize * 0.45f, true);
            RLFireVisuals.Apply(zoneObject, cellSize, seed + cell.x * 31 + cell.y * 97);
            activeFireZones.Add(hazardZone);
            RLTrainingGenerationUtility.AddIfMissing(activeFireCells, cell);
        }

        private void ClearGeneratedFireZones()
        {
            var root = fireRoot != null ? fireRoot : transform;
            for (var i = root.childCount - 1; i >= 0; i--)
            {
                RLTrainingGenerationUtility.DestroyUnityObject(root.GetChild(i).gameObject);
            }
        }

        private void OnValidate()
        {
            cellSize = Mathf.Max(0.25f, cellSize);
            spreadInterval = Mathf.Max(0.1f, spreadInterval);
            minIgnitionDelay = Mathf.Max(0.1f, minIgnitionDelay);
            maxIgnitionDelay = Mathf.Max(minIgnitionDelay, maxIgnitionDelay);
        }
    }
}
