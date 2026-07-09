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
        private float spreadInterval = 5f;

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

        private readonly List<Vector2Int> activeFireCells = new List<Vector2Int>();
        private readonly Vector2Int[] directions =
        {
            new Vector2Int(1, 0),
            new Vector2Int(-1, 0),
            new Vector2Int(0, 1),
            new Vector2Int(0, -1)
        };

        private System.Random random;
        private float spreadTimer;

        public int Seed { get { return seed; } }
        public IReadOnlyList<RLHazardZone> ActiveFireZones { get { return activeFireZones; } }
        public IReadOnlyList<Vector2Int> ActiveFireCells { get { return activeFireCells; } }

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
            spreadTimer = 0f;
            ClearGeneratedFireZones();
            activeFireZones.Clear();
            activeFireCells.Clear();

            for (var i = 0; i < initialFireCells.Length; i++)
            {
                if (IsSpreadCandidate(initialFireCells[i]))
                {
                    AddFireZone(initialFireCells[i]);
                }
            }
        }

        public bool StepSpread()
        {
            if (activeFireCells.Count == 0)
            {
                return false;
            }

            if (random == null)
            {
                random = new System.Random(seed);
            }

            var candidates = new List<Vector2Int>();
            for (var i = 0; i < activeFireCells.Count; i++)
            {
                var fireCell = activeFireCells[i];
                for (var directionIndex = 0; directionIndex < directions.Length; directionIndex++)
                {
                    var candidate = fireCell + directions[directionIndex];
                    if (IsSpreadCandidate(candidate))
                    {
                        RLTrainingGenerationUtility.AddIfMissing(candidates, candidate);
                    }
                }
            }

            if (candidates.Count == 0)
            {
                return false;
            }

            var chosenCell = candidates[random.Next(candidates.Count)];
            AddFireZone(chosenCell);
            return true;
        }

        private void Update()
        {
            if (!spreadOnTimer || spreadInterval <= 0f)
            {
                return;
            }

            spreadTimer += Time.deltaTime;
            while (spreadTimer >= spreadInterval)
            {
                spreadTimer -= spreadInterval;
                if (!StepSpread())
                {
                    break;
                }
            }
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
        }
    }
}
