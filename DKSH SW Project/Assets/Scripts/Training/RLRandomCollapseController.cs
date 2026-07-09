using System.Collections.Generic;
using UnityEngine;

namespace DKSH.Spiderbot.Training
{
    [DisallowMultipleComponent]
    public sealed class RLRandomCollapseController : MonoBehaviour
    {
        [SerializeField]
        private int seed;

        [SerializeField]
        private Vector2Int mapSize;

        [SerializeField, Min(0.25f)]
        private float cellSize = 1f;

        [SerializeField, Min(0)]
        private int collapseCount = 3;

        [SerializeField]
        private Transform collapseRoot;

        [SerializeField]
        private Vector2Int[] candidateCells = new Vector2Int[0];

        [SerializeField]
        private List<RLHazardZone> activeCollapseZones = new List<RLHazardZone>();

        private readonly List<Vector2Int> activeCollapseCells = new List<Vector2Int>();

        public int Seed { get { return seed; } }
        public IReadOnlyList<RLHazardZone> ActiveCollapseZones { get { return activeCollapseZones; } }
        public IReadOnlyList<Vector2Int> ActiveCollapseCells { get { return activeCollapseCells; } }

        public void Configure(
            int deterministicSeed,
            Vector2Int generatedMapSize,
            float generatedCellSize,
            Transform generatedCollapseRoot,
            int generatedCollapseCount,
            IEnumerable<Vector2Int> generatedCandidateCells)
        {
            seed = deterministicSeed;
            mapSize = generatedMapSize;
            cellSize = Mathf.Max(0.25f, generatedCellSize);
            collapseRoot = generatedCollapseRoot != null ? generatedCollapseRoot : transform;
            collapseCount = Mathf.Max(0, generatedCollapseCount);
            candidateCells = RLTrainingGenerationUtility.CopySortedCells(generatedCandidateCells);
            ResetDeterministic();
        }

        public void ResetDeterministic()
        {
            ClearGeneratedCollapseZones();
            activeCollapseZones.Clear();
            activeCollapseCells.Clear();

            if (collapseCount <= 0 || candidateCells.Length == 0)
            {
                return;
            }

            var random = new System.Random(seed);
            var pool = new List<Vector2Int>(candidateCells);
            var targetCount = Mathf.Min(collapseCount, pool.Count);
            for (var i = 0; i < targetCount; i++)
            {
                var chosenIndex = random.Next(pool.Count);
                var chosenCell = pool[chosenIndex];
                pool.RemoveAt(chosenIndex);
                AddCollapseZone(chosenCell);
            }
        }

        private void AddCollapseZone(Vector2Int cell)
        {
            var zoneObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
            zoneObject.name = string.Format("Collapse_{0}_{1}", cell.x, cell.y);
            zoneObject.transform.SetParent(collapseRoot != null ? collapseRoot : transform, false);
            zoneObject.transform.localPosition = RLTrainingGenerationUtility.CellToLocalPosition(cell, mapSize, cellSize, 0.45f);
            zoneObject.transform.localScale = new Vector3(cellSize * 0.85f, 0.9f, cellSize * 0.85f);

            var hazardZone = zoneObject.AddComponent<RLHazardZone>();
            hazardZone.Initialize(RLHazardKind.Collapse, cell, cellSize * 0.5f, true);
            activeCollapseZones.Add(hazardZone);
            RLTrainingGenerationUtility.AddIfMissing(activeCollapseCells, cell);
        }

        private void ClearGeneratedCollapseZones()
        {
            var root = collapseRoot != null ? collapseRoot : transform;
            for (var i = root.childCount - 1; i >= 0; i--)
            {
                RLTrainingGenerationUtility.DestroyUnityObject(root.GetChild(i).gameObject);
            }
        }

        private void OnValidate()
        {
            cellSize = Mathf.Max(0.25f, cellSize);
            collapseCount = Mathf.Max(0, collapseCount);
        }
    }
}
