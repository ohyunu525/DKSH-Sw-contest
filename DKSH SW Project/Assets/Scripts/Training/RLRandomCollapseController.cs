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

        [SerializeField, Min(0.5f)]
        private float dropHeight = 9f;

        [SerializeField, Min(0f)]
        private float minReleaseTime = 1.5f;

        [SerializeField, Min(0f)]
        private float maxReleaseTime = 10f;

        [SerializeField]
        private Transform collapseRoot;

        [SerializeField]
        private Vector2Int[] candidateCells = new Vector2Int[0];

        [SerializeField]
        private List<RLHazardZone> activeCollapseZones = new List<RLHazardZone>();

        [SerializeField]
        private List<RLDebrisHazard> scheduledDebris = new List<RLDebrisHazard>();

        private readonly List<Vector2Int> activeCollapseCells = new List<Vector2Int>();
        private float elapsedTime;

        public int Seed { get { return seed; } }
        public IReadOnlyList<RLHazardZone> ActiveCollapseZones { get { return activeCollapseZones; } }
        public IReadOnlyList<Vector2Int> ActiveCollapseCells { get { return activeCollapseCells; } }
        public IReadOnlyList<RLDebrisHazard> ScheduledDebris { get { return scheduledDebris; } }

        public void Configure(
            int deterministicSeed,
            Vector2Int generatedMapSize,
            float generatedCellSize,
            Transform generatedCollapseRoot,
            int generatedCollapseCount,
            IEnumerable<Vector2Int> generatedCandidateCells,
            float generatedDropHeight)
        {
            seed = deterministicSeed;
            mapSize = generatedMapSize;
            cellSize = Mathf.Max(0.25f, generatedCellSize);
            collapseRoot = generatedCollapseRoot != null ? generatedCollapseRoot : transform;
            collapseCount = Mathf.Max(0, generatedCollapseCount);
            dropHeight = Mathf.Max(0.5f, generatedDropHeight);
            candidateCells = RLTrainingGenerationUtility.CopySortedCells(generatedCandidateCells);
            ResetDeterministic();
        }

        public void ResetDeterministic()
        {
            ClearGeneratedCollapseZones();
            activeCollapseZones.Clear();
            activeCollapseCells.Clear();
            scheduledDebris.Clear();
            elapsedTime = 0f;

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
                var releaseTime = UnityRandomRange(random, minReleaseTime, Mathf.Max(minReleaseTime, maxReleaseTime));
                AddDebris(chosenCell, releaseTime, i, random);
            }
        }

        private void Update()
        {
            elapsedTime += Time.deltaTime;
            for (var i = 0; i < scheduledDebris.Count; i++)
            {
                if (scheduledDebris[i] != null)
                {
                    scheduledDebris[i].TickRelease(elapsedTime);
                }
            }
        }

        private void AddDebris(Vector2Int cell, float releaseTime, int debrisIndex, System.Random random)
        {
            var zoneObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
            zoneObject.name = string.Format("FallingDebris_{0}_{1}_{2}", debrisIndex, cell.x, cell.y);
            zoneObject.transform.SetParent(collapseRoot != null ? collapseRoot : transform, false);
            var localPosition = RLTrainingGenerationUtility.CellToLocalPosition(
                cell,
                mapSize,
                cellSize,
                dropHeight + UnityRandomRange(random, 0f, 2.5f));
            var localRotation = Quaternion.Euler(
                UnityRandomRange(random, -25f, 25f),
                UnityRandomRange(random, 0f, 360f),
                UnityRandomRange(random, -25f, 25f));
            var debrisScale = UnityRandomRange(random, cellSize * 0.35f, cellSize * 0.75f);
            var localScale = new Vector3(
                debrisScale,
                UnityRandomRange(random, cellSize * 0.25f, cellSize * 0.65f),
                debrisScale);

            zoneObject.transform.localPosition = localPosition;
            zoneObject.transform.localRotation = localRotation;
            zoneObject.transform.localScale = localScale;

            var body = zoneObject.AddComponent<Rigidbody>();
            body.mass = Mathf.Max(0.2f, debrisScale);
            body.isKinematic = true;
            body.useGravity = false;

            var hazardZone = zoneObject.AddComponent<RLHazardZone>();
            hazardZone.Initialize(RLHazardKind.Collapse, cell, cellSize * 0.5f, true);
            var debris = zoneObject.AddComponent<RLDebrisHazard>();
            debris.Configure(cell, releaseTime, localPosition, localRotation, localScale);

            activeCollapseZones.Add(hazardZone);
            scheduledDebris.Add(debris);
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
            dropHeight = Mathf.Max(0.5f, dropHeight);
            minReleaseTime = Mathf.Max(0f, minReleaseTime);
            maxReleaseTime = Mathf.Max(minReleaseTime, maxReleaseTime);
        }

        private static float UnityRandomRange(System.Random random, float minInclusive, float maxInclusive)
        {
            return Mathf.Lerp(minInclusive, maxInclusive, (float)random.NextDouble());
        }
    }
}
