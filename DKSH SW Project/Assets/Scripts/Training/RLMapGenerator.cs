using System;
using System.Collections.Generic;
using UnityEngine;

namespace DKSH.Spiderbot.Training
{
    [DisallowMultipleComponent]
    public sealed class RLMapGenerator : MonoBehaviour
    {
        private const string GeneratedEnvironmentPrefix = "Environment_";
        private const int RetrySeedOffset = 1000003;

        [Header("Generation")]
        [SerializeField]
        private RLMapLevel selectedLevel = RLMapLevel.Flat;

        [SerializeField, Min(1)]
        private int mapCount = 1;

        [SerializeField]
        private int baseSeed = 1000;

        [SerializeField, Min(1f)]
        private float spacing = 30f;

        [SerializeField]
        private bool regenerateOnPlay = true;

        [SerializeField]
        private Transform generatedEnvironmentsParent;

        [Header("Map Shape")]
        [SerializeField, Min(6)]
        private int mapWidth = 12;

        [SerializeField, Min(6)]
        private int mapDepth = 12;

        [SerializeField, Min(0.25f)]
        private float cellSize = 1f;

        [SerializeField, Min(1)]
        private int maxGenerationAttempts = 8;

        [Header("Generated References")]
        [SerializeField]
        private List<GeneratedTrainingEnvironment> generatedEnvironments = new List<GeneratedTrainingEnvironment>();

        private static readonly Vector2Int[] PathDirections =
        {
            new Vector2Int(1, 0),
            new Vector2Int(-1, 0),
            new Vector2Int(0, 1),
            new Vector2Int(0, -1)
        };

        public RLMapLevel SelectedLevel
        {
            get { return selectedLevel; }
            set { selectedLevel = value; }
        }

        public int MapCount
        {
            get { return mapCount; }
            set { mapCount = Mathf.Max(1, value); }
        }

        public int BaseSeed
        {
            get { return baseSeed; }
            set { baseSeed = value; }
        }

        public float Spacing
        {
            get { return spacing; }
            set { spacing = Mathf.Max(1f, value); }
        }

        public bool RegenerateOnPlay
        {
            get { return regenerateOnPlay; }
            set { regenerateOnPlay = value; }
        }

        public Transform GeneratedEnvironmentsParent
        {
            get { return generatedEnvironmentsParent; }
            set { generatedEnvironmentsParent = value; }
        }

        public int MapWidth
        {
            get { return mapWidth; }
            set { mapWidth = Mathf.Max(6, value); }
        }

        public int MapDepth
        {
            get { return mapDepth; }
            set { mapDepth = Mathf.Max(6, value); }
        }

        public float CellSize
        {
            get { return cellSize; }
            set { cellSize = Mathf.Max(0.25f, value); }
        }

        public IReadOnlyList<GeneratedTrainingEnvironment> Environments
        {
            get { return generatedEnvironments; }
        }

        public GeneratedTrainingEnvironment GetEnvironment(int index)
        {
            if (index < 0 || index >= generatedEnvironments.Count)
            {
                return null;
            }

            return generatedEnvironments[index];
        }

        [ContextMenu("Generate RL Maps")]
        public void Generate()
        {
            ClampSettings();
            ClearGeneratedEnvironments();

            var parent = ResolveParent();
            var columns = Mathf.Max(1, Mathf.CeilToInt(Mathf.Sqrt(mapCount)));

            for (var index = 0; index < mapCount; index++)
            {
                GeneratedTrainingEnvironment environment;
                if (TryGenerateEnvironment(index, parent, columns, out environment))
                {
                    generatedEnvironments.Add(environment);
                }
            }
        }

        [ContextMenu("Clear Generated RL Maps")]
        public void ClearGeneratedEnvironments()
        {
            var targets = new List<GameObject>();
            for (var i = 0; i < generatedEnvironments.Count; i++)
            {
                if (generatedEnvironments[i] != null)
                {
                    AddIfMissing(targets, generatedEnvironments[i].gameObject);
                }
            }

            var parent = ResolveParent();
            for (var i = parent.childCount - 1; i >= 0; i--)
            {
                var child = parent.GetChild(i);
                if (!child.name.StartsWith(GeneratedEnvironmentPrefix, StringComparison.Ordinal))
                {
                    continue;
                }

                var environment = child.GetComponent<GeneratedTrainingEnvironment>();
                if (environment != null && (environment.Generator == null || environment.Generator == this))
                {
                    AddIfMissing(targets, child.gameObject);
                }
            }

            generatedEnvironments.Clear();
            for (var i = 0; i < targets.Count; i++)
            {
                RLTrainingGenerationUtility.DestroyUnityObject(targets[i]);
            }
        }

        private void Start()
        {
            if (regenerateOnPlay)
            {
                Generate();
            }
        }

        private bool TryGenerateEnvironment(int index, Transform parent, int columns, out GeneratedTrainingEnvironment environment)
        {
            var mapSeed = unchecked(baseSeed + index);
            for (var attempt = 0; attempt < maxGenerationAttempts; attempt++)
            {
                var attemptSeed = GetAttemptSeed(mapSeed, attempt);
                var context = CreateContext(index, attemptSeed, selectedLevel, parent, columns);
                var generated = GenerateSelectedLevel(context);

                if (generated && ValidateGeneratedMap(context))
                {
                    environment = FinalizeEnvironment(context);
                    return true;
                }

                Debug.LogWarning(
                    string.Format(
                        "RLMapGenerator failed to generate valid map index {0} level {1} with seed {2} on attempt {3}. Retrying.",
                        index,
                        selectedLevel,
                        attemptSeed,
                        attempt + 1),
                    this);
                RLTrainingGenerationUtility.DestroyUnityObject(context.RootObject);
            }

            Debug.LogError(
                string.Format(
                    "RLMapGenerator failed after {0} attempts for map index {1}. Creating a flat fallback map.",
                    maxGenerationAttempts,
                    index),
                this);

            var fallbackContext = CreateContext(index, mapSeed, RLMapLevel.Flat, parent, columns);
            GenerateFlatMap(fallbackContext);
            environment = FinalizeEnvironment(fallbackContext);
            return true;
        }

        private MapBuildContext CreateContext(int index, int seed, RLMapLevel level, Transform parent, int columns)
        {
            var mapSize = new Vector2Int(mapWidth, mapDepth);
            var rootObject = new GameObject(GetEnvironmentName(index, level, seed));
            rootObject.transform.SetParent(parent, false);
            rootObject.transform.localPosition = GetGridPosition(index, columns);

            var context = new MapBuildContext
            {
                Generator = this,
                Index = index,
                Seed = seed,
                Level = level,
                Random = new System.Random(seed),
                MapSize = mapSize,
                CellSize = cellSize,
                RootObject = rootObject,
                Root = rootObject.transform,
                Environment = rootObject.AddComponent<GeneratedTrainingEnvironment>(),
                ObstacleCells = new bool[mapSize.x, mapSize.y],
                HazardCells = new bool[mapSize.x, mapSize.y],
                CollapseCells = new bool[mapSize.x, mapSize.y]
            };

            context.GeometryRoot = CreateChild("Geometry", context.Root);
            context.ObstaclesRoot = CreateChild("Obstacles", context.Root);
            context.HazardsRoot = CreateChild("Hazards", context.Root);
            context.MarkersRoot = CreateChild("Markers", context.Root);
            return context;
        }

        private bool GenerateSelectedLevel(MapBuildContext context)
        {
            switch (context.Level)
            {
                case RLMapLevel.Flat:
                    return GenerateFlatMap(context);
                case RLMapLevel.SeededStair:
                    return GenerateStairMap(context);
                case RLMapLevel.SeededHill:
                    return GenerateHillMap(context);
                case RLMapLevel.SeededBuilding:
                    return GenerateBuildingMap(context);
                case RLMapLevel.BuildingFixedFire:
                    return GenerateBuildingMap(context) && AddFixedFire(context);
                case RLMapLevel.BuildingSpreadingFire:
                    return GenerateBuildingMap(context) && AddSpreadingFire(context);
                case RLMapLevel.BuildingSpreadingFireRandomCollapse:
                    return GenerateBuildingMap(context) && AddRandomCollapse(context) && AddSpreadingFire(context);
                default:
                    Debug.LogWarning(string.Format("Unsupported RL map level {0}. Falling back to flat map.", context.Level), this);
                    return GenerateFlatMap(context);
            }
        }

        private bool GenerateFlatMap(MapBuildContext context)
        {
            CreateFloor(context);
            SetStartAndTarget(
                context,
                new Vector2Int(1, 1),
                new Vector2Int(context.MapSize.x - 2, context.MapSize.y - 2));
            return true;
        }

        private bool GenerateStairMap(MapBuildContext context)
        {
            GenerateFlatMap(context);

            var horizontal = context.Random.Next(0, 2) == 0;
            var stepCount = context.Random.Next(4, Mathf.Max(5, Mathf.Min(context.MapSize.x, context.MapSize.y) - 2));
            var laneOffset = context.Random.Next(-1, 2);
            var centerX = context.MapSize.x / 2;
            var centerY = context.MapSize.y / 2;

            for (var i = 0; i < stepCount; i++)
            {
                var cell = horizontal
                    ? new Vector2Int(Mathf.Clamp(2 + i, 1, context.MapSize.x - 2), Mathf.Clamp(centerY + laneOffset, 1, context.MapSize.y - 2))
                    : new Vector2Int(Mathf.Clamp(centerX + laneOffset, 1, context.MapSize.x - 2), Mathf.Clamp(2 + i, 1, context.MapSize.y - 2));

                var height = 0.18f * (i + 1);
                var localPosition = RLTrainingGenerationUtility.CellToLocalPosition(cell, context.MapSize, context.CellSize, height * 0.5f);
                var localScale = horizontal
                    ? new Vector3(context.CellSize, height, context.CellSize * 2.5f)
                    : new Vector3(context.CellSize * 2.5f, height, context.CellSize);
                CreatePrimitiveBlock("StairStep_" + i, context.GeometryRoot, localPosition, localScale);
            }

            return true;
        }

        private bool GenerateHillMap(MapBuildContext context)
        {
            GenerateFlatMap(context);

            var usedCells = new List<Vector2Int>();
            var hillCount = context.Random.Next(4, 7);
            var attempts = hillCount * 8;

            while (usedCells.Count < hillCount && attempts > 0)
            {
                attempts--;
                var cell = new Vector2Int(
                    context.Random.Next(2, context.MapSize.x - 2),
                    context.Random.Next(2, context.MapSize.y - 2));

                if (context.IsStartOrTarget(cell) || RLTrainingGenerationUtility.ContainsCell(usedCells, cell))
                {
                    continue;
                }

                usedCells.Add(cell);
                var radius = context.CellSize * UnityRandomRange(context.Random, 1.25f, 2.2f);
                var height = UnityRandomRange(context.Random, 0.35f, 0.85f);
                var hill = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                hill.name = string.Format("Hill_{0}_{1}", cell.x, cell.y);
                hill.transform.SetParent(context.GeometryRoot, false);
                hill.transform.localPosition = RLTrainingGenerationUtility.CellToLocalPosition(cell, context.MapSize, context.CellSize, height * 0.35f);
                hill.transform.localScale = new Vector3(radius, height, radius);
            }

            return true;
        }

        private bool GenerateBuildingMap(MapBuildContext context)
        {
            CreateFloor(context);

            if (!TryChooseStartAndTarget(context))
            {
                return false;
            }

            var pathCells = CreateProtectedPath(context);
            for (var i = 0; i < pathCells.Count; i++)
            {
                context.ProtectedCells.Add(pathCells[i]);
            }

            AddPerimeterWalls(context);
            AddRandomBuildingObstacles(context);
            CreateObstacleObjects(context);
            return true;
        }

        private bool AddFixedFire(MapBuildContext context)
        {
            var fireCount = Mathf.Max(1, Mathf.RoundToInt(context.MapSize.x * context.MapSize.y / 80f));
            var fireCells = ChooseAvailableCells(context, fireCount, true);
            if (fireCells.Count == 0)
            {
                Debug.LogWarning("RLMapGenerator could not place fixed fire cells.", this);
                return false;
            }

            for (var i = 0; i < fireCells.Count; i++)
            {
                AddFireZone(context, fireCells[i], "FixedFire");
            }

            return true;
        }

        private bool AddSpreadingFire(MapBuildContext context)
        {
            var initialFireCells = ChooseAvailableCells(context, 1, true);
            if (initialFireCells.Count == 0)
            {
                Debug.LogWarning("RLMapGenerator could not place spreading fire seed cells.", this);
                return false;
            }

            var controllerObject = new GameObject("SpreadingFireController");
            controllerObject.transform.SetParent(context.HazardsRoot, false);
            var fireRoot = CreateChild("SpreadingFireZones", controllerObject.transform);
            var controller = controllerObject.AddComponent<RLSpreadingFireController>();
            var blockedCells = GetCombinedCells(context.ObstacleCells, context.CollapseCells);
            var protectedCells = RLTrainingGenerationUtility.CopySortedCells(context.ProtectedCells);
            controller.Configure(
                GetDerivedSeed(context.Seed, 401),
                context.MapSize,
                context.CellSize,
                fireRoot,
                initialFireCells,
                blockedCells,
                protectedCells);

            for (var i = 0; i < controller.ActiveFireCells.Count; i++)
            {
                MarkHazard(context, controller.ActiveFireCells[i]);
            }

            return true;
        }

        private bool AddRandomCollapse(MapBuildContext context)
        {
            var candidateCells = GetAvailableCells(context, true);
            if (candidateCells.Count == 0)
            {
                Debug.LogWarning("RLMapGenerator could not find random collapse candidate cells.", this);
                return false;
            }

            var collapseCount = Mathf.Max(1, Mathf.RoundToInt(context.MapSize.x * context.MapSize.y / 60f));
            var controllerObject = new GameObject("RandomCollapseController");
            controllerObject.transform.SetParent(context.HazardsRoot, false);
            var collapseRoot = CreateChild("CollapseZones", controllerObject.transform);
            var controller = controllerObject.AddComponent<RLRandomCollapseController>();
            controller.Configure(
                GetDerivedSeed(context.Seed, 809),
                context.MapSize,
                context.CellSize,
                collapseRoot,
                collapseCount,
                candidateCells);

            for (var i = 0; i < controller.ActiveCollapseCells.Count; i++)
            {
                MarkCollapse(context, controller.ActiveCollapseCells[i]);
            }

            return true;
        }

        private bool ValidateGeneratedMap(MapBuildContext context)
        {
            if (!context.HasStartAndTarget)
            {
                Debug.LogWarning("RLMapGenerator validation failed because start/target was not assigned.", this);
                return false;
            }

            if (!context.IsInside(context.StartCell) || !context.IsInside(context.TargetCell))
            {
                Debug.LogWarning("RLMapGenerator validation failed because start/target is outside the map.", this);
                return false;
            }

            if (context.IsUnsafe(context.StartCell) || context.IsUnsafe(context.TargetCell))
            {
                Debug.LogWarning("RLMapGenerator validation failed because start/target overlaps blocked or hazard cells.", this);
                return false;
            }

            if (!HasPath(context, context.StartCell, context.TargetCell))
            {
                Debug.LogWarning("RLMapGenerator validation failed because no path exists from start to target.", this);
                return false;
            }

            return true;
        }

        private GeneratedTrainingEnvironment FinalizeEnvironment(MapBuildContext context)
        {
            var startPoint = CreateStartMarker(context);
            var targetPoint = CreateTargetMarker(context);
            var obstacleCells = GridToCells(context.ObstacleCells);
            var hazardCells = GridToCells(context.HazardCells);
            var collapseCells = GridToCells(context.CollapseCells);

            context.Environment.Initialize(
                this,
                context.Index,
                context.Seed,
                context.Level,
                context.Root,
                context.GeometryRoot,
                context.ObstaclesRoot,
                context.HazardsRoot,
                startPoint,
                targetPoint,
                context.MapSize,
                context.CellSize,
                context.StartCell,
                context.TargetCell,
                obstacleCells,
                hazardCells,
                collapseCells);

            return context.Environment;
        }

        private void CreateFloor(MapBuildContext context)
        {
            var floor = GameObject.CreatePrimitive(PrimitiveType.Cube);
            floor.name = "Floor";
            floor.transform.SetParent(context.GeometryRoot, false);
            floor.transform.localPosition = new Vector3(0f, -0.05f, 0f);
            floor.transform.localScale = new Vector3(context.MapSize.x * context.CellSize, 0.1f, context.MapSize.y * context.CellSize);
        }

        private bool TryChooseStartAndTarget(MapBuildContext context)
        {
            var minDistance = Mathf.Max(4, (context.MapSize.x + context.MapSize.y) / 3);
            for (var attempt = 0; attempt < 128; attempt++)
            {
                var start = new Vector2Int(
                    context.Random.Next(1, context.MapSize.x - 1),
                    context.Random.Next(1, context.MapSize.y - 1));
                var target = new Vector2Int(
                    context.Random.Next(1, context.MapSize.x - 1),
                    context.Random.Next(1, context.MapSize.y - 1));

                if (start == target)
                {
                    continue;
                }

                var distance = Mathf.Abs(start.x - target.x) + Mathf.Abs(start.y - target.y);
                if (distance < minDistance)
                {
                    continue;
                }

                SetStartAndTarget(context, start, target);
                return true;
            }

            SetStartAndTarget(
                context,
                new Vector2Int(1, 1),
                new Vector2Int(context.MapSize.x - 2, context.MapSize.y - 2));
            return true;
        }

        private void SetStartAndTarget(MapBuildContext context, Vector2Int start, Vector2Int target)
        {
            context.StartCell = start;
            context.TargetCell = target;
            context.HasStartAndTarget = true;
            context.ProtectedCells.Add(start);
            context.ProtectedCells.Add(target);
        }

        private List<Vector2Int> CreateProtectedPath(MapBuildContext context)
        {
            var pathCells = new List<Vector2Int>();
            var current = context.StartCell;
            pathCells.Add(current);

            var guard = context.MapSize.x * context.MapSize.y * 4;
            while (current != context.TargetCell && guard > 0)
            {
                guard--;
                var canMoveX = current.x != context.TargetCell.x;
                var canMoveY = current.y != context.TargetCell.y;
                var moveX = canMoveX && (!canMoveY || context.Random.Next(0, 2) == 0);

                if (moveX)
                {
                    current.x += Math.Sign(context.TargetCell.x - current.x);
                }
                else
                {
                    current.y += Math.Sign(context.TargetCell.y - current.y);
                }

                RLTrainingGenerationUtility.AddIfMissing(pathCells, current);
            }

            return pathCells;
        }

        private void AddPerimeterWalls(MapBuildContext context)
        {
            for (var x = 0; x < context.MapSize.x; x++)
            {
                MarkObstacle(context, new Vector2Int(x, 0));
                MarkObstacle(context, new Vector2Int(x, context.MapSize.y - 1));
            }

            for (var y = 0; y < context.MapSize.y; y++)
            {
                MarkObstacle(context, new Vector2Int(0, y));
                MarkObstacle(context, new Vector2Int(context.MapSize.x - 1, y));
            }
        }

        private void AddRandomBuildingObstacles(MapBuildContext context)
        {
            var obstacleTarget = Mathf.RoundToInt(context.MapSize.x * context.MapSize.y * 0.22f);
            var placed = 0;
            var attempts = obstacleTarget * 8;

            while (placed < obstacleTarget && attempts > 0)
            {
                attempts--;
                var cell = new Vector2Int(
                    context.Random.Next(1, context.MapSize.x - 1),
                    context.Random.Next(1, context.MapSize.y - 1));

                if (!CanPlaceBlockingCell(context, cell, true))
                {
                    continue;
                }

                MarkObstacle(context, cell);
                placed++;
            }
        }

        private void CreateObstacleObjects(MapBuildContext context)
        {
            for (var y = 0; y < context.MapSize.y; y++)
            {
                for (var x = 0; x < context.MapSize.x; x++)
                {
                    if (!context.ObstacleCells[x, y])
                    {
                        continue;
                    }

                    var cell = new Vector2Int(x, y);
                    var localPosition = RLTrainingGenerationUtility.CellToLocalPosition(cell, context.MapSize, context.CellSize, 1.1f);
                    var localScale = new Vector3(context.CellSize * 0.95f, 2.2f, context.CellSize * 0.95f);
                    CreatePrimitiveBlock(string.Format("Obstacle_{0}_{1}", x, y), context.ObstaclesRoot, localPosition, localScale);
                }
            }
        }

        private bool CanPlaceBlockingCell(MapBuildContext context, Vector2Int cell, bool avoidProtectedPath)
        {
            return context.IsInside(cell) &&
                !context.IsStartOrTarget(cell) &&
                !context.IsUnsafe(cell) &&
                (!avoidProtectedPath || !context.ProtectedCells.Contains(cell));
        }

        private List<Vector2Int> ChooseAvailableCells(MapBuildContext context, int count, bool avoidProtectedPath)
        {
            var candidates = GetAvailableCells(context, avoidProtectedPath);
            var chosenCells = new List<Vector2Int>();
            while (chosenCells.Count < count && candidates.Count > 0)
            {
                var chosenIndex = context.Random.Next(candidates.Count);
                var chosenCell = candidates[chosenIndex];
                candidates.RemoveAt(chosenIndex);
                chosenCells.Add(chosenCell);
            }

            if (chosenCells.Count < count)
            {
                Debug.LogWarning(
                    string.Format("RLMapGenerator placed {0}/{1} requested cells for level {2}.", chosenCells.Count, count, context.Level),
                    this);
            }

            return chosenCells;
        }

        private List<Vector2Int> GetAvailableCells(MapBuildContext context, bool avoidProtectedPath)
        {
            var candidates = new List<Vector2Int>();
            for (var y = 1; y < context.MapSize.y - 1; y++)
            {
                for (var x = 1; x < context.MapSize.x - 1; x++)
                {
                    var cell = new Vector2Int(x, y);
                    if (CanPlaceBlockingCell(context, cell, avoidProtectedPath))
                    {
                        candidates.Add(cell);
                    }
                }
            }

            return candidates;
        }

        private void AddFireZone(MapBuildContext context, Vector2Int cell, string prefix)
        {
            var fireObject = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            fireObject.name = string.Format("{0}_{1}_{2}", prefix, cell.x, cell.y);
            fireObject.transform.SetParent(context.HazardsRoot, false);
            fireObject.transform.localPosition = RLTrainingGenerationUtility.CellToLocalPosition(cell, context.MapSize, context.CellSize, 0.06f);
            fireObject.transform.localScale = new Vector3(context.CellSize * 0.7f, 0.06f, context.CellSize * 0.7f);

            var collider = fireObject.GetComponent<Collider>();
            if (collider != null)
            {
                collider.isTrigger = true;
            }

            var hazard = fireObject.AddComponent<RLHazardZone>();
            hazard.Initialize(RLHazardKind.Fire, cell, context.CellSize * 0.45f, true);
            MarkHazard(context, cell);
        }

        private Transform CreateStartMarker(MapBuildContext context)
        {
            var marker = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            marker.name = "StartPosition";
            marker.transform.SetParent(context.MarkersRoot, false);
            marker.transform.localPosition = RLTrainingGenerationUtility.CellToLocalPosition(context.StartCell, context.MapSize, context.CellSize, 0.25f);
            marker.transform.localScale = Vector3.one * Mathf.Max(0.25f, context.CellSize * 0.35f);

            var collider = marker.GetComponent<Collider>();
            if (collider != null)
            {
                RLTrainingGenerationUtility.DestroyUnityObject(collider);
            }

            return marker.transform;
        }

        private Transform CreateTargetMarker(MapBuildContext context)
        {
            var marker = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            marker.name = "TargetPosition";
            marker.transform.SetParent(context.MarkersRoot, false);
            marker.transform.localPosition = RLTrainingGenerationUtility.CellToLocalPosition(context.TargetCell, context.MapSize, context.CellSize, 0.06f);
            marker.transform.localScale = new Vector3(context.CellSize * 0.55f, 0.06f, context.CellSize * 0.55f);

            var collider = marker.GetComponent<Collider>();
            if (collider != null)
            {
                collider.isTrigger = true;
            }

            return marker.transform;
        }

        private bool HasPath(MapBuildContext context, Vector2Int start, Vector2Int target)
        {
            var visited = new bool[context.MapSize.x, context.MapSize.y];
            var queue = new Queue<Vector2Int>();
            visited[start.x, start.y] = true;
            queue.Enqueue(start);

            while (queue.Count > 0)
            {
                var current = queue.Dequeue();
                if (current == target)
                {
                    return true;
                }

                for (var i = 0; i < PathDirections.Length; i++)
                {
                    var next = current + PathDirections[i];
                    if (!context.IsInside(next) || visited[next.x, next.y] || context.IsUnsafe(next))
                    {
                        continue;
                    }

                    visited[next.x, next.y] = true;
                    queue.Enqueue(next);
                }
            }

            return false;
        }

        private void MarkObstacle(MapBuildContext context, Vector2Int cell)
        {
            if (context.IsInside(cell))
            {
                context.ObstacleCells[cell.x, cell.y] = true;
            }
        }

        private void MarkHazard(MapBuildContext context, Vector2Int cell)
        {
            if (context.IsInside(cell))
            {
                context.HazardCells[cell.x, cell.y] = true;
            }
        }

        private void MarkCollapse(MapBuildContext context, Vector2Int cell)
        {
            if (context.IsInside(cell))
            {
                context.CollapseCells[cell.x, cell.y] = true;
            }
        }

        private Vector2Int[] GridToCells(bool[,] grid)
        {
            var cells = new List<Vector2Int>();
            var width = grid.GetLength(0);
            var height = grid.GetLength(1);
            for (var y = 0; y < height; y++)
            {
                for (var x = 0; x < width; x++)
                {
                    if (grid[x, y])
                    {
                        cells.Add(new Vector2Int(x, y));
                    }
                }
            }

            return cells.ToArray();
        }

        private Vector2Int[] GetCombinedCells(bool[,] firstGrid, bool[,] secondGrid)
        {
            var cells = new List<Vector2Int>();
            AppendGridCells(cells, firstGrid);
            AppendGridCells(cells, secondGrid);
            cells.Sort(RLTrainingGenerationUtility.CompareCells);
            return cells.ToArray();
        }

        private static void AppendGridCells(List<Vector2Int> cells, bool[,] grid)
        {
            var width = grid.GetLength(0);
            var height = grid.GetLength(1);
            for (var y = 0; y < height; y++)
            {
                for (var x = 0; x < width; x++)
                {
                    if (grid[x, y])
                    {
                        RLTrainingGenerationUtility.AddIfMissing(cells, new Vector2Int(x, y));
                    }
                }
            }
        }

        private Vector3 GetGridPosition(int index, int columns)
        {
            var column = index % columns;
            var row = index / columns;
            return new Vector3(column * spacing, 0f, row * spacing);
        }

        private Transform ResolveParent()
        {
            return generatedEnvironmentsParent != null ? generatedEnvironmentsParent : transform;
        }

        private static string GetEnvironmentName(int index, RLMapLevel level, int seed)
        {
            return string.Format("{0}{1:000}_{2}_Seed{3}", GeneratedEnvironmentPrefix, index, level, seed);
        }

        private static int GetAttemptSeed(int mapSeed, int attempt)
        {
            unchecked
            {
                return mapSeed + attempt * RetrySeedOffset;
            }
        }

        private static int GetDerivedSeed(int seed, int salt)
        {
            unchecked
            {
                return seed * 397 + salt;
            }
        }

        private static Transform CreateChild(string name, Transform parent)
        {
            var child = new GameObject(name);
            child.transform.SetParent(parent, false);
            return child.transform;
        }

        private static GameObject CreatePrimitiveBlock(string name, Transform parent, Vector3 localPosition, Vector3 localScale)
        {
            var block = GameObject.CreatePrimitive(PrimitiveType.Cube);
            block.name = name;
            block.transform.SetParent(parent, false);
            block.transform.localPosition = localPosition;
            block.transform.localScale = localScale;
            return block;
        }

        private static float UnityRandomRange(System.Random random, float minInclusive, float maxInclusive)
        {
            return Mathf.Lerp(minInclusive, maxInclusive, (float)random.NextDouble());
        }

        private static void AddIfMissing(List<GameObject> objects, GameObject target)
        {
            if (target == null || objects.Contains(target))
            {
                return;
            }

            objects.Add(target);
        }

        private void ClampSettings()
        {
            mapCount = Mathf.Max(1, mapCount);
            spacing = Mathf.Max(1f, spacing);
            mapWidth = Mathf.Max(6, mapWidth);
            mapDepth = Mathf.Max(6, mapDepth);
            cellSize = Mathf.Max(0.25f, cellSize);
            maxGenerationAttempts = Mathf.Max(1, maxGenerationAttempts);
        }

        private void OnValidate()
        {
            ClampSettings();
        }

        private sealed class MapBuildContext
        {
            public RLMapGenerator Generator;
            public int Index;
            public int Seed;
            public RLMapLevel Level;
            public System.Random Random;
            public Vector2Int MapSize;
            public float CellSize;
            public GameObject RootObject;
            public Transform Root;
            public Transform GeometryRoot;
            public Transform ObstaclesRoot;
            public Transform HazardsRoot;
            public Transform MarkersRoot;
            public GeneratedTrainingEnvironment Environment;
            public bool[,] ObstacleCells;
            public bool[,] HazardCells;
            public bool[,] CollapseCells;
            public Vector2Int StartCell;
            public Vector2Int TargetCell;
            public bool HasStartAndTarget;
            public HashSet<Vector2Int> ProtectedCells = new HashSet<Vector2Int>();

            public bool IsInside(Vector2Int cell)
            {
                return cell.x >= 0 && cell.y >= 0 && cell.x < MapSize.x && cell.y < MapSize.y;
            }

            public bool IsStartOrTarget(Vector2Int cell)
            {
                return HasStartAndTarget && (cell == StartCell || cell == TargetCell);
            }

            public bool IsUnsafe(Vector2Int cell)
            {
                return IsInside(cell) &&
                    (ObstacleCells[cell.x, cell.y] || HazardCells[cell.x, cell.y] || CollapseCells[cell.x, cell.y]);
            }
        }
    }
}
