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
        private const float StairStepRise = 0.22f;
        private const float BuildingFloorHeight = 3.2f;
        private const float HillMaxNeighborHeightDelta = 0.45f;

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
                FloorCount = 1,
                FloorHeight = BuildingFloorHeight,
                StartNode = Vector3Int.zero,
                TargetNode = Vector3Int.zero,
                WalkableNodes = new Vector3Int[0],
                StairNodes = new Vector3Int[0],
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
                    return GenerateBuildingMap(context) && AddSpreadingFire(context) && AddRandomCollapse(context);
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
            CreateFloor(context);

            var horizontal = context.MapSize.x >= context.MapSize.y;
            if (context.MapSize.x == context.MapSize.y)
            {
                horizontal = context.Random.Next(0, 2) == 0;
            }

            var usableLength = horizontal ? context.MapSize.x : context.MapSize.y;
            var stepCount = Mathf.Clamp(usableLength - 6, 4, 8);
            var centerX = context.MapSize.x / 2;
            var centerY = context.MapSize.y / 2;
            var width = context.CellSize * 3f;
            var topHeight = stepCount * StairStepRise;
            var bottomCell = horizontal
                ? new Vector2Int(1, centerY)
                : new Vector2Int(centerX, 1);
            var topCell = horizontal
                ? new Vector2Int(stepCount + 3, centerY)
                : new Vector2Int(centerX, stepCount + 3);

            topCell.x = Mathf.Clamp(topCell.x, 1, context.MapSize.x - 2);
            topCell.y = Mathf.Clamp(topCell.y, 1, context.MapSize.y - 2);
            var surfaceHeights = new Dictionary<Vector2Int, float>();

            var bottomPlatformScale = horizontal
                ? new Vector3(context.CellSize * 2f, 0.1f, width)
                : new Vector3(width, 0.1f, context.CellSize * 2f);
            CreatePrimitiveBlock(
                "StairBottomPlatform",
                context.GeometryRoot,
                RLTrainingGenerationUtility.CellToLocalPosition(bottomCell, context.MapSize, context.CellSize, 0.02f),
                bottomPlatformScale);
            MarkStairSurfaceCells(context, surfaceHeights, bottomCell, horizontal, 0, 1, 1, 0.07f);

            for (var i = 0; i < stepCount; i++)
            {
                var cell = horizontal
                    ? new Vector2Int(Mathf.Clamp(2 + i, 1, context.MapSize.x - 2), centerY)
                    : new Vector2Int(centerX, Mathf.Clamp(2 + i, 1, context.MapSize.y - 2));

                var height = StairStepRise * (i + 1);
                var localPosition = RLTrainingGenerationUtility.CellToLocalPosition(cell, context.MapSize, context.CellSize, height * 0.5f);
                var localScale = horizontal
                    ? new Vector3(context.CellSize, height, width)
                    : new Vector3(width, height, context.CellSize);
                CreatePrimitiveBlock("StairStep_" + i, context.GeometryRoot, localPosition, localScale);
                MarkStairSurfaceCells(context, surfaceHeights, cell, horizontal, 0, 0, 1, height);
            }

            var topPlatformScale = horizontal
                ? new Vector3(context.CellSize * 2.5f, topHeight, width)
                : new Vector3(width, topHeight, context.CellSize * 2.5f);
            CreatePrimitiveBlock(
                "StairUpperPlatform",
                context.GeometryRoot,
                RLTrainingGenerationUtility.CellToLocalPosition(topCell, context.MapSize, context.CellSize, topHeight * 0.5f),
                topPlatformScale);
            MarkStairSurfaceCells(context, surfaceHeights, topCell, horizontal, 1, 1, 1, topHeight);

            Vector2Int startCell;
            float startSurfaceHeight;
            if (!TryChooseStairStartCell(context, topCell, surfaceHeights, out startCell, out startSurfaceHeight))
            {
                Debug.LogWarning("RLMapGenerator could not place a valid stair start cell.", this);
                return false;
            }

            SetStartAndTarget(context, startCell, topCell);
            context.StartLocalPosition = RLTrainingGenerationUtility.CellToLocalPosition(startCell, context.MapSize, context.CellSize, startSurfaceHeight + 0.25f);
            context.TargetLocalPosition = RLTrainingGenerationUtility.CellToLocalPosition(topCell, context.MapSize, context.CellSize, topHeight + 0.08f);
            context.HasCustomStartPosition = true;
            context.HasCustomTargetPosition = true;

            return true;
        }

        private bool TryChooseStairStartCell(
            MapBuildContext context,
            Vector2Int targetCell,
            Dictionary<Vector2Int, float> surfaceHeights,
            out Vector2Int startCell,
            out float startSurfaceHeight)
        {
            var candidates = new List<Vector2Int>();
            for (var y = 1; y < context.MapSize.y - 1; y++)
            {
                for (var x = 1; x < context.MapSize.x - 1; x++)
                {
                    var cell = new Vector2Int(x, y);
                    if (cell == targetCell || context.IsUnsafe(cell) || !HasPath(context, cell, targetCell))
                    {
                        continue;
                    }

                    candidates.Add(cell);
                }
            }

            if (candidates.Count == 0)
            {
                startCell = Vector2Int.zero;
                startSurfaceHeight = 0f;
                return false;
            }

            startCell = candidates[context.Random.Next(candidates.Count)];
            startSurfaceHeight = GetStairSurfaceHeight(surfaceHeights, startCell);
            return true;
        }

        private static float GetStairSurfaceHeight(Dictionary<Vector2Int, float> surfaceHeights, Vector2Int cell)
        {
            float surfaceHeight;
            return surfaceHeights != null && surfaceHeights.TryGetValue(cell, out surfaceHeight)
                ? surfaceHeight
                : 0f;
        }

        private void MarkStairSurfaceCells(
            MapBuildContext context,
            Dictionary<Vector2Int, float> surfaceHeights,
            Vector2Int centerCell,
            bool horizontal,
            int backwardCells,
            int forwardCells,
            int halfWidthCells,
            float surfaceHeight)
        {
            for (var along = -backwardCells; along <= forwardCells; along++)
            {
                for (var across = -halfWidthCells; across <= halfWidthCells; across++)
                {
                    var cell = horizontal
                        ? new Vector2Int(centerCell.x + along, centerCell.y + across)
                        : new Vector2Int(centerCell.x + across, centerCell.y + along);
                    if (!context.IsInside(cell))
                    {
                        continue;
                    }

                    float existingHeight;
                    if (!surfaceHeights.TryGetValue(cell, out existingHeight) || surfaceHeight > existingHeight)
                    {
                        surfaceHeights[cell] = surfaceHeight;
                    }
                }
            }
        }

        private bool GenerateHillMap(MapBuildContext context)
        {
            var heights = GeneratePerlinHeights(context);
            context.CellHeights = CalculateCellHeights(context, heights);
            CreateHillTerrainMesh(context, heights);

            if (!TryChooseHillStartAndTarget(context))
            {
                Debug.LogWarning("RLMapGenerator could not place valid hill start/target cells.", this);
                return false;
            }

            return true;
        }

        private bool GenerateBuildingMap(MapBuildContext context)
        {
            var building = CreateBuildingGrid(context);
            if (building == null)
            {
                return false;
            }

            var layeredPath = FindLayeredPath(building, building.StartNode, building.TargetNode);
            if (layeredPath.Count == 0)
            {
                Debug.LogWarning("RLMapGenerator could not validate a layered building path before obstacle placement.", this);
                return false;
            }

            for (var i = 0; i < layeredPath.Count; i++)
            {
                building.ProtectedNodes.Add(layeredPath[i]);
                context.ProtectedCells.Add(new Vector2Int(layeredPath[i].x, layeredPath[i].y));
            }

            AddSparseBuildingObstacles(context, building);
            layeredPath = FindLayeredPath(building, building.StartNode, building.TargetNode);
            if (layeredPath.Count == 0)
            {
                Debug.LogWarning("RLMapGenerator sparse building obstacles blocked the only layered path.", this);
                return false;
            }

            context.HasLayeredNavigation = true;
            context.FloorCount = building.FloorCount;
            context.FloorHeight = BuildingFloorHeight;
            context.WalkableNodes = GetWalkableNodes(building);
            context.StairNodes = GetStairNodes(building);
            SetStartAndTarget(
                context,
                new Vector2Int(building.StartNode.x, building.StartNode.y),
                new Vector2Int(building.TargetNode.x, building.TargetNode.y));
            context.StartNode = building.StartNode;
            context.TargetNode = building.TargetNode;
            context.StartLocalPosition = GetBuildingNodeLocalPosition(context, building.StartNode, 0.25f);
            context.TargetLocalPosition = GetBuildingNodeLocalPosition(context, building.TargetNode, 0.08f);
            context.HasCustomStartPosition = true;
            context.HasCustomTargetPosition = true;

            CreateBuildingGeometry(context, building);
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
                candidateCells,
                GetCollapseDropHeight(context));

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

            if (context.HasLayeredNavigation)
            {
                if (!HasLayeredPath(context, context.StartNode, context.TargetNode))
                {
                    Debug.LogWarning("RLMapGenerator validation failed because no layered building path exists from start to target.", this);
                    return false;
                }

                return true;
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
                context.FloorCount,
                context.FloorHeight,
                context.StartCell,
                context.TargetCell,
                context.StartNode,
                context.TargetNode,
                obstacleCells,
                hazardCells,
                collapseCells,
                context.WalkableNodes,
                context.StairNodes);

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
            context.StartNode = new Vector3Int(start.x, start.y, 0);
            context.TargetNode = new Vector3Int(target.x, target.y, 0);
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

        private float[,] GeneratePerlinHeights(MapBuildContext context)
        {
            var vertexWidth = context.MapSize.x + 1;
            var vertexDepth = context.MapSize.y + 1;
            var heights = new float[vertexWidth, vertexDepth];
            var offsetX = Mathf.Abs(context.Seed % 10007) * 0.037f;
            var offsetZ = Mathf.Abs((context.Seed / 17) % 10007) * 0.041f;

            for (var z = 0; z < vertexDepth; z++)
            {
                for (var x = 0; x < vertexWidth; x++)
                {
                    var broad = Mathf.PerlinNoise((x + offsetX) * 0.12f, (z + offsetZ) * 0.12f);
                    var detail = Mathf.PerlinNoise((x + offsetX + 31f) * 0.28f, (z + offsetZ + 19f) * 0.28f);
                    heights[x, z] = Mathf.Lerp(0.05f, 1.15f, broad * 0.78f + detail * 0.22f);
                }
            }

            ConstrainHillSlopes(context, heights);
            return heights;
        }

        private void ConstrainHillSlopes(MapBuildContext context, float[,] heights)
        {
            var maxDelta = Mathf.Max(HillMaxNeighborHeightDelta, context.CellSize * 0.42f);
            var width = heights.GetLength(0);
            var depth = heights.GetLength(1);

            var maxPasses = Mathf.Max(width, depth) * 2;
            for (var pass = 0; pass < maxPasses; pass++)
            {
                var changed = false;
                for (var z = 0; z < depth; z++)
                {
                    for (var x = 0; x < width; x++)
                    {
                        changed |= ConstrainHillHeightToNeighbor(heights, x, z, x - 1, z, maxDelta);
                        changed |= ConstrainHillHeightToNeighbor(heights, x, z, x, z - 1, maxDelta);
                    }
                }

                for (var z = depth - 1; z >= 0; z--)
                {
                    for (var x = width - 1; x >= 0; x--)
                    {
                        changed |= ConstrainHillHeightToNeighbor(heights, x, z, x + 1, z, maxDelta);
                        changed |= ConstrainHillHeightToNeighbor(heights, x, z, x, z + 1, maxDelta);
                    }
                }

                if (!changed)
                {
                    return;
                }
            }
        }

        private static bool ConstrainHillHeightToNeighbor(float[,] heights, int x, int z, int neighborX, int neighborZ, float maxDelta)
        {
            if (neighborX < 0 || neighborZ < 0 || neighborX >= heights.GetLength(0) || neighborZ >= heights.GetLength(1))
            {
                return false;
            }

            var original = heights[x, z];
            var neighbor = heights[neighborX, neighborZ];
            heights[x, z] = Mathf.Clamp(original, neighbor - maxDelta, neighbor + maxDelta);
            return !Mathf.Approximately(original, heights[x, z]);
        }

        private float[,] CalculateCellHeights(MapBuildContext context, float[,] vertexHeights)
        {
            var cellHeights = new float[context.MapSize.x, context.MapSize.y];
            for (var z = 0; z < context.MapSize.y; z++)
            {
                for (var x = 0; x < context.MapSize.x; x++)
                {
                    cellHeights[x, z] =
                        (vertexHeights[x, z] +
                        vertexHeights[x + 1, z] +
                        vertexHeights[x, z + 1] +
                        vertexHeights[x + 1, z + 1]) * 0.25f;
                }
            }

            return cellHeights;
        }

        private void CreateHillTerrainMesh(MapBuildContext context, float[,] heights)
        {
            var terrainObject = new GameObject("PerlinHillTerrain");
            terrainObject.transform.SetParent(context.GeometryRoot, false);

            var vertexWidth = context.MapSize.x + 1;
            var vertexDepth = context.MapSize.y + 1;
            var vertices = new Vector3[vertexWidth * vertexDepth];
            var triangles = new int[context.MapSize.x * context.MapSize.y * 6];
            var halfWidth = context.MapSize.x * context.CellSize * 0.5f;
            var halfDepth = context.MapSize.y * context.CellSize * 0.5f;

            for (var z = 0; z < vertexDepth; z++)
            {
                for (var x = 0; x < vertexWidth; x++)
                {
                    var index = z * vertexWidth + x;
                    vertices[index] = new Vector3(
                        x * context.CellSize - halfWidth,
                        heights[x, z],
                        z * context.CellSize - halfDepth);
                }
            }

            var triangleIndex = 0;
            for (var z = 0; z < context.MapSize.y; z++)
            {
                for (var x = 0; x < context.MapSize.x; x++)
                {
                    var lowerLeft = z * vertexWidth + x;
                    var lowerRight = lowerLeft + 1;
                    var upperLeft = lowerLeft + vertexWidth;
                    var upperRight = upperLeft + 1;

                    triangles[triangleIndex++] = lowerLeft;
                    triangles[triangleIndex++] = upperLeft;
                    triangles[triangleIndex++] = lowerRight;
                    triangles[triangleIndex++] = lowerRight;
                    triangles[triangleIndex++] = upperLeft;
                    triangles[triangleIndex++] = upperRight;
                }
            }

            var mesh = new Mesh
            {
                name = "Generated Perlin Hill Terrain",
                vertices = vertices,
                triangles = triangles
            };
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();

            var meshFilter = terrainObject.AddComponent<MeshFilter>();
            meshFilter.sharedMesh = mesh;
            var meshRenderer = terrainObject.AddComponent<MeshRenderer>();
            meshRenderer.sharedMaterial = CreateRuntimeMaterial("Runtime Hill Terrain", new Color(0.35f, 0.58f, 0.28f, 1f));
            var meshCollider = terrainObject.AddComponent<MeshCollider>();
            meshCollider.sharedMesh = mesh;
        }

        private bool TryChooseHillStartAndTarget(MapBuildContext context)
        {
            var candidates = new List<Vector2Int>();
            for (var z = 1; z < context.MapSize.y - 1; z++)
            {
                for (var x = 1; x < context.MapSize.x - 1; x++)
                {
                    var cell = new Vector2Int(x, z);
                    if (IsTraversableHillCell(context, cell))
                    {
                        candidates.Add(cell);
                    }
                }
            }

            if (candidates.Count < 2)
            {
                return false;
            }

            var minDistance = Mathf.Max(4, (context.MapSize.x + context.MapSize.y) / 3);
            for (var attempt = 0; attempt < 128; attempt++)
            {
                var start = candidates[context.Random.Next(candidates.Count)];
                var target = candidates[context.Random.Next(candidates.Count)];
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
                context.StartLocalPosition = RLTrainingGenerationUtility.CellToLocalPosition(
                    start,
                    context.MapSize,
                    context.CellSize,
                    context.CellHeights[start.x, start.y] + 0.25f);
                context.TargetLocalPosition = RLTrainingGenerationUtility.CellToLocalPosition(
                    target,
                    context.MapSize,
                    context.CellSize,
                    context.CellHeights[target.x, target.y] + 0.08f);
                context.HasCustomStartPosition = true;
                context.HasCustomTargetPosition = true;
                return true;
            }

            return false;
        }

        private bool IsTraversableHillCell(MapBuildContext context, Vector2Int cell)
        {
            var height = context.CellHeights[cell.x, cell.y];
            var maxDelta = Mathf.Max(HillMaxNeighborHeightDelta, context.CellSize * 0.42f);
            for (var i = 0; i < PathDirections.Length; i++)
            {
                var neighbor = cell + PathDirections[i];
                if (!context.IsInside(neighbor))
                {
                    continue;
                }

                if (Mathf.Abs(height - context.CellHeights[neighbor.x, neighbor.y]) > maxDelta)
                {
                    return false;
                }
            }

            return true;
        }

        private BuildingGrid CreateBuildingGrid(MapBuildContext context)
        {
            var floorCount = 2 + context.Random.Next(0, 2);
            var building = new BuildingGrid(context.MapSize.x, context.MapSize.y, floorCount);
            var corridorX = context.MapSize.x / 2;
            var corridorZ = context.MapSize.y / 2;
            var stairCell = new Vector2Int(corridorX, corridorZ);

            for (var floor = 0; floor < floorCount; floor++)
            {
                MarkCorridor(building, floor, corridorX, corridorZ);
                MarkRoom(building, floor, 1, 1, corridorX - 2, corridorZ - 2);
                MarkRoom(building, floor, corridorX + 2, 1, context.MapSize.x - 2, corridorZ - 2);
                MarkRoom(building, floor, 1, corridorZ + 2, corridorX - 2, context.MapSize.y - 2);
                MarkRoom(building, floor, corridorX + 2, corridorZ + 2, context.MapSize.x - 2, context.MapSize.y - 2);
                MarkRoomDoorways(building, floor, corridorX, corridorZ);

                building.Walkable[floor, stairCell.x, stairCell.y] = true;
                building.Stairs[floor, stairCell.x, stairCell.y] = true;
            }

            building.StartNode = FindNearestWalkableNode(building, new Vector3Int(1, 1, 0));
            building.TargetNode = FindNearestWalkableNode(building, new Vector3Int(context.MapSize.x - 2, context.MapSize.y - 2, floorCount - 1));
            return building;
        }

        private void MarkCorridor(BuildingGrid building, int floor, int corridorX, int corridorZ)
        {
            for (var z = 1; z < building.Depth - 1; z++)
            {
                building.Walkable[floor, corridorX, z] = true;
                if (corridorX - 1 > 0)
                {
                    building.Walkable[floor, corridorX - 1, z] = true;
                }
            }

            for (var x = 1; x < building.Width - 1; x++)
            {
                building.Walkable[floor, x, corridorZ] = true;
                if (corridorZ - 1 > 0)
                {
                    building.Walkable[floor, x, corridorZ - 1] = true;
                }
            }
        }

        private void MarkRoom(BuildingGrid building, int floor, int minX, int minZ, int maxX, int maxZ)
        {
            minX = Mathf.Clamp(minX, 1, building.Width - 2);
            minZ = Mathf.Clamp(minZ, 1, building.Depth - 2);
            maxX = Mathf.Clamp(maxX, 1, building.Width - 2);
            maxZ = Mathf.Clamp(maxZ, 1, building.Depth - 2);
            if (minX > maxX || minZ > maxZ)
            {
                return;
            }

            for (var z = minZ; z <= maxZ; z++)
            {
                for (var x = minX; x <= maxX; x++)
                {
                    building.Walkable[floor, x, z] = true;
                }
            }
        }

        private void MarkRoomDoorways(BuildingGrid building, int floor, int corridorX, int corridorZ)
        {
            MarkWalkableIfInside(building, floor, corridorX - 2, corridorZ - 1);
            MarkWalkableIfInside(building, floor, corridorX - 1, corridorZ - 2);
            MarkWalkableIfInside(building, floor, corridorX + 1, corridorZ - 2);
            MarkWalkableIfInside(building, floor, corridorX + 2, corridorZ - 1);
            MarkWalkableIfInside(building, floor, corridorX - 2, corridorZ + 1);
            MarkWalkableIfInside(building, floor, corridorX - 1, corridorZ + 2);
            MarkWalkableIfInside(building, floor, corridorX + 1, corridorZ + 2);
            MarkWalkableIfInside(building, floor, corridorX + 2, corridorZ + 1);
        }

        private void MarkWalkableIfInside(BuildingGrid building, int floor, int x, int z)
        {
            if (floor < 0 || floor >= building.FloorCount || x < 1 || z < 1 || x >= building.Width - 1 || z >= building.Depth - 1)
            {
                return;
            }

            building.Walkable[floor, x, z] = true;
        }

        private void AddSparseBuildingObstacles(MapBuildContext context, BuildingGrid building)
        {
            var candidates = new List<Vector3Int>();
            for (var floor = 0; floor < building.FloorCount; floor++)
            {
                for (var z = 1; z < building.Depth - 1; z++)
                {
                    for (var x = 1; x < building.Width - 1; x++)
                    {
                        var node = new Vector3Int(x, z, floor);
                        var cell = new Vector2Int(x, z);
                        if (IsBuildingWalkable(building, node) &&
                            !building.Stairs[floor, x, z] &&
                            !building.ProtectedNodes.Contains(node) &&
                            !context.ProtectedCells.Contains(cell))
                        {
                            candidates.Add(node);
                        }
                    }
                }
            }

            var obstacleTarget = Mathf.Max(1, Mathf.RoundToInt(candidates.Count * 0.04f));
            for (var i = 0; i < obstacleTarget && candidates.Count > 0; i++)
            {
                var chosenIndex = context.Random.Next(candidates.Count);
                var node = candidates[chosenIndex];
                candidates.RemoveAt(chosenIndex);
                building.Blocked[node.z, node.x, node.y] = true;
                MarkObstacle(context, new Vector2Int(node.x, node.y));
            }
        }

        private void CreateBuildingGeometry(MapBuildContext context, BuildingGrid building)
        {
            for (var floor = 0; floor < building.FloorCount; floor++)
            {
                var floorY = floor * BuildingFloorHeight;
                CreatePrimitiveBlock(
                    "BuildingFloor_" + floor,
                    context.GeometryRoot,
                    new Vector3(0f, floorY - 0.05f, 0f),
                    new Vector3(building.Width * context.CellSize, 0.1f, building.Depth * context.CellSize));

                for (var z = 0; z < building.Depth; z++)
                {
                    for (var x = 0; x < building.Width; x++)
                    {
                        var cell = new Vector2Int(x, z);
                        if (!building.Walkable[floor, x, z])
                        {
                            MarkObstacle(context, cell);
                            CreatePrimitiveBlock(
                                string.Format("Wall_F{0}_{1}_{2}", floor, x, z),
                                context.GeometryRoot,
                                RLTrainingGenerationUtility.CellToLocalPosition(cell, context.MapSize, context.CellSize, floorY + 1.25f),
                                new Vector3(context.CellSize * 0.95f, 2.5f, context.CellSize * 0.95f));
                            continue;
                        }

                        if (building.Blocked[floor, x, z])
                        {
                            CreatePrimitiveBlock(
                                string.Format("IndoorObstacle_F{0}_{1}_{2}", floor, x, z),
                                context.ObstaclesRoot,
                                RLTrainingGenerationUtility.CellToLocalPosition(cell, context.MapSize, context.CellSize, floorY + 0.45f),
                                new Vector3(context.CellSize * 0.45f, 0.9f, context.CellSize * 0.45f));
                        }
                    }
                }
            }

            CreateBuildingStairVisuals(context, building);
        }

        private void CreateBuildingStairVisuals(MapBuildContext context, BuildingGrid building)
        {
            var stairCell = new Vector2Int(building.Width / 2, building.Depth / 2);
            var stepCount = 7;
            for (var floor = 0; floor < building.FloorCount - 1; floor++)
            {
                var floorY = floor * BuildingFloorHeight;
                for (var step = 0; step < stepCount; step++)
                {
                    var t = (step + 1f) / stepCount;
                    var local = RLTrainingGenerationUtility.CellToLocalPosition(
                        stairCell,
                        context.MapSize,
                        context.CellSize,
                        floorY + BuildingFloorHeight * t * 0.5f);
                    local.z += Mathf.Lerp(-context.CellSize * 0.45f, context.CellSize * 0.45f, t);

                    CreatePrimitiveBlock(
                        string.Format("StairConnector_F{0}_{1}", floor, step),
                        context.GeometryRoot,
                        local,
                        new Vector3(context.CellSize * 0.8f, BuildingFloorHeight * t, context.CellSize * 0.28f));
                }
            }
        }

        private Vector3Int FindNearestWalkableNode(BuildingGrid building, Vector3Int preferred)
        {
            preferred.x = Mathf.Clamp(preferred.x, 0, building.Width - 1);
            preferred.y = Mathf.Clamp(preferred.y, 0, building.Depth - 1);
            preferred.z = Mathf.Clamp(preferred.z, 0, building.FloorCount - 1);
            if (IsBuildingWalkable(building, preferred))
            {
                return preferred;
            }

            for (var radius = 1; radius < Mathf.Max(building.Width, building.Depth); radius++)
            {
                for (var z = preferred.y - radius; z <= preferred.y + radius; z++)
                {
                    for (var x = preferred.x - radius; x <= preferred.x + radius; x++)
                    {
                        var node = new Vector3Int(x, z, preferred.z);
                        if (IsBuildingWalkable(building, node))
                        {
                            return node;
                        }
                    }
                }
            }

            return new Vector3Int(building.Width / 2, building.Depth / 2, preferred.z);
        }

        private List<Vector3Int> FindLayeredPath(BuildingGrid building, Vector3Int start, Vector3Int target)
        {
            var cameFrom = new Dictionary<Vector3Int, Vector3Int>();
            var visited = new HashSet<Vector3Int>();
            var queue = new Queue<Vector3Int>();
            visited.Add(start);
            queue.Enqueue(start);

            while (queue.Count > 0)
            {
                var current = queue.Dequeue();
                if (current == target)
                {
                    return ReconstructLayeredPath(cameFrom, start, target);
                }

                var neighbors = GetLayeredNeighbors(building, current);
                for (var i = 0; i < neighbors.Count; i++)
                {
                    var next = neighbors[i];
                    if (visited.Contains(next))
                    {
                        continue;
                    }

                    visited.Add(next);
                    cameFrom[next] = current;
                    queue.Enqueue(next);
                }
            }

            return new List<Vector3Int>();
        }

        private List<Vector3Int> ReconstructLayeredPath(Dictionary<Vector3Int, Vector3Int> cameFrom, Vector3Int start, Vector3Int target)
        {
            var path = new List<Vector3Int>();
            var current = target;
            path.Add(current);

            while (current != start)
            {
                Vector3Int previous;
                if (!cameFrom.TryGetValue(current, out previous))
                {
                    return new List<Vector3Int>();
                }

                current = previous;
                path.Add(current);
            }

            path.Reverse();
            return path;
        }

        private List<Vector3Int> GetLayeredNeighbors(BuildingGrid building, Vector3Int current)
        {
            var neighbors = new List<Vector3Int>();
            AddLayeredNeighbor(building, current + new Vector3Int(1, 0, 0), neighbors);
            AddLayeredNeighbor(building, current + new Vector3Int(-1, 0, 0), neighbors);
            AddLayeredNeighbor(building, current + new Vector3Int(0, 1, 0), neighbors);
            AddLayeredNeighbor(building, current + new Vector3Int(0, -1, 0), neighbors);

            if (IsBuildingStair(building, current))
            {
                var up = current + new Vector3Int(0, 0, 1);
                var down = current + new Vector3Int(0, 0, -1);
                if (IsBuildingStair(building, up))
                {
                    AddLayeredNeighbor(building, up, neighbors);
                }

                if (IsBuildingStair(building, down))
                {
                    AddLayeredNeighbor(building, down, neighbors);
                }
            }

            return neighbors;
        }

        private void AddLayeredNeighbor(BuildingGrid building, Vector3Int node, List<Vector3Int> neighbors)
        {
            if (IsBuildingWalkable(building, node))
            {
                neighbors.Add(node);
            }
        }

        private bool HasLayeredPath(MapBuildContext context, Vector3Int start, Vector3Int target)
        {
            var building = new BuildingGrid(context.MapSize.x, context.MapSize.y, context.FloorCount);
            for (var i = 0; i < context.WalkableNodes.Length; i++)
            {
                var node = context.WalkableNodes[i];
                building.Walkable[node.z, node.x, node.y] = true;
            }

            for (var i = 0; i < context.StairNodes.Length; i++)
            {
                var node = context.StairNodes[i];
                building.Stairs[node.z, node.x, node.y] = true;
            }

            return FindLayeredPath(building, start, target).Count > 0;
        }

        private bool IsBuildingWalkable(BuildingGrid building, Vector3Int node)
        {
            return node.x >= 0 &&
                node.y >= 0 &&
                node.z >= 0 &&
                node.x < building.Width &&
                node.y < building.Depth &&
                node.z < building.FloorCount &&
                building.Walkable[node.z, node.x, node.y] &&
                !building.Blocked[node.z, node.x, node.y];
        }

        private bool IsBuildingStair(BuildingGrid building, Vector3Int node)
        {
            return node.x >= 0 &&
                node.y >= 0 &&
                node.z >= 0 &&
                node.x < building.Width &&
                node.y < building.Depth &&
                node.z < building.FloorCount &&
                building.Stairs[node.z, node.x, node.y];
        }

        private Vector3Int[] GetWalkableNodes(BuildingGrid building)
        {
            var nodes = new List<Vector3Int>();
            for (var floor = 0; floor < building.FloorCount; floor++)
            {
                for (var z = 0; z < building.Depth; z++)
                {
                    for (var x = 0; x < building.Width; x++)
                    {
                        var node = new Vector3Int(x, z, floor);
                        if (IsBuildingWalkable(building, node))
                        {
                            nodes.Add(node);
                        }
                    }
                }
            }

            return nodes.ToArray();
        }

        private Vector3Int[] GetStairNodes(BuildingGrid building)
        {
            var nodes = new List<Vector3Int>();
            for (var floor = 0; floor < building.FloorCount; floor++)
            {
                for (var z = 0; z < building.Depth; z++)
                {
                    for (var x = 0; x < building.Width; x++)
                    {
                        if (building.Stairs[floor, x, z] && !building.Blocked[floor, x, z])
                        {
                            nodes.Add(new Vector3Int(x, z, floor));
                        }
                    }
                }
            }

            return nodes.ToArray();
        }

        private Vector3 GetBuildingNodeLocalPosition(MapBuildContext context, Vector3Int node, float yOffset)
        {
            return RLTrainingGenerationUtility.CellToLocalPosition(
                new Vector2Int(node.x, node.y),
                context.MapSize,
                context.CellSize,
                node.z * BuildingFloorHeight + yOffset);
        }

        private float GetCollapseDropHeight(MapBuildContext context)
        {
            return context.HasLayeredNavigation
                ? context.FloorCount * BuildingFloorHeight + 4f
                : 9f;
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
            RLFireVisuals.Apply(fireObject, context.CellSize, context.Seed + cell.x * 43 + cell.y * 101);
            MarkHazard(context, cell);
        }

        private Transform CreateStartMarker(MapBuildContext context)
        {
            var marker = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            marker.name = "StartPosition";
            marker.transform.SetParent(context.MarkersRoot, false);
            marker.transform.localPosition = context.HasCustomStartPosition
                ? context.StartLocalPosition
                : RLTrainingGenerationUtility.CellToLocalPosition(context.StartCell, context.MapSize, context.CellSize, 0.25f);
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
            marker.transform.localPosition = context.HasCustomTargetPosition
                ? context.TargetLocalPosition
                : RLTrainingGenerationUtility.CellToLocalPosition(context.TargetCell, context.MapSize, context.CellSize, 0.06f);
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

        private static Material CreateRuntimeMaterial(string materialName, Color baseColor)
        {
            var shader = Shader.Find("Universal Render Pipeline/Lit");
            if (shader == null)
            {
                shader = Shader.Find("Standard");
            }

            if (shader == null)
            {
                return null;
            }

            var material = new Material(shader)
            {
                name = materialName
            };

            if (material.HasProperty("_BaseColor"))
            {
                material.SetColor("_BaseColor", baseColor);
            }

            if (material.HasProperty("_Color"))
            {
                material.SetColor("_Color", baseColor);
            }

            return material;
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
            public int FloorCount;
            public float FloorHeight;
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
            public float[,] CellHeights;
            public Vector2Int StartCell;
            public Vector2Int TargetCell;
            public Vector3Int StartNode;
            public Vector3Int TargetNode;
            public bool HasStartAndTarget;
            public bool HasCustomStartPosition;
            public bool HasCustomTargetPosition;
            public Vector3 StartLocalPosition;
            public Vector3 TargetLocalPosition;
            public bool HasLayeredNavigation;
            public Vector3Int[] WalkableNodes;
            public Vector3Int[] StairNodes;
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

        private sealed class BuildingGrid
        {
            public BuildingGrid(int width, int depth, int floorCount)
            {
                Width = width;
                Depth = depth;
                FloorCount = floorCount;
                Walkable = new bool[floorCount, width, depth];
                Blocked = new bool[floorCount, width, depth];
                Stairs = new bool[floorCount, width, depth];
            }

            public int Width;
            public int Depth;
            public int FloorCount;
            public bool[,,] Walkable;
            public bool[,,] Blocked;
            public bool[,,] Stairs;
            public Vector3Int StartNode;
            public Vector3Int TargetNode;
            public HashSet<Vector3Int> ProtectedNodes = new HashSet<Vector3Int>();
        }
    }
}
