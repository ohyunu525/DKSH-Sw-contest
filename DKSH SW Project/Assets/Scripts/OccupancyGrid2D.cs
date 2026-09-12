using System;
using System.Collections.Generic;
using UnityEngine;

namespace DKSH.Spiderbot.Mapping
{
    public enum OccupancyCellState : sbyte
    {
        Unknown = -1,
        Free = 0,
        Occupied = 100
    }

    [DisallowMultipleComponent]
    public sealed class OccupancyGrid2D : MonoBehaviour
    {
        [Header("Grid (Unity X-Z plane)")]
        [SerializeField, Min(0.01f)]
        private float resolution = 0.1f;

        [SerializeField, Min(1)]
        private int width = 200;

        [SerializeField, Min(1)]
        private int height = 200;

        [SerializeField]
        private Vector3 mapOrigin = new Vector3(-10f, 0.02f, -10f);

        [Header("Inverse sensor model")]
        [SerializeField, Min(1)]
        private int occupiedThreshold = 3;

        [SerializeField, Min(1)]
        private int occupiedEvidenceIncrement = 4;

        [SerializeField, Min(1)]
        private int freeEvidenceDecrement = 1;

        [SerializeField]
        private int minimumEvidence = -20;

        [SerializeField]
        private int maximumEvidence = 20;

        [Header("Debug")]
        [SerializeField]
        private bool drawDebugBounds = true;

        [SerializeField]
        private Color debugBoundsColor = new Color(0.2f, 0.8f, 1f, 0.8f);

        private OccupancyCellState[] cells;
        private short[] evidence;
        private readonly List<int> changedCellIndices = new List<int>(1024);
        private int batchDepth;

        public event Action<OccupancyGrid2D, IReadOnlyList<int>> CellsChanged;
        public event Action<OccupancyGrid2D> GridReset;

        public float Resolution { get { return resolution; } }
        public int Width { get { return width; } }
        public int Height { get { return height; } }
        public int CellCount { get { return width * height; } }
        public Vector3 MapOrigin { get { return mapOrigin; } }
        public int OccupiedThreshold { get { return occupiedThreshold; } }
        public int Version { get; private set; }
        public int UnknownCellCount { get; private set; }
        public int FreeCellCount { get; private set; }
        public int OccupiedCellCount { get; private set; }

        public IReadOnlyList<OccupancyCellState> Cells
        {
            get
            {
                EnsureStorage();
                return cells;
            }
        }

        public void Configure(
            float configuredResolution,
            int configuredWidth,
            int configuredHeight,
            Vector3 configuredOrigin)
        {
            resolution = Mathf.Max(0.01f, configuredResolution);
            width = Mathf.Max(1, configuredWidth);
            height = Mathf.Max(1, configuredHeight);
            mapOrigin = configuredOrigin;
            ResetGrid();
        }

        public void ConfigureEvidence(
            int configuredOccupiedThreshold,
            int configuredOccupiedIncrement,
            int configuredFreeDecrement,
            int configuredMinimumEvidence,
            int configuredMaximumEvidence)
        {
            occupiedThreshold = Mathf.Max(1, configuredOccupiedThreshold);
            occupiedEvidenceIncrement = Mathf.Max(occupiedThreshold, configuredOccupiedIncrement);
            freeEvidenceDecrement = Mathf.Max(1, configuredFreeDecrement);
            minimumEvidence = Mathf.Min(-1, configuredMinimumEvidence);
            maximumEvidence = Mathf.Max(occupiedThreshold, configuredMaximumEvidence);
            ResetGrid();
        }

        public bool WorldToGrid(Vector3 worldPosition, out int x, out int y)
        {
            WorldToGridUnchecked(worldPosition, out x, out y);
            return IsInsideGrid(x, y);
        }

        public Vector2Int WorldToGridUnchecked(Vector3 worldPosition)
        {
            WorldToGridUnchecked(worldPosition, out var x, out var y);
            return new Vector2Int(x, y);
        }

        public void WorldToGridUnchecked(Vector3 worldPosition, out int x, out int y)
        {
            x = Mathf.FloorToInt((worldPosition.x - mapOrigin.x) / resolution);
            y = Mathf.FloorToInt((worldPosition.z - mapOrigin.z) / resolution);
        }

        public Vector3 GridToWorld(int x, int y)
        {
            return new Vector3(
                mapOrigin.x + (x + 0.5f) * resolution,
                mapOrigin.y,
                mapOrigin.z + (y + 0.5f) * resolution);
        }

        public Vector3 GridCornerToWorld(int x, int y)
        {
            return new Vector3(
                mapOrigin.x + x * resolution,
                mapOrigin.y,
                mapOrigin.z + y * resolution);
        }

        public bool IsInsideGrid(int x, int y)
        {
            return x >= 0 && x < width && y >= 0 && y < height;
        }

        public int GetFlatIndex(int x, int y)
        {
            return IsInsideGrid(x, y) ? y * width + x : -1;
        }

        public bool TryGetCoordinates(int index, out int x, out int y)
        {
            if (index < 0 || index >= CellCount)
            {
                x = -1;
                y = -1;
                return false;
            }

            x = index % width;
            y = index / width;
            return true;
        }

        public OccupancyCellState GetCell(int x, int y)
        {
            EnsureStorage();
            var index = GetFlatIndex(x, y);
            return index >= 0 ? cells[index] : OccupancyCellState.Unknown;
        }

        public OccupancyCellState GetCellByIndex(int index)
        {
            EnsureStorage();
            return index >= 0 && index < cells.Length
                ? cells[index]
                : OccupancyCellState.Unknown;
        }

        public bool TryGetCell(int x, int y, out OccupancyCellState state)
        {
            EnsureStorage();
            var index = GetFlatIndex(x, y);
            if (index < 0)
            {
                state = OccupancyCellState.Unknown;
                return false;
            }

            state = cells[index];
            return true;
        }

        public short GetEvidence(int x, int y)
        {
            EnsureStorage();
            var index = GetFlatIndex(x, y);
            return index >= 0 ? evidence[index] : (short)0;
        }

        public bool MarkFree(int x, int y)
        {
            return ApplyObservation(x, y, false);
        }

        public bool MarkOccupied(int x, int y)
        {
            return ApplyObservation(x, y, true);
        }

        internal bool MarkFreeByIndex(int index)
        {
            return ApplyObservationByIndex(index, false);
        }

        internal bool MarkOccupiedByIndex(int index)
        {
            return ApplyObservationByIndex(index, true);
        }

        public void BeginBatchUpdate()
        {
            EnsureStorage();
            if (batchDepth == 0)
            {
                changedCellIndices.Clear();
            }

            batchDepth++;
        }

        public int EndBatchUpdate()
        {
            if (batchDepth <= 0)
            {
                return 0;
            }

            batchDepth--;
            if (batchDepth > 0)
            {
                return changedCellIndices.Count;
            }

            if (changedCellIndices.Count > 0)
            {
                Version++;
                if (CellsChanged != null)
                {
                    CellsChanged(this, changedCellIndices);
                }
            }

            return changedCellIndices.Count;
        }

        public void CopyCellsTo(OccupancyCellState[] destination)
        {
            EnsureStorage();
            if (destination == null || destination.Length < cells.Length)
            {
                throw new ArgumentException("Destination must contain at least CellCount elements.", "destination");
            }

            Array.Copy(cells, destination, cells.Length);
        }

        public void CopyCellValuesTo(sbyte[] destination)
        {
            EnsureStorage();
            if (destination == null || destination.Length < cells.Length)
            {
                throw new ArgumentException("Destination must contain at least CellCount elements.", "destination");
            }

            for (var i = 0; i < cells.Length; i++)
            {
                destination[i] = (sbyte)cells[i];
            }
        }

        public void ResetGrid()
        {
            ClampConfiguration();
            var requiredCellCount = width * height;
            cells = new OccupancyCellState[requiredCellCount];
            evidence = new short[requiredCellCount];

            for (var i = 0; i < requiredCellCount; i++)
            {
                cells[i] = OccupancyCellState.Unknown;
            }

            UnknownCellCount = requiredCellCount;
            FreeCellCount = 0;
            OccupiedCellCount = 0;
            batchDepth = 0;
            changedCellIndices.Clear();
            Version++;

            if (GridReset != null)
            {
                GridReset(this);
            }
        }

        private void Awake()
        {
            EnsureStorage();
        }

        private void OnValidate()
        {
            ClampConfiguration();
            if (Application.isPlaying && (cells == null || cells.Length != width * height))
            {
                ResetGrid();
            }
        }

        private bool ApplyObservation(int x, int y, bool occupied)
        {
            var index = GetFlatIndex(x, y);
            return ApplyObservationByIndex(index, occupied);
        }

        private bool ApplyObservationByIndex(int index, bool occupied)
        {
            EnsureStorage();
            if (index < 0 || index >= cells.Length)
            {
                return false;
            }

            var autoBatch = batchDepth == 0;
            if (autoBatch)
            {
                BeginBatchUpdate();
            }

            var previousState = cells[index];
            var increment = occupied ? occupiedEvidenceIncrement : -freeEvidenceDecrement;
            var updatedEvidence = Mathf.Clamp(
                evidence[index] + increment,
                minimumEvidence,
                maximumEvidence);
            evidence[index] = (short)updatedEvidence;

            var updatedState = updatedEvidence >= occupiedThreshold
                ? OccupancyCellState.Occupied
                : OccupancyCellState.Free;
            var stateChanged = previousState != updatedState;

            if (stateChanged)
            {
                DecrementStateCount(previousState);
                cells[index] = updatedState;
                IncrementStateCount(updatedState);
                changedCellIndices.Add(index);
            }

            if (autoBatch)
            {
                EndBatchUpdate();
            }

            return stateChanged;
        }

        private void EnsureStorage()
        {
            ClampConfiguration();
            if (cells == null || evidence == null || cells.Length != width * height || evidence.Length != width * height)
            {
                ResetGrid();
            }
        }

        private void ClampConfiguration()
        {
            resolution = Mathf.Max(0.01f, resolution);
            width = Mathf.Max(1, width);
            height = Mathf.Max(1, height);
            occupiedThreshold = Mathf.Max(1, occupiedThreshold);
            occupiedEvidenceIncrement = Mathf.Max(occupiedThreshold, occupiedEvidenceIncrement);
            freeEvidenceDecrement = Mathf.Max(1, freeEvidenceDecrement);
            minimumEvidence = Mathf.Min(-1, minimumEvidence);
            maximumEvidence = Mathf.Max(occupiedThreshold, maximumEvidence);
        }

        private void DecrementStateCount(OccupancyCellState state)
        {
            switch (state)
            {
                case OccupancyCellState.Unknown:
                    UnknownCellCount--;
                    break;
                case OccupancyCellState.Free:
                    FreeCellCount--;
                    break;
                case OccupancyCellState.Occupied:
                    OccupiedCellCount--;
                    break;
            }
        }

        private void IncrementStateCount(OccupancyCellState state)
        {
            switch (state)
            {
                case OccupancyCellState.Unknown:
                    UnknownCellCount++;
                    break;
                case OccupancyCellState.Free:
                    FreeCellCount++;
                    break;
                case OccupancyCellState.Occupied:
                    OccupiedCellCount++;
                    break;
            }
        }

        private void OnDrawGizmosSelected()
        {
            if (!drawDebugBounds)
            {
                return;
            }

            var size = new Vector3(width * resolution, 0.01f, height * resolution);
            var center = mapOrigin + new Vector3(size.x * 0.5f, 0f, size.z * 0.5f);
            Gizmos.color = debugBoundsColor;
            Gizmos.DrawWireCube(center, size);
        }
    }
}
