using System.Collections.Generic;
using UnityEngine;

namespace DKSH.Spiderbot.Mapping
{
    [DisallowMultipleComponent]
    public sealed class OccupancyGrid2DRenderer : MonoBehaviour
    {
        [SerializeField]
        private OccupancyGrid2D occupancyGrid;

        [SerializeField]
        private Transform robotTransform;

        [SerializeField]
        private bool debugVisualization = true;

        [SerializeField]
        private Color unknownColor = new Color(0.35f, 0.35f, 0.35f, 1f);

        [SerializeField]
        private Color freeColor = new Color(0.92f, 0.92f, 0.88f, 1f);

        [SerializeField]
        private Color occupiedColor = new Color(0.04f, 0.04f, 0.04f, 1f);

        [SerializeField]
        private Color robotColor = new Color(1f, 0.15f, 0.05f, 1f);

        [SerializeField, Min(0f)]
        private float visualizationHeightOffset = 0.025f;

        [SerializeField, Min(0.02f)]
        private float robotMarkerSize = 0.3f;

        private const string VisualizationObjectName = "Occupancy Grid Visualization";
        private const string RobotMarkerObjectName = "Occupancy Grid Robot Marker";

        private OccupancyGrid2D subscribedGrid;
        private GameObject visualizationObject;
        private MeshRenderer visualizationRenderer;
        private Mesh visualizationMesh;
        private Texture2D gridTexture;
        private Material gridMaterial;
        private GameObject robotMarkerObject;
        private Material robotMaterial;
        private Mesh robotMesh;

        public Texture2D GridTexture { get { return gridTexture; } }
        public Transform RobotTransform { get { return robotTransform; } }

        public void Configure(OccupancyGrid2D grid, Transform trackedRobot)
        {
            occupancyGrid = grid;
            robotTransform = trackedRobot;
            SubscribeToGrid();
            RebuildVisualization();
        }

        public void RebuildVisualization()
        {
            ResolveReferences();
            if (occupancyGrid == null)
            {
                return;
            }

            EnsureVisualizationResources();
            RepaintAllCells();
            UpdateVisibility();
            UpdateRobotMarker();
        }

        private void Reset()
        {
            ResolveReferences();
        }

        private void Awake()
        {
            ResolveReferences();
        }

        private void OnEnable()
        {
            ResolveReferences();
            SubscribeToGrid();
            RebuildVisualization();
        }

        private void OnDisable()
        {
            UnsubscribeFromGrid();
        }

        private void OnDestroy()
        {
            UnsubscribeFromGrid();
            DestroyRuntimeObject(visualizationMesh);
            DestroyRuntimeObject(gridTexture);
            DestroyRuntimeObject(gridMaterial);
            DestroyRuntimeObject(robotMesh);
            DestroyRuntimeObject(robotMaterial);
            DestroyRuntimeObject(visualizationObject);
            DestroyRuntimeObject(robotMarkerObject);
        }

        private void LateUpdate()
        {
            ResolveReferences();
            SubscribeToGrid();
            UpdateVisibility();
            UpdateRobotMarker();
        }

        private void OnValidate()
        {
            visualizationHeightOffset = Mathf.Max(0f, visualizationHeightOffset);
            robotMarkerSize = Mathf.Max(0.02f, robotMarkerSize);
        }

        private void ResolveReferences()
        {
            if (occupancyGrid == null)
            {
                occupancyGrid = GetComponent<OccupancyGrid2D>();
            }

            if (occupancyGrid == null)
            {
                occupancyGrid = GetComponentInParent<OccupancyGrid2D>();
            }
        }

        private void SubscribeToGrid()
        {
            if (!isActiveAndEnabled || subscribedGrid == occupancyGrid)
            {
                return;
            }

            UnsubscribeFromGrid();
            subscribedGrid = occupancyGrid;
            if (subscribedGrid != null)
            {
                subscribedGrid.CellsChanged += HandleCellsChanged;
                subscribedGrid.GridReset += HandleGridReset;
            }
        }

        private void UnsubscribeFromGrid()
        {
            if (subscribedGrid == null)
            {
                return;
            }

            subscribedGrid.CellsChanged -= HandleCellsChanged;
            subscribedGrid.GridReset -= HandleGridReset;
            subscribedGrid = null;
        }

        private void HandleCellsChanged(OccupancyGrid2D grid, IReadOnlyList<int> changedIndices)
        {
            EnsureVisualizationResources();
            if (gridTexture == null)
            {
                return;
            }

            for (var i = 0; i < changedIndices.Count; i++)
            {
                var index = changedIndices[i];
                if (!grid.TryGetCoordinates(index, out var x, out var y))
                {
                    continue;
                }

                gridTexture.SetPixel(x, y, GetCellColor(grid.GetCellByIndex(index)));
            }

            gridTexture.Apply(false, false);
        }

        private void HandleGridReset(OccupancyGrid2D grid)
        {
            RebuildVisualization();
        }

        private void EnsureVisualizationResources()
        {
            if (occupancyGrid == null)
            {
                return;
            }

            EnsureTexture();
            EnsureGridMaterial();
            EnsureVisualizationObject();
            EnsureRobotMarker();
            UpdateGridMesh();
        }

        private void EnsureTexture()
        {
            if (gridTexture != null &&
                gridTexture.width == occupancyGrid.Width &&
                gridTexture.height == occupancyGrid.Height)
            {
                return;
            }

            DestroyRuntimeObject(gridTexture);
            gridTexture = new Texture2D(
                occupancyGrid.Width,
                occupancyGrid.Height,
                TextureFormat.RGBA32,
                false)
            {
                name = "Runtime Occupancy Grid Texture",
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp,
                hideFlags = HideFlags.DontSave
            };
        }

        private void EnsureGridMaterial()
        {
            if (gridMaterial == null)
            {
                var shader = Shader.Find("Universal Render Pipeline/Unlit");
                if (shader == null)
                {
                    shader = Shader.Find("Unlit/Texture");
                }

                if (shader == null)
                {
                    return;
                }

                gridMaterial = new Material(shader)
                {
                    name = "Runtime Occupancy Grid Material",
                    hideFlags = HideFlags.DontSave
                };
            }

            if (gridMaterial.HasProperty("_BaseMap"))
            {
                gridMaterial.SetTexture("_BaseMap", gridTexture);
            }

            if (gridMaterial.HasProperty("_MainTex"))
            {
                gridMaterial.SetTexture("_MainTex", gridTexture);
            }

            if (gridMaterial.HasProperty("_BaseColor"))
            {
                gridMaterial.SetColor("_BaseColor", Color.white);
            }

            if (gridMaterial.HasProperty("_Color"))
            {
                gridMaterial.SetColor("_Color", Color.white);
            }
        }

        private void EnsureVisualizationObject()
        {
            if (visualizationObject == null)
            {
                var existing = transform.Find(VisualizationObjectName);
                visualizationObject = existing != null ? existing.gameObject : null;
            }

            if (visualizationObject == null)
            {
                visualizationObject = new GameObject(VisualizationObjectName)
                {
                    hideFlags = HideFlags.DontSave
                };
                visualizationObject.transform.SetParent(transform, false);
            }

            var meshFilter = visualizationObject.GetComponent<MeshFilter>();
            if (meshFilter == null)
            {
                meshFilter = visualizationObject.AddComponent<MeshFilter>();
            }

            visualizationRenderer = visualizationObject.GetComponent<MeshRenderer>();
            if (visualizationRenderer == null)
            {
                visualizationRenderer = visualizationObject.AddComponent<MeshRenderer>();
            }

            if (visualizationMesh == null)
            {
                visualizationMesh = new Mesh
                {
                    name = "Runtime Occupancy Grid Mesh",
                    hideFlags = HideFlags.DontSave
                };
            }

            meshFilter.sharedMesh = visualizationMesh;
            visualizationRenderer.sharedMaterial = gridMaterial;
        }

        private void UpdateGridMesh()
        {
            if (visualizationMesh == null || occupancyGrid == null)
            {
                return;
            }

            var bottomLeft = transform.InverseTransformPoint(occupancyGrid.GridCornerToWorld(0, 0));
            var bottomRight = transform.InverseTransformPoint(occupancyGrid.GridCornerToWorld(occupancyGrid.Width, 0));
            var topRight = transform.InverseTransformPoint(
                occupancyGrid.GridCornerToWorld(occupancyGrid.Width, occupancyGrid.Height));
            var topLeft = transform.InverseTransformPoint(occupancyGrid.GridCornerToWorld(0, occupancyGrid.Height));
            var heightOffset = transform.InverseTransformVector(Vector3.up * visualizationHeightOffset);

            visualizationMesh.Clear();
            visualizationMesh.vertices = new[]
            {
                bottomLeft + heightOffset,
                bottomRight + heightOffset,
                topRight + heightOffset,
                topLeft + heightOffset
            };
            visualizationMesh.uv = new[]
            {
                new Vector2(0f, 0f),
                new Vector2(1f, 0f),
                new Vector2(1f, 1f),
                new Vector2(0f, 1f)
            };
            visualizationMesh.triangles = new[] { 0, 2, 1, 0, 3, 2 };
            visualizationMesh.RecalculateNormals();
            visualizationMesh.RecalculateBounds();
        }

        private void EnsureRobotMarker()
        {
            if (robotMarkerObject == null)
            {
                var existing = transform.Find(RobotMarkerObjectName);
                robotMarkerObject = existing != null ? existing.gameObject : null;
            }

            if (robotMarkerObject == null)
            {
                robotMarkerObject = new GameObject(RobotMarkerObjectName)
                {
                    hideFlags = HideFlags.DontSave
                };
                robotMarkerObject.transform.SetParent(transform, false);
            }

            var meshFilter = robotMarkerObject.GetComponent<MeshFilter>();
            if (meshFilter == null)
            {
                meshFilter = robotMarkerObject.AddComponent<MeshFilter>();
            }

            var meshRenderer = robotMarkerObject.GetComponent<MeshRenderer>();
            if (meshRenderer == null)
            {
                meshRenderer = robotMarkerObject.AddComponent<MeshRenderer>();
            }

            if (robotMesh == null)
            {
                robotMesh = new Mesh
                {
                    name = "Runtime Occupancy Grid Robot Marker Mesh",
                    hideFlags = HideFlags.DontSave,
                    vertices = new[]
                    {
                        new Vector3(0f, 0f, 0.5f),
                        new Vector3(0.5f, 0f, 0f),
                        new Vector3(0f, 0f, -0.5f),
                        new Vector3(-0.5f, 0f, 0f)
                    },
                    triangles = new[] { 0, 1, 2, 0, 2, 3 }
                };
                robotMesh.RecalculateNormals();
                robotMesh.RecalculateBounds();
            }

            if (robotMaterial == null)
            {
                var shader = Shader.Find("Universal Render Pipeline/Unlit");
                if (shader == null)
                {
                    shader = Shader.Find("Unlit/Color");
                }

                if (shader != null)
                {
                    robotMaterial = new Material(shader)
                    {
                        name = "Runtime Occupancy Grid Robot Material",
                        hideFlags = HideFlags.DontSave
                    };
                }
            }

            if (robotMaterial != null)
            {
                if (robotMaterial.HasProperty("_BaseColor"))
                {
                    robotMaterial.SetColor("_BaseColor", robotColor);
                }

                if (robotMaterial.HasProperty("_Color"))
                {
                    robotMaterial.SetColor("_Color", robotColor);
                }
            }

            meshFilter.sharedMesh = robotMesh;
            meshRenderer.sharedMaterial = robotMaterial;
        }

        private void RepaintAllCells()
        {
            if (occupancyGrid == null || gridTexture == null)
            {
                return;
            }

            var colors = new Color32[occupancyGrid.CellCount];
            var cells = occupancyGrid.Cells;
            for (var i = 0; i < cells.Count; i++)
            {
                colors[i] = GetCellColor(cells[i]);
            }

            gridTexture.SetPixels32(colors);
            gridTexture.Apply(false, false);
        }

        private Color GetCellColor(OccupancyCellState state)
        {
            switch (state)
            {
                case OccupancyCellState.Free:
                    return freeColor;
                case OccupancyCellState.Occupied:
                    return occupiedColor;
                default:
                    return unknownColor;
            }
        }

        private void UpdateVisibility()
        {
            if (visualizationRenderer != null)
            {
                visualizationRenderer.enabled = debugVisualization;
            }

            if (robotMarkerObject != null)
            {
                robotMarkerObject.SetActive(debugVisualization && robotTransform != null);
            }
        }

        private void UpdateRobotMarker()
        {
            if (robotMarkerObject == null || robotTransform == null || occupancyGrid == null)
            {
                return;
            }

            var robotPosition = robotTransform.position;
            robotMarkerObject.transform.position = new Vector3(
                robotPosition.x,
                occupancyGrid.MapOrigin.y + visualizationHeightOffset + 0.015f,
                robotPosition.z);
            robotMarkerObject.transform.rotation = Quaternion.Euler(
                0f,
                robotTransform.eulerAngles.y,
                0f);
            robotMarkerObject.transform.localScale = Vector3.one * robotMarkerSize;
        }

        private static void DestroyRuntimeObject(Object target)
        {
            if (target == null)
            {
                return;
            }

            if (Application.isPlaying)
            {
                Destroy(target);
            }
            else
            {
                DestroyImmediate(target);
            }
        }
    }
}
