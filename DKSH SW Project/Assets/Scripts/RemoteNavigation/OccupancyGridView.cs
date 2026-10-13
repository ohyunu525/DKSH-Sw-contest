using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UIElements;

namespace DKSH.Spiderbot.RemoteNavigation
{
    public sealed class MapViewport
    {
        public Rect Bounds;
        public float Zoom = 1;
        public Vector2 Pan;
        public Rect ImageRect(NavigationMap map)
        {
            float scale = Mathf.Min(Bounds.width / map.Width, Bounds.height / map.Height) * Zoom;
            Vector2 size = new Vector2(map.Width, map.Height) * scale;
            return new Rect(Bounds.center + Pan - size * .5f, size);
        }
        public Vector2 MapToScreen(NavigationMap map, Vector2 point)
        {
            Rect r = ImageRect(map); Vector2 p = map.MapToLocal(point);
            return new Vector2(r.x + p.x / (map.Width * map.Resolution) * r.width,
                r.yMax - p.y / (map.Height * map.Resolution) * r.height);
        }
        public bool ScreenToMap(NavigationMap map, Vector2 screen, out Vector2 point)
        {
            Rect r = ImageRect(map);
            point = map.LocalToMap(new Vector2((screen.x - r.x) / r.width * map.Width * map.Resolution,
                (r.yMax - screen.y) / r.height * map.Height * map.Resolution));
            return Bounds.Contains(screen) && r.Contains(screen);
        }
        public void ZoomAt(NavigationMap map, Vector2 cursor, float multiplier)
        {
            ScreenToMap(map, cursor, out Vector2 point);
            Zoom = Mathf.Clamp(Zoom * multiplier, .5f, 6);
            Pan += cursor - MapToScreen(map, point);
        }
        public void Fit() { Zoom = 1; Pan = Vector2.zero; }
    }

    public sealed class OccupancyGridView : VisualElement, IDisposable
    {
        public event Action<Waypoint> GoalSelected;
        public event Action<string> HintChanged;
        private readonly Image image;
        private readonly VisualElement overlay;
        private readonly List<Label> numbers = new List<Label>();
        private Texture2D texture;
        private NavigationMap map;
        private GatewayPacket status;
        private IReadOnlyList<Waypoint> goals;
        private Waypoint candidate;
        private bool dragging, panning;
        private Vector2 lastPointer;
        private int pointerId;
        public bool EditingEnabled { get; set; } = true;
        public readonly MapViewport Viewport = new MapViewport();
        private static readonly Color Cyan = new Color(.28f, .86f, .94f);
        private static readonly Color Amber = new Color(1, .73f, .29f);

        public OccupancyGridView()
        {
            name = "map-view"; AddToClassList("map-view");
            image = new Image { pickingMode = PickingMode.Ignore, scaleMode = ScaleMode.StretchToFill };
            image.style.position = Position.Absolute; Add(image);
            overlay = new VisualElement { pickingMode = PickingMode.Ignore };
            overlay.style.position = Position.Absolute;
            overlay.style.left = overlay.style.right = overlay.style.top = overlay.style.bottom = 0;
            overlay.generateVisualContent += Draw; Add(overlay);
            for (int i = 0; i < 10; i++)
            {
                var label = new Label((i + 1).ToString()) { pickingMode = PickingMode.Ignore };
                label.AddToClassList("map-number"); label.style.display = DisplayStyle.None;
                numbers.Add(label); Add(label);
            }
            RegisterCallback<GeometryChangedEvent>(_ => LayoutMap());
            RegisterCallback<PointerDownEvent>(OnDown);
            RegisterCallback<PointerMoveEvent>(OnMove);
            RegisterCallback<PointerUpEvent>(OnUp);
            RegisterCallback<PointerCaptureOutEvent>(_ => { dragging = false; panning = false; candidate = null; });
            RegisterCallback<WheelEvent>(evt =>
            {
                if (map == null) return;
                Viewport.ZoomAt(map, evt.localMousePosition, Mathf.Exp(-evt.delta.y * .07f));
                LayoutMap(); evt.StopPropagation();
            });
        }

        public void Bind(NavigationMap newMap, GatewayPacket newStatus, IReadOnlyList<Waypoint> newGoals)
        {
            if (map != newMap)
            {
                bool frameChanged = map == null || newMap == null || map.Revision != newMap.Revision;
                map = newMap; dragging = false; candidate = null;
                if (texture != null) UnityEngine.Object.Destroy(texture);
                texture = null;
                if (map != null)
                {
                    texture = new Texture2D(map.Width, map.Height, TextureFormat.RGBA32, false)
                    { filterMode = FilterMode.Point, wrapMode = TextureWrapMode.Clamp, name = "Remote occupancy map" };
                    var pixels = new Color32[map.Width * map.Height];
                    for (int y = 0; y < map.Height; y++)
                    for (int x = 0; x < map.Width; x++)
                    {
                        int value = map.Cell(x, y);
                        pixels[y * map.Width + x] = value < 0 ? new Color32(48, 61, 78, 255) :
                            value >= 50 ? new Color32(15, 24, 36, 255) :
                            ((x + y) % 2 == 0 ? new Color32(172, 190, 201, 255) : new Color32(169, 187, 198, 255));
                    }
                    texture.SetPixels32(pixels); texture.Apply();
                    if (frameChanged) Viewport.Fit();
                }
                image.image = texture; LayoutMap();
            }
            status = newStatus; goals = newGoals; LayoutNumbers(); overlay.MarkDirtyRepaint();
        }

        private void LayoutMap()
        {
            Viewport.Bounds = contentRect;
            if (map == null || contentRect.width <= 0 || contentRect.height <= 0) return;
            Rect r = Viewport.ImageRect(map);
            image.style.left = r.x; image.style.top = r.y; image.style.width = r.width; image.style.height = r.height;
            LayoutNumbers(); overlay.MarkDirtyRepaint();
        }
        private void LayoutNumbers()
        {
            for (int i = 0; i < numbers.Count; i++)
            {
                bool visible = map != null && goals != null && i < goals.Count;
                numbers[i].style.display = visible ? DisplayStyle.Flex : DisplayStyle.None;
                if (!visible) continue;
                Vector2 p = Viewport.MapToScreen(map, goals[i].Position);
                numbers[i].style.left = p.x + 12; numbers[i].style.top = p.y - 27;
            }
        }
        public void FitMap() { Viewport.Fit(); LayoutMap(); }

        private void OnDown(PointerDownEvent evt)
        {
            if (map == null) return;
            lastPointer = evt.localPosition; pointerId = evt.pointerId;
            if (evt.button == 2) panning = true;
            else if (evt.button == 0 && EditingEnabled &&
                Viewport.ScreenToMap(map, evt.localPosition, out Vector2 point))
            {
                float radius = status != null ? status.footprint_radius : .35f;
                if (!map.Validate(point, radius, out string reason)) { HintChanged?.Invoke(reason); return; }
                candidate = new Waypoint(point, status?.yaw ?? 0); dragging = true;
                HintChanged?.Invoke("Drag to set the arrival heading, then release.");
            }
            if (dragging || panning) { this.CapturePointer(pointerId); evt.StopPropagation(); }
            overlay.MarkDirtyRepaint();
        }
        private void OnMove(PointerMoveEvent evt)
        {
            if (map == null) return;
            Vector2 cursor = evt.localPosition;
            if (panning) { Viewport.Pan += cursor - lastPointer; LayoutMap(); }
            else if (dragging && candidate != null)
            {
                Vector2 start = Viewport.MapToScreen(map, candidate.Position);
                if ((cursor - start).sqrMagnitude > 64)
                {
                    Viewport.ScreenToMap(map, cursor, out Vector2 point);
                    Vector2 direction = point - candidate.Position;
                    candidate.yaw = Mathf.Atan2(direction.y, direction.x);
                }
                overlay.MarkDirtyRepaint();
            }
            lastPointer = cursor;
        }
        private void OnUp(PointerUpEvent evt)
        {
            if (evt.pointerId != pointerId) return;
            if (dragging && candidate != null && EditingEnabled && contentRect.Contains(evt.localPosition))
                GoalSelected?.Invoke(new Waypoint(candidate.Position, candidate.yaw));
            dragging = false; panning = false; candidate = null;
            this.ReleasePointer(pointerId); overlay.MarkDirtyRepaint();
        }

        private void Draw(MeshGenerationContext context)
        {
            if (map == null || status == null) return;
            Painter2D p = context.painter2D;
            if (status.path != null && status.path.Length > 1)
            {
                p.strokeColor = Cyan; p.lineWidth = 3; p.BeginPath();
                p.MoveTo(Viewport.MapToScreen(map, status.path[0].Position));
                for (int i = 1; i < status.path.Length; i++) p.LineTo(Viewport.MapToScreen(map, status.path[i].Position));
                p.Stroke();
            }
            if (goals != null)
                foreach (var goal in goals) DrawArrow(p, goal, Amber, 10);
            if (candidate != null) DrawArrow(p, candidate, Color.white, 12);
            var robot = new Waypoint(new Vector2(status.x, status.y), status.yaw);
            Vector2 centre = Viewport.MapToScreen(map, robot.Position);
            float pixels = Viewport.ImageRect(map).width / (map.Width * map.Resolution);
            p.strokeColor = new Color(Cyan.r, Cyan.g, Cyan.b, .4f); p.lineWidth = 1;
            p.BeginPath(); p.Arc(centre, Mathf.Max(8, status.footprint_radius * pixels), 0, 360); p.Stroke();
            for (int i = 0; i < 6; i++)
            {
                Vector2 leg = NavigationMap.Rotate(Vector2.right * 17, i * Mathf.PI / 3);
                p.strokeColor = Cyan; p.lineWidth = 2; p.BeginPath(); p.MoveTo(centre + leg * .4f); p.LineTo(centre + leg); p.Stroke();
            }
            DrawArrow(p, robot, Cyan, 8);
        }
        private void DrawArrow(Painter2D p, Waypoint point, Color color, float radius)
        {
            Vector2 c = Viewport.MapToScreen(map, point.Position);
            float angle = point.yaw - map.OriginYaw;
            Vector2 d = new Vector2(Mathf.Cos(angle), -Mathf.Sin(angle)), n = new Vector2(-d.y, d.x);
            p.fillColor = color; p.BeginPath(); p.Arc(c, radius, 0, 360); p.Fill();
            p.strokeColor = color; p.lineWidth = 3; p.BeginPath(); p.MoveTo(c); p.LineTo(c + d * 28); p.Stroke();
            p.BeginPath(); p.MoveTo(c + d * 32); p.LineTo(c + d * 21 + n * 6);
            p.LineTo(c + d * 21 - n * 6); p.ClosePath(); p.Fill();
        }
        public void Dispose() { if (texture != null) UnityEngine.Object.Destroy(texture); }
    }
}
