using System;
using System.IO;
using UnityEngine;
using UnityEngine.UIElements;

namespace DKSH.Spiderbot.RemoteNavigation
{
    [DisallowMultipleComponent, RequireComponent(typeof(UIDocument))]
    public sealed class RemoteNavigationApp : MonoBehaviour
    {
        [SerializeField] private string gatewayHost = "127.0.0.1";
        [SerializeField] private int gatewayPort = 8765;
        [SerializeField] private string gatewayToken = "";
        [SerializeField] private bool startWithGateway;
        private INavigationClient client;
        private readonly WaypointEditor editor = new WaypointEditor();
        private OccupancyGridView view;
        private Label stateLabel, detailLabel, positionLabel, progressLabel, modeLabel, hintLabel, mapLabel;
        private Button execute, cancel, stop, reset, linkButton, saveDraft, loadDraft;
        private TextField hostField, tokenField;
        private IntegerField portField;
        private VisualElement rows;
        private ScrollView sidebar;
        private VisualElement connectionCard;
        private Waypoint[] executingGoals = Array.Empty<Waypoint>();
        private bool built;
        private bool previousRunInBackground, backgroundConfigured;
        public INavigationClient Client => client;
        public WaypointEditor Editor => editor;
        public OccupancyGridView MapView => view;
        public string SavedDraftPath => Path.Combine(Application.persistentDataPath, "Missions", "last-mission.json");

        private void OnEnable()
        {
            if (!Application.isPlaying) return;
            previousRunInBackground = Application.runInBackground;
            Application.runInBackground = true; backgroundConfigured = true;
            BuildInterface();
            if (startWithGateway) ConnectGateway(); else SwitchToLocal();
        }
        private void OnDisable()
        {
            client?.Dispose(); client = null; view?.Dispose(); view = null; built = false;
            if (backgroundConfigured) Application.runInBackground = previousRunInBackground;
            backgroundConfigured = false;
        }
        private void Update()
        {
            if (!built || client == null) return;
            client.Tick(Time.unscaledDeltaTime);
            var status = client.Status;
            if (editor.Points.Count > 0 && client.Map != null && editor.MapRevision != client.Map.Revision)
            { editor.Clear(); RefreshRows(); hintLabel.text = "Map changed. Select new waypoints."; }
            bool busy = status.state == "Sent" || status.state == "Accepted" || status.state == "Navigating" || status.state == "Canceling" || status.state == "StopUnconfirmed";
            view.EditingEnabled = client.CanSubmit;
            view.Bind(client.Map, status, busy ? executingGoals : editor.Points);
            stateLabel.text = status.state ?? "Waiting";
            stateLabel.EnableInClassList("warning", status.stop_latched || !client.Connected);
            detailLabel.text = status.detail ?? "Waiting for robot state.";
            positionLabel.text = $"x {status.x:F2} m    y {status.y:F2} m\nheading {status.yaw * Mathf.Rad2Deg:F0}°";
            progressLabel.text = $"{Mathf.Min(status.waypoint_index, status.waypoint_count)} / {status.waypoint_count} reached\n{status.distance_remaining:F2} m remaining in this leg";
            mapLabel.text = client.Map == null ? "Waiting for map" :
                $"map  ·  {client.Map.Width * client.Map.Resolution:F1} × {client.Map.Height * client.Map.Resolution:F1} m  ·  {client.Map.Resolution:F2} m / cell";
            execute.SetEnabled(client.CanSubmit && editor.Points.Count > 0);
            saveDraft.SetEnabled(client.CanSubmit && editor.Points.Count > 0);
            loadDraft.SetEnabled(client.CanSubmit);
            cancel.SetEnabled(client.Connected && busy);
            stop.SetEnabled(client.Connected);
            reset.SetEnabled(client.Connected && status.ready && status.stop_latched && !busy && status.state != "StopUnconfirmed");
            if (client is SimulatedNavigationClient local)
            { linkButton.text = local.Connected ? "Test link loss" : "Restore link"; linkButton.SetEnabled(true); }
            else linkButton.SetEnabled(false);
        }

        public void SwitchToLocal()
        {
            client?.Dispose(); client = new SimulatedNavigationClient();
            editor.Clear(); executingGoals = Array.Empty<Waypoint>();
            if (!built) return;
            modeLabel.text = "LOCAL SIMULATION";
            hintLabel.text = "Click a free area. Drag to set heading. Execute when ready."; RefreshRows();
        }

        public void ConnectGateway()
        {
            string host = hostField.value.Trim(); int port = portField.value;
            if (string.IsNullOrEmpty(host) || port < 1 || port > 65535)
            { hintLabel.text = "Enter a host and a port between 1 and 65535."; return; }
            if (host != "127.0.0.1" && host != "localhost" && host != "::1" && tokenField.value.Length < 16)
            { hintLabel.text = "A remote gateway requires a shared token of at least 16 characters."; return; }
            client?.Dispose();
            client = new TcpNavigationClient(host, port, tokenField.value);
            editor.Clear(); executingGoals = Array.Empty<Waypoint>(); RefreshRows();
            modeLabel.text = "REMOTE SIMULATION"; hintLabel.text = "Waiting for the gateway map and robot state.";
        }

        public bool ExecuteDraft()
        {
            if (client == null || !client.Submit(editor.Points, editor.MapRevision))
            { if (built) hintLabel.text = client?.Status.detail ?? "Mission was not sent."; return false; }
            executingGoals = editor.Copy();
            hintLabel.text = "Mission submitted. Robot state shows its progress.";
            return true;
        }

        public bool SaveDraft()
        {
            if (client == null || !client.CanSubmit || editor.MapRevision != client.Map?.Revision) return false;
            bool saved = MissionDraftStore.Save(SavedDraftPath, client.Map, editor.Points, client.Status.footprint_radius, out string reason);
            if (built) hintLabel.text = reason;
            return saved;
        }

        public bool LoadDraft()
        {
            if (client == null || !client.CanSubmit) return false;
            bool loaded = MissionDraftStore.Load(SavedDraftPath, client.Map, client.Status.footprint_radius, editor, out string reason);
            if (built) { hintLabel.text = reason; if (loaded) RefreshRows(); }
            return loaded;
        }

        public void AddWaypoint(Waypoint point)
        {
            if (client?.Map == null || !client.CanSubmit) return;
            if (point == null || !NavigationMap.Finite(point.yaw) ||
                !client.Map.Validate(point.Position, client.Status.footprint_radius, out _))
            { hintLabel.text = "The waypoint needs free space and sufficient clearance."; return; }
            if (!editor.Add(point, client.Map.Revision)) { hintLabel.text = "A mission can contain up to 10 waypoints."; return; }
            hintLabel.text = $"Waypoint {editor.Points.Count}: x {point.x:F2}, y {point.y:F2}, heading {point.yaw * Mathf.Rad2Deg:F0}°";
            RefreshRows();
        }

        private void BuildInterface()
        {
            var root = GetComponent<UIDocument>().rootVisualElement;
            root.Clear(); root.AddToClassList("app");
            var stylesheet = Resources.Load<StyleSheet>("RemoteNavigation");
            if (stylesheet != null) root.styleSheets.Add(stylesheet);
            root.style.unityFont = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");

            var header = Element(root, "header");
            var branding = Element(header, "branding");
            Label(branding, "DKSH / SPIDERBOT", "eyebrow");
            Label(branding, "Waypoint control", "title");
            var sourceControls = Element(header, "source-controls");
            modeLabel = Label(sourceControls, "LOCAL SIMULATION", "mode-badge");
            Button(sourceControls, "Connection settings", () => sidebar.ScrollTo(connectionCard), "small-button");
            var body = Element(root, "body");
            var mapColumn = Element(body, "map-column");
            var toolbar = Element(mapColumn, "map-toolbar");
            mapLabel = Label(toolbar, "Map", "map-meta");
            Button(toolbar, "Fit map", () => view.FitMap(), "small-button");
            view = new OccupancyGridView(); mapColumn.Add(view);
            view.GoalSelected += AddWaypoint;
            view.HintChanged += text => hintLabel.text = text;
            var legend = Element(mapColumn, "legend");
            Label(legend, "● robot   ● waypoint   ━ route", "legend-colors");
            Label(legend, "Wheel: zoom   ·   Middle drag: pan", "muted");
            hintLabel = Label(mapColumn, "", "hint");
            var sidebarColumn = Element(body, "sidebar-column");
            sidebar = new ScrollView(ScrollViewMode.Vertical);
            sidebar.AddToClassList("sidebar"); sidebarColumn.Add(sidebar);
            var statusCard = Element(sidebar, "card");
            Label(statusCard, "ROBOT STATUS", "eyebrow");
            stateLabel = Label(statusCard, "Idle", "state");
            detailLabel = Label(statusCard, "", "detail");
            var metrics = Element(statusCard, "metrics");
            positionLabel = Label(metrics, "", "position");
            progressLabel = Label(metrics, "", "muted");
            var missionCard = Element(sidebar, "card", "mission-card");
            var row = Element(missionCard, "section-row");
            Label(row, "WAYPOINTS", "eyebrow");
            Button(row, "Clear", () => { if (client.CanSubmit) { editor.Clear(); RefreshRows(); } }, "small-button");
            var append = new Toggle("Append waypoints") { value = false };
            append.RegisterValueChangedCallback(evt => editor.AppendMode = evt.newValue); missionCard.Add(append);
            var scroll = new ScrollView(ScrollViewMode.Vertical); scroll.AddToClassList("waypoint-scroll"); missionCard.Add(scroll);
            rows = scroll.contentContainer;
            var draftActions = Element(missionCard, "section-row", "draft-actions");
            saveDraft = Button(draftActions, "Save draft", () => SaveDraft(), "small-button");
            loadDraft = Button(draftActions, "Load draft", () => LoadDraft(), "small-button");
            execute = Button(missionCard, "Execute mission", () => ExecuteDraft(), "primary-button");
            cancel = Button(missionCard, "Cancel mission", () => client.Cancel(), "secondary-button");
            var safetyControls = Element(sidebarColumn, "safety-controls");
            stop = Button(safetyControls, "STOP", () => client.Stop(), "stop-button");
            reset = Button(safetyControls, "Reset stop", () => client.ResetStop(), "secondary-button");
            connectionCard = Element(sidebar, "card", "connection-card");
            Label(connectionCard, "SIMULATION SOURCE", "eyebrow");
            hostField = new TextField("Host") { value = gatewayHost }; connectionCard.Add(hostField);
            portField = new IntegerField("Port") { value = gatewayPort }; connectionCard.Add(portField);
            tokenField = new TextField("Token") { value = gatewayToken, isPasswordField = true }; connectionCard.Add(tokenField);
            var connectionRow = Element(connectionCard, "section-row");
            Button(connectionRow, "Connect", ConnectGateway, "secondary-button");
            Button(connectionRow, "Local demo", SwitchToLocal, "secondary-button");
            linkButton = Button(connectionCard, "Test link loss", () =>
            {
                if (client is SimulatedNavigationClient local) local.SetLinkAvailable(!local.Connected);
            }, "small-button");
            Label(root, "SENSOR-FREE PREVIEW  ·  Stop applies to the virtual robot  ·  Reconnection never resumes a mission", "footer");
            built = true;
        }

        private void RefreshRows()
        {
            if (rows == null) return;
            rows.Clear();
            if (editor.Points.Count == 0) { Label(rows, "Select an open area on the map.", "empty"); return; }
            for (int i = 0; i < editor.Points.Count; i++)
            {
                int index = i; var point = editor.Points[i];
                var row = Element(rows, "waypoint-row");
                Label(row, $"{i + 1:00}", "waypoint-index");
                Label(row, $"{point.x:F2}, {point.y:F2} m\n{point.yaw * Mathf.Rad2Deg:F0}°", "waypoint-coordinate");
                Button(row, "↑", () => { if (client.CanSubmit) { editor.Move(index, -1); RefreshRows(); } }, "row-button");
                Button(row, "×", () => { if (client.CanSubmit) { editor.Remove(index); RefreshRows(); } }, "row-button");
            }
        }
        private static VisualElement Element(VisualElement parent, params string[] classes)
        {
            var element = new VisualElement(); foreach (string c in classes) element.AddToClassList(c);
            parent.Add(element); return element;
        }
        private static Label Label(VisualElement parent, string text, string className)
        {
            var label = new Label(text); label.AddToClassList(className); parent.Add(label); return label;
        }
        private static Button Button(VisualElement parent, string text, Action clicked, string className)
        {
            var button = new Button(clicked) { text = text }; button.AddToClassList(className); parent.Add(button); return button;
        }
    }
}
