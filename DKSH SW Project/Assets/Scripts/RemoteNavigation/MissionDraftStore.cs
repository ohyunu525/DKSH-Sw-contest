using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;

namespace DKSH.Spiderbot.RemoteNavigation
{
    // Only coordinates are persisted. Loading never restores a session or sends a command.
    public static class MissionDraftStore
    {
        public const int MaximumBytes = 16384;

        [Serializable]
        private sealed class Document
        {
            public int version;
            public string map_revision;
            public int width, height;
            public float resolution, origin_x, origin_y, origin_yaw;
            public Waypoint[] waypoints;
        }

        public static bool Save(string path, NavigationMap map, IReadOnlyList<Waypoint> goals, float radius, out string reason)
        {
            var validated = new WaypointEditor();
            if (!validated.Replace(goals, map, radius, out reason)) return false;
            var document = new Document
            {
                version = 1, map_revision = map.Revision, width = map.Width, height = map.Height,
                resolution = map.Resolution, origin_x = map.Origin.x, origin_y = map.Origin.y,
                origin_yaw = map.OriginYaw, waypoints = validated.Copy()
            };
            string json = JsonUtility.ToJson(document, true);
            if (Encoding.UTF8.GetByteCount(json) > MaximumBytes)
            { reason = "The draft is too large to save."; return false; }
            string temporary = path + ".tmp";
            try
            {
                Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path)));
                File.WriteAllText(temporary, json, new UTF8Encoding(false));
                if (File.Exists(path)) File.Replace(temporary, path, null);
                else File.Move(temporary, path);
                reason = "Draft saved. Loading it requires the same map and a new Execute action.";
                return true;
            }
            catch (Exception ex) when (ex is IOException || ex is UnauthorizedAccessException || ex is ArgumentException || ex is NotSupportedException)
            {
                reason = "Could not save the draft: " + ex.Message; return false;
            }
            finally
            {
                try { if (File.Exists(temporary)) File.Delete(temporary); }
                catch (IOException) { }
                catch (UnauthorizedAccessException) { }
            }
        }

        public static bool Load(string path, NavigationMap map, float radius, WaypointEditor editor, out string reason)
        {
            reason = "Wait for a map before loading a draft.";
            if (map == null || editor == null) return false;
            try
            {
                // Bound reads even if a file is replaced while it is being opened.
                using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
                if (stream.Length > MaximumBytes) { reason = "The saved draft is too large."; return false; }
                var bytes = new byte[(int)stream.Length];
                int count = 0;
                while (count < bytes.Length)
                {
                    int read = stream.Read(bytes, count, bytes.Length - count);
                    if (read == 0) break;
                    count += read;
                }
                string json = new UTF8Encoding(false, true).GetString(bytes, 0, count);
                var document = JsonUtility.FromJson<Document>(json);
                if (document == null || document.version != 1)
                { reason = "Unsupported or invalid saved draft."; return false; }
                if (document.map_revision != map.Revision || document.width != map.Width || document.height != map.Height ||
                    document.resolution != map.Resolution || document.origin_x != map.Origin.x ||
                    document.origin_y != map.Origin.y || document.origin_yaw != map.OriginYaw)
                { reason = "Saved draft belongs to a different map. Select new waypoints."; return false; }
                if (!editor.Replace(document.waypoints, map, radius, out reason)) return false;
                reason = "Draft loaded and checked against the current map. Review it, then Execute.";
                return true;
            }
            catch (FileNotFoundException) { reason = "No saved draft yet. Select waypoints and Save draft first."; return false; }
            catch (DirectoryNotFoundException) { reason = "No saved draft yet. Select waypoints and Save draft first."; return false; }
            catch (Exception ex) when (ex is IOException || ex is UnauthorizedAccessException || ex is ArgumentException || ex is NotSupportedException)
            { reason = "Could not load the draft: " + ex.Message; return false; }
        }
    }
}
