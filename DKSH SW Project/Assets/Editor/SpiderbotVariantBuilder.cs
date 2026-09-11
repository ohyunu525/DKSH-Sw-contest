#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.SceneManagement;

namespace DKSH.Spiderbot.Editor
{
    // The Python builder preserves the CAD meshes and supplies calibrated local
    // geometry. Unity receives the same link tree as the URDF and USD outputs.
    public static class SpiderbotVariantBuilder
    {
        [Serializable] private sealed class AssemblyData
        {
            public int legCount;
            public float spawnHeight;
            public NodeData[] nodes;
            public MeshData[] meshes;
        }
        [Serializable] private sealed class NodeData
        {
            public string name, parent, mesh, axis;
            public float[] position, centerOfMass, inertia;
            public float yaw, mass;
            public ColliderData[] colliders;
        }
        [Serializable] private sealed class ColliderData
        {
            public float[] position, size;
            public float yaw;
        }
        [Serializable] private sealed class MeshData
        {
            public string name;
            public float[] vertices, color;
            public int[] triangles;
        }

        private const string Output = "Assets/Prefabs/spiderbot/Variants";

        // Reflection (x,y,z) -> (x,z,y): metres, Y-up Unity coordinates.
        private static Vector3 UnityVector(float[] v) => new Vector3(v[0], v[2], v[1]);
        private static Quaternion UnityYaw(float radians) => Quaternion.AngleAxis(-radians * Mathf.Rad2Deg, Vector3.up);

        [MenuItem("DKSH/Spiderbot/Build 6 and 8 Leg Models")]
        public static void Build()
        {
            string repository = Path.GetFullPath(Path.Combine(Application.dataPath, "../.."));
            string[] args = Environment.GetCommandLineArgs();
            for (int i = 0; i + 1 < args.Length; i++)
                if (args[i] == "-spiderbotRepository") repository = args[i + 1];
            EnsureFolder(Output);
            foreach (int count in new[] { 8, 6 })
            {
                string source = Path.Combine(repository, "outputs/spiderbot-unity", $"spiderbot_{count}leg.json");
                if (!File.Exists(source)) throw new FileNotFoundException("Run scripts/build_spiderbot_variants.py first", source);
                AssemblyData data = JsonUtility.FromJson<AssemblyData>(File.ReadAllText(source));
                BuildVariant(data);
            }
            AssetDatabase.SaveAssets();
            Debug.Log("DKSH_SPIDERBOT_VARIANTS_PASS: 8 legs / 24 joints; 6 legs / 18 joints.");
        }

        private static void BuildVariant(AssemblyData data)
        {
            string folder = $"{Output}/Spiderbot{data.legCount}Leg";
            EnsureFolder(folder);
            var meshes = new Dictionary<string, Mesh>();
            var materials = new Dictionary<string, Material>();
            foreach (MeshData source in data.meshes)
            {
                var mesh = new Mesh { name = source.name, indexFormat = IndexFormat.UInt32 };
                var vertices = new Vector3[source.vertices.Length / 3];
                for (int i = 0; i < vertices.Length; i++)
                    vertices[i] = new Vector3(source.vertices[3*i], source.vertices[3*i+2], source.vertices[3*i+1]);
                mesh.vertices = vertices;
                int[] triangles = (int[])source.triangles.Clone();
                for (int i = 0; i < triangles.Length; i += 3)
                    (triangles[i+1], triangles[i+2]) = (triangles[i+2], triangles[i+1]);
                mesh.triangles = triangles;
                mesh.RecalculateNormals();
                mesh.RecalculateBounds();
                meshes[source.name] = SaveOrUpdate(mesh, $"{folder}/{source.name}.asset");
                Shader shader = Shader.Find("Universal Render Pipeline/Lit") ?? Shader.Find("Standard");
                if (shader == null) throw new InvalidOperationException("No supported surface shader");
                var material = new Material(shader) { name = source.name };
                var color = new Color(source.color[0], source.color[1], source.color[2]);
                material.color = color;
                if (material.HasProperty("_BaseColor")) material.SetColor("_BaseColor", color);
                if (material.HasProperty("_Smoothness")) material.SetFloat("_Smoothness", 0.3f);
                materials[source.name] = SaveOrUpdate(material, $"{folder}/{source.name}.mat");
            }

            // Build in an isolated preview scene so an open user scene is never changed.
            Scene scene = EditorSceneManager.NewPreviewScene();
            GameObject root = new GameObject($"Spiderbot{data.legCount}Leg");
            SceneManager.MoveGameObjectToScene(root, scene);
            try
            {
                var links = new Dictionary<string, Transform>();
                foreach (NodeData node in data.nodes)
                {
                    GameObject link = node.parent.Length == 0 ? root : new GameObject(node.name);
                    if (node.parent.Length > 0)
                    {
                        link.transform.SetParent(links[node.parent], false);
                        link.transform.localPosition = UnityVector(node.position);
                        link.transform.localRotation = UnityYaw(node.yaw);
                    }
                    links[node.name] = link.transform;
                    var body = link.AddComponent<ArticulationBody>();
                    body.mass = node.mass;
                    body.useGravity = true;
                    body.immovable = false;
                    body.linearDamping = 0.05f;
                    body.angularDamping = 0.05f;
                    body.jointFriction = 0.01f;
                    body.solverIterations = 8;
                    body.solverVelocityIterations = 2;
                    if (node.parent.Length > 0)
                    {
                        body.jointType = ArticulationJointType.RevoluteJoint;
                        body.twistLock = ArticulationDofLock.LimitedMotion;
                        body.matchAnchors = false;
                        // A reflected right-handed rotation reverses the axis.
                        Vector3 axis = node.axis == "Z" ? Vector3.down : Vector3.back;
                        body.anchorPosition = Vector3.zero;
                        body.anchorRotation = Quaternion.FromToRotation(Vector3.right, axis);
                        body.parentAnchorPosition = link.transform.localPosition;
                        body.parentAnchorRotation = link.transform.localRotation * body.anchorRotation;
                        body.xDrive = new ArticulationDrive {
                            lowerLimit = -90, upperLimit = 90, stiffness = 8,
                            damping = 0.35f, forceLimit = 0.980665f, target = 0
                        };
                        body.maxJointVelocity = 400 * Mathf.Deg2Rad;
                    }
                    GameObject visual = new GameObject("Visual");
                    visual.transform.SetParent(link.transform, false);
                    visual.AddComponent<MeshFilter>().sharedMesh = meshes[node.mesh];
                    visual.AddComponent<MeshRenderer>().sharedMaterial = materials[node.mesh];
                    for (int i = 0; i < node.colliders.Length; i++)
                    {
                        ColliderData spec = node.colliders[i];
                        GameObject collision = new GameObject($"Collision_{i}");
                        collision.transform.SetParent(link.transform, false);
                        collision.transform.localPosition = UnityVector(spec.position);
                        collision.transform.localRotation = UnityYaw(spec.yaw);
                        collision.AddComponent<BoxCollider>().size = UnityVector(spec.size);
                    }
                    body.automaticCenterOfMass = false;
                    body.centerOfMass = UnityVector(node.centerOfMass);
                    body.automaticInertiaTensor = false;
                    body.inertiaTensor = UnityVector(node.inertia);
                    body.inertiaTensorRotation = Quaternion.identity;
                }
                root.transform.position = new Vector3(0, data.spawnHeight, 0);
                string path = $"{folder}/Spiderbot{data.legCount}Leg.prefab";
                PrefabUtility.SaveAsPrefabAsset(root, path, out bool success);
                if (!success) throw new InvalidOperationException("Could not save " + path);
                var saved = AssetDatabase.LoadAssetAtPath<GameObject>(path);
                var bodies = saved.GetComponentsInChildren<ArticulationBody>();
                if (bodies.Length != 1 + data.legCount * 3)
                    throw new InvalidOperationException("Unexpected articulation count in " + path);
                foreach (var renderer in saved.GetComponentsInChildren<MeshRenderer>())
                    if (renderer.sharedMaterial == null) throw new InvalidOperationException("Missing material");
                foreach (var filter in saved.GetComponentsInChildren<MeshFilter>())
                    if (filter.sharedMesh == null || filter.sharedMesh.vertexCount == 0)
                        throw new InvalidOperationException("Missing mesh");
                Debug.Log($"BUILT {path}: {bodies.Length} bodies, {data.legCount * 3} hinges");
            }
            finally { EditorSceneManager.ClosePreviewScene(scene); }
        }

        private static T SaveOrUpdate<T>(T generated, string path) where T : UnityEngine.Object
        {
            T existing = AssetDatabase.LoadAssetAtPath<T>(path);
            if (existing == null)
            {
                AssetDatabase.CreateAsset(generated, path);
                return generated;
            }
            EditorUtility.CopySerialized(generated, existing);
            UnityEngine.Object.DestroyImmediate(generated);
            EditorUtility.SetDirty(existing);
            return existing;
        }

        private static void EnsureFolder(string path)
        {
            if (AssetDatabase.IsValidFolder(path)) return;
            string parent = Path.GetDirectoryName(path).Replace('\\', '/');
            EnsureFolder(parent);
            AssetDatabase.CreateFolder(parent, Path.GetFileName(path));
        }
    }
}
#endif
