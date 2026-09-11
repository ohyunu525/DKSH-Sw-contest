"""Serialize CAD variant assets for Unity without requiring an active Editor.

Mesh layout follows the uncompressed Mesh v10 layout already used by this
project. Articulation fields follow Unity's ArticulationBodyEditor. The C#
SpiderbotVariantBuilder can re-save and verify these with a licensed Editor.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import uuid

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'DKSH SW Project/Assets/Prefabs/spiderbot/Variants'
HEADER = '%YAML 1.1\n%TAG !u! tag:unity3d.com,2011:\n'
ZERO = {'fileID': 0}
IDENTITY = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}


def ref(file_id, guid=None):
    result = {'fileID': file_id}
    if guid:
        result.update(guid=guid, type=2)
    return result


def common(go=None):
    result = dict(m_ObjectHideFlags=0, m_CorrespondingSourceObject=ZERO,
                  m_PrefabInstance=ZERO, m_PrefabAsset=ZERO)
    if go is not None:
        result['m_GameObject'] = ref(go)
    return result


def vector(v):
    return {key: float(value) for key, value in zip('xyz', v)}


def unity_vector(v):
    return vector([v[0], v[2], v[1]])


def quaternion(rotation):
    return dict(zip('xyzw', (float(v) for v in rotation.as_quat())))


def yaw_rotation(angle):
    return Rotation.from_rotvec([0, -angle, 0])


def meta(path, importer='NativeFormatImporter', main_id=0):
    path = Path(path)
    meta_path = path.with_name(path.name + '.meta')
    if meta_path.exists():
        return re.search(r'^guid: ([a-f0-9]+)$', meta_path.read_text(), re.M)[1]
    guid = uuid.uuid5(uuid.NAMESPACE_URL, 'dksh-spiderbot/' + path.relative_to(ROOT).as_posix()).hex
    data = {'fileFormatVersion': 2, 'guid': guid}
    if path.is_dir():
        data['folderAsset'] = 'yes'
        importer = 'DefaultImporter'
    fields = {'externalObjects': {}}
    if main_id:
        fields['mainObjectFileID'] = main_id
    fields.update(userData='', assetBundleName='', assetBundleVariant='')
    data[importer] = fields
    meta_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')
    return guid


class Dumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


def document(kind, file_id, name, data):
    return f'--- !u!{kind} &{file_id}\n' + yaml.dump({name: data}, Dumper=Dumper,
        sort_keys=False, width=2**30, allow_unicode=True)


def write_mesh(path, source):
    vertices = np.array(source['vertices'], dtype='<f4').reshape(-1, 3)[:, [0, 2, 1]]
    faces = np.array(source['triangles'], dtype='<u4').reshape(-1, 3)[:, [0, 2, 1]]
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    vertex_bytes = np.column_stack([vertices, mesh.vertex_normals]).astype('<f4').tobytes()
    bounds = {'m_Center': vector(mesh.bounds.mean(axis=0)), 'm_Extent': vector(mesh.extents / 2)}
    channels = [{'stream': 0, 'offset': offset, 'format': 0, 'dimension': dimension}
                for offset, dimension in [(0, 3), (12, 3)] + [(0, 0)] * 12]
    data = dict(common(), m_Name=source['name'], serializedVersion=10,
                m_SubMeshes=[dict(serializedVersion=2, firstByte=0, indexCount=int(faces.size),
                    topology=0, baseVertex=0, firstVertex=0, vertexCount=len(vertices), localAABB=bounds)],
                m_Shapes=dict(vertices=[], shapes=[], channels=[], fullWeights=[]), m_BindPose=[],
                m_BoneNameHashes='', m_RootBoneNameHash=0, m_BonesAABB=[],
                m_VariableBoneCountWeights={'m_Data': ''}, m_MeshCompression=0, m_IsReadable=1,
                m_KeepVertices=1, m_KeepIndices=1, m_IndexFormat=1, m_IndexBuffer=faces.astype('<u4').tobytes().hex(),
                m_VertexData=dict(serializedVersion=3, m_VertexCount=len(vertices), m_Channels=channels,
                    m_DataSize=len(vertex_bytes), _typelessdata=vertex_bytes.hex()),
                m_LocalAABB=bounds, m_MeshUsageFlags=0, m_BakedConvexCollisionMesh='',
                m_BakedTriangleCollisionMesh='', m_MeshOptimizationFlags=1,
                m_StreamData=dict(offset=0, size=0, path=''))
    compressed = {}
    for name in ('Vertices', 'UV', 'Normals', 'Tangents', 'Weights', 'NormalSigns',
                 'TangentSigns', 'FloatColors', 'BoneIndices', 'Triangles'):
        compressed['m_' + name] = dict(m_NumItems=0, m_Range=0, m_Start=0, m_Data='', m_BitSize=0)
    compressed['m_UVInfo'] = 0
    data['m_CompressedMesh'] = compressed
    data['m_MeshMetrics[0]'] = data['m_MeshMetrics[1]'] = 1
    path.write_text(HEADER + document(43, 4300000, 'Mesh', data), encoding='utf-8')
    return meta(path, main_id=4300000)


def write_material(path, source):
    color = dict(zip('rgba', (*source['color'], 1)))
    # Official URP Lit shader GUID, also used by the project's installed URP.
    shader = {'fileID': 4800000, 'guid': '933532a4fcc9baf4fa0491de14d08ed7', 'type': 3}
    data = dict(common(), serializedVersion=8, m_Name=source['name'], m_Shader=shader,
                m_ValidKeywords=[], m_InvalidKeywords=[], m_LightmapFlags=4,
                m_EnableInstancingVariants=1, m_DoubleSidedGI=0, m_CustomRenderQueue=-1,
                stringTagMap={}, disabledShaderPasses=[], m_LockedProperties='',
                m_SavedProperties=dict(serializedVersion=3, m_TexEnvs=[], m_Ints=[],
                    m_Floats=[{'_Smoothness': 0.3}, {'_Metallic': 0.1}, {'_Surface': 0}, {'_ZWrite': 1}],
                    m_Colors=[{'_BaseColor': color}, {'_Color': color}]), m_BuildTextureStacks=[])
    path.write_text(HEADER + document(21, 2100000, 'Material', data), encoding='utf-8')
    return meta(path, main_id=2100000)


def build_prefab(folder, assembly, meshes, materials):
    objects = []
    transforms = {}
    next_id = 1000

    def game_object(name, parent=None, position=(0, 0, 0), rotation=IDENTITY):
        nonlocal next_id
        go, tr = next_id, next_id + 1
        next_id += 10
        obj = dict(common(), serializedVersion=6, m_Component=[{'component': ref(tr)}],
                   m_Layer=0, m_Name=name, m_TagString='Untagged', m_Icon=ZERO,
                   m_NavMeshLayer=0, m_StaticEditorFlags=0, m_IsActive=1)
        transform = dict(common(go), serializedVersion=2, m_LocalRotation=rotation,
                         m_LocalPosition=vector(position), m_LocalScale=vector((1, 1, 1)),
                         m_ConstrainProportionsScale=0, m_Children=[], m_Father=ref(parent or 0),
                         m_LocalEulerAnglesHint=vector((0, 0, 0)))
        objects.extend([(1, go, 'GameObject', obj), (4, tr, 'Transform', transform)])
        transforms[tr] = transform
        if parent:
            transforms[parent]['m_Children'].append(ref(tr))
        return go, tr, obj

    def component(kind, name, file_id, go, obj, values):
        obj['m_Component'].append({'component': ref(file_id)})
        objects.append((kind, file_id, name, dict(common(go), **values)))

    links = {}
    for node in assembly['nodes']:
        root = not node['parent']
        pos = [0, assembly['spawnHeight'], 0] if root else [node['position'][i] for i in (0, 2, 1)]
        rotation = yaw_rotation(node['yaw'])
        go, tr, obj = game_object(f'Spiderbot{assembly["legCount"]}Leg' if root else node['name'],
            links.get(node['parent']), pos, quaternion(rotation))
        links[node['name']] = tr
        # Align Unity's articulation X-axis to the reflected URDF hinge axis.
        anchor = Rotation.from_euler('z', -90, degrees=True) if node['axis'] == 'Z' else Rotation.from_euler('y', 90, degrees=True)
        active_drive = dict(lowerLimit=-90, upperLimit=90, stiffness=8, damping=0.35,
                            forceLimit=0.980665, target=0, targetVelocity=0, driveType=0)
        zero_drive = dict(lowerLimit=0, upperLimit=0, stiffness=0, damping=0,
                          forceLimit=0, target=0, targetVelocity=0, driveType=0)
        component(171741748, 'ArticulationBody', go+2, go, obj, dict(
            m_Enabled=1, serializedVersion=5, m_Mass=node['mass'], m_ImplicitCom=0,
            m_ImplicitTensor=0, m_CenterOfMass=unity_vector(node['centerOfMass']),
            m_InertiaTensor=unity_vector(node['inertia']), m_InertiaRotation=IDENTITY,
            m_ParentAnchorPosition=unity_vector(node['position']),
            m_ParentAnchorRotation=quaternion(rotation * anchor), m_AnchorPosition=vector((0, 0, 0)),
            m_AnchorRotation=quaternion(anchor), m_MatchAnchors=0, m_ArticulationJointType=0 if root else 2,
            m_LinearX=0, m_LinearY=0, m_LinearZ=0, m_SwingY=0, m_SwingZ=0, m_Twist=0 if root else 1,
            m_XDrive=active_drive, m_YDrive=zero_drive, m_ZDrive=zero_drive,
            m_LinearDamping=0.05, m_AngularDamping=0.05, m_JointFriction=0.01, m_Immovable=0,
            m_UseGravity=1, m_CollisionDetectionMode=0))
        component(33, 'MeshFilter', go+3, go, obj, {'m_Mesh': ref(4300000, meshes[node['mesh']])})
        component(23, 'MeshRenderer', go+4, go, obj, dict(m_Enabled=1, m_CastShadows=1,
            m_ReceiveShadows=1, m_DynamicOccludee=1, m_StaticShadowCaster=0, m_MotionVectors=1,
            m_LightProbeUsage=1, m_ReflectionProbeUsage=1, m_RayTracingMode=2,
            m_RenderingLayerMask=1, m_RendererPriority=0,
            m_Materials=[ref(2100000, materials[node['mesh']])],
            m_StaticBatchInfo=dict(firstSubMesh=0, subMeshCount=0), m_StaticBatchRoot=ZERO,
            m_ProbeAnchor=ZERO, m_LightProbeVolumeOverride=ZERO, m_ScaleInLightmap=1,
            m_ReceiveGI=1, m_SelectedEditorRenderState=3, m_SortingLayerID=0,
            m_SortingLayer=0, m_SortingOrder=0, m_AdditionalVertexStreams=ZERO))
        for index, collider in enumerate(node['colliders']):
            cgo, ctr, cobj = game_object(f'Collision_{index}', tr,
                [collider['position'][i] for i in (0, 2, 1)], quaternion(yaw_rotation(collider['yaw'])))
            component(65, 'BoxCollider', cgo+2, cgo, cobj, dict(m_Material=ZERO,
                m_IncludeLayers=dict(serializedVersion=2, m_Bits=0),
                m_ExcludeLayers=dict(serializedVersion=2, m_Bits=0), m_LayerOverridePriority=0,
                m_IsTrigger=0, m_ProvidesContacts=0, m_Enabled=1, serializedVersion=3,
                m_Size=unity_vector(collider['size']), m_Center=vector((0, 0, 0))))
    path = folder / f'Spiderbot{assembly["legCount"]}Leg.prefab'
    path.write_text(HEADER + ''.join(document(*item) for item in objects), encoding='utf-8')
    meta(path, importer='PrefabImporter')
    return path


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    meta(OUTPUT)
    for count in (8, 6):
        folder = OUTPUT / f'Spiderbot{count}Leg'
        folder.mkdir(exist_ok=True)
        meta(folder)
        assembly = json.loads((ROOT/'outputs/spiderbot-unity'/f'spiderbot_{count}leg.json').read_text())
        meshes, materials = {}, {}
        for source in assembly['meshes']:
            meshes[source['name']] = write_mesh(folder/f'{source["name"]}.asset', source)
            materials[source['name']] = write_material(folder/f'{source["name"]}.mat', source)
        print('UNITY_ASSET_READY', build_prefab(folder, assembly, meshes, materials))
    meta(ROOT/'DKSH SW Project/Assets/Editor/SpiderbotVariantBuilder.cs', importer='MonoImporter')


if __name__ == '__main__':
    main()
