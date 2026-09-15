"""Cross-check generated CAD assets without requiring Unity or Isaac Sim."""
from __future__ import annotations

import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdGeom, UsdPhysics
import yaml

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'isaaclab_project/assets/spiderbot_variants'
UNITY = ROOT / 'DKSH SW Project/Assets/Prefabs/spiderbot/Variants'


def unity_documents(path):
    text = path.read_text(encoding='utf-8')
    parts = re.split(r'^--- !u!(\d+) &(\d+)\s*$', text, flags=re.M)
    return {int(parts[i+1]): (int(parts[i]), yaml.safe_load(parts[i+2]))
            for i in range(1, len(parts), 3)}


def values(v, keys='xyz'):
    return np.array([v[k] for k in keys], dtype=float)


def matrix(position, rotation):
    m = np.eye(4)
    m[:3, :3] = rotation
    m[:3, 3] = position
    return m


def check(count):
    folder = ASSETS / f'spiderbot_{count}leg'
    urdf = ET.parse(folder / f'spiderbot_{count}leg.urdf').getroot()
    links = urdf.findall('link'); joints = urdf.findall('joint')
    assert len(links) == count * 3 + 1 and len(joints) == count * 3
    names = {link.attrib['name'] for link in links}
    children = {j.find('child').attrib['link'] for j in joints}
    assert names - children == {'base'}
    for link in links:
        mesh_path = link.find('visual/geometry/mesh').attrib['filename']
        assert (folder / mesh_path).is_file()
        assert float(link.find('inertial/mass').attrib['value']) > 0

    stage = Usd.Stage.Open(str(folder / f'spiderbot_{count}leg.usd'))
    rigid = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    hinges = [UsdPhysics.RevoluteJoint(p) for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
    assert len(rigid) == count * 3 + 1 and len(hinges) == count * 3
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    cache = UsdGeom.XformCache()
    for joint in hinges:
        a = stage.GetPrimAtPath(joint.GetBody0Rel().GetTargets()[0])
        b = stage.GetPrimAtPath(joint.GetBody1Rel().GetTargets()[0])
        assert a and b
        pos_a = cache.GetLocalToWorldTransform(a).Transform(joint.GetLocalPos0Attr().Get())
        pos_b = cache.GetLocalToWorldTransform(b).Transform(joint.GetLocalPos1Attr().Get())
        assert np.allclose(pos_a, pos_b, atol=1e-7), joint.GetPath()
    for prim in rigid:
        visual = UsdGeom.Mesh(stage.GetPrimAtPath(str(prim.GetPath()) + '/visual'))
        assert visual and len(visual.GetPointsAttr().Get()) > 0

    prefab_folder = UNITY / f'Spiderbot{count}Leg'
    docs = unity_documents(prefab_folder / f'Spiderbot{count}Leg.prefab')
    assert sum(k == 171741748 for k, _ in docs.values()) == count * 3 + 1
    hinges_unity = [d['ArticulationBody'] for k, d in docs.values()
                    if k == 171741748 and d['ArticulationBody']['m_ArticulationJointType'] == 2]
    assert len(hinges_unity) == count * 3
    guids = {yaml.safe_load(p.read_text())['guid']: p.with_suffix('') for p in prefab_folder.glob('*.meta')}

    def check_references(obj):
        if isinstance(obj, dict):
            if 'fileID' in obj and obj['fileID']:
                if 'guid' in obj:
                    assert obj['guid'] in guids, obj
                    assert guids[obj['guid']].is_file()
                else:
                    assert obj['fileID'] in docs, obj
            for item in obj.values():
                check_references(item)
        elif isinstance(obj, list):
            for item in obj:
                check_references(item)
    for _, data in docs.values():
        check_references(data)

    transforms = {key: data['Transform'] for key, (kind, data) in docs.items() if kind == 4}
    by_go = {t['m_GameObject']['fileID']: key for key, t in transforms.items()}
    world = {}

    def world_transform(key):
        if key not in world:
            t = transforms[key]
            m = matrix(values(t['m_LocalPosition']), Rotation.from_quat(values(t['m_LocalRotation'], 'xyzw')).as_matrix())
            parent = t['m_Father']['fileID']
            world[key] = world_transform(parent) @ m if parent else m
        return world[key]

    for body in hinges_unity:
        tr = by_go[body['m_GameObject']['fileID']]
        parent = transforms[tr]['m_Father']['fileID']
        frame_a = world_transform(parent) @ matrix(values(body['m_ParentAnchorPosition']),
            Rotation.from_quat(values(body['m_ParentAnchorRotation'], 'xyzw')).as_matrix())
        frame_b = world_transform(tr) @ matrix(values(body['m_AnchorPosition']),
            Rotation.from_quat(values(body['m_AnchorRotation'], 'xyzw')).as_matrix())
        assert np.allclose(frame_a, frame_b, atol=1e-7), tr

    for path in prefab_folder.glob('*.asset'):
        data = next(iter(unity_documents(path).values()))[1]['Mesh']
        vertex_data = data['m_VertexData']
        vertex_bytes = bytes.fromhex(vertex_data['_typelessdata'])
        assert len(vertex_bytes) == vertex_data['m_DataSize'] == vertex_data['m_VertexCount'] * 24
        vertex_array = np.frombuffer(vertex_bytes, '<f4').reshape(-1, 6)
        faces = np.frombuffer(bytes.fromhex(data['m_IndexBuffer']), '<u4')
        assert np.isfinite(vertex_array).all() and faces.max() < len(vertex_array)
        # Compare actual serialized Unity vertex coordinates to the USD meshes.
        source = UsdGeom.Mesh(stage.GetPrimAtPath('/Robot/base/visual')) if path.stem == 'base' else UsdGeom.Mesh(stage.GetPrimAtPath(f'/Robot/leg_0_{path.stem}/visual'))
        usd_vertices = np.array(source.GetPointsAttr().Get())
        assert np.allclose(vertex_array[:, :3], usd_vertices[:, [0, 2, 1]], atol=1e-7)
    print(f'VALIDATED {count} legs: URDF/USD/Unity structure, mesh buffers, references and hinge frames')


def main():
    manifest = json.loads((ASSETS / 'assembly_manifest.json').read_text())
    groups = [part for ids in manifest['groups'].values() for part in ids]
    assert sorted(groups) == list(range(17)), 'Every source component must occur exactly once'
    for count in (8, 6):
        check(count)


if __name__ == '__main__':
    main()
