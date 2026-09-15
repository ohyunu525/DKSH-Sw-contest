"""Build six/eight-legged CAD assemblies, URDFs, and articulated USD assets.

Dependencies: numpy, trimesh, scipy, ufbx==0.0.5, usd-core, matplotlib.
The source is the assembled, disconnected CAD geometry in completeLEG.fbx.
Raw CAD coordinates are interpreted as millimetres (the FBX display scale is
not a reliable physical unit). No simplification or stretching of visuals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
import ufbx
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'DKSH SW Project/Assets/Prefabs/spiderbot'
DEST = ROOT / 'isaaclab_project/assets/spiderbot_variants'

# Indices are deterministic connected components of the source FBX (hash is
# verified below). The whole assembled leg is partitioned exactly once.
GROUPS = {
    'mount': [0, 4, 5, 16],
    'hip': [2, 3, 7, 9, 12],
    'femur': [8, 10, 13, 14],
    'tibia': [1, 6, 11, 15],
}
COLORS = {'base': (0.11, 0.15, 0.19), 'mount': (0.22, 0.26, 0.30),
          'hip': (0.16, 0.47, 0.65), 'femur': (0.31, 0.64, 0.73),
          'tibia': (0.92, 0.47, 0.18)}
EXPECTED_COMPONENT_FACES = [3162, 1784, 2888, 4518, 2384, 2936, 2888,
                            4002, 2310, 3162, 3624, 2310, 2310, 3162,
                            4402, 1784, 2686]
EXPECTED_SOURCE_SHA256 = '208dcdee9c9d47adc933e12fa8a74612a2d9eaaa24f8443706d5102654170ac3'
MASS = {'hip': 0.100, 'femur': 0.100, 'tibia': 0.10227}
RADIUS = 0.120


def ry(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rz(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def transform(rotation=np.eye(3), position=np.zeros(3)):
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = position
    return matrix


def read_cad_fbx(path):
    """Read the supplied STL-to-FBX export's raw CAD mesh, with strict checks."""
    scene = ufbx.load_file(str(path), load_external_files=False)
    mesh_nodes = [node for node in scene.nodes if node.mesh is not None]
    if len(mesh_nodes) != 1:
        raise ValueError(f'Expected one raw CAD mesh in {path}')
    node = mesh_nodes[0]
    mesh = node.mesh
    vertices = np.array(list(mesh.vertices), dtype=float)
    indices = list(mesh.vertex_indices)
    triangles = []
    for face in mesh.faces:
        if face.num_indices != 3:
            raise ValueError('This CAD export is expected to contain only triangles')
        triangles.append(indices[face.index_begin:face.index_begin + 3])
    if not np.isfinite(vertices).all():
        raise ValueError('Non-finite CAD coordinates')
    # Retain source face order, needed for reproducible component identities.
    return trimesh.Trimesh(vertices, triangles, process=True)


def prepare_leg():
    if hashlib.sha256((SOURCE / 'completeLEG.fbx').read_bytes()).hexdigest() != EXPECTED_SOURCE_SHA256:
        raise ValueError('Source CAD changed: review part grouping and hinge calibration before rebuilding')
    reference = read_cad_fbx(SOURCE / 'completeLEG.fbx')
    components = list(reference.split(only_watertight=False))
    actual = [len(mesh.faces) for mesh in components]
    if actual != EXPECTED_COMPONENT_FACES:
        raise ValueError('CAD assembly changed: review component groups and pivots before rebuilding')
    # Centres of the opposed motor/bearing geometry identify the hinge lines.
    centres = [mesh.bounds.mean(axis=0) for mesh in components]
    hip_pivot = (centres[2] + centres[12]) / 2
    mid_pivot = centres[8].copy()
    end_pivot = (centres[6] + centres[11]) / 2
    mid_pivot[1] = end_pivot[1] = hip_pivot[1]
    rotation = ry(math.pi / 4)
    pivots = {'mount': hip_pivot, 'hip': hip_pivot,
              'femur': mid_pivot, 'tibia': end_pivot}
    meshes = {}
    for name, ids in GROUPS.items():
        mesh = trimesh.util.concatenate([components[i] for i in ids])
        mesh.vertices = (mesh.vertices - pivots[name]) @ rotation.T * 0.001
        meshes[name] = mesh
    offsets = {
        'hip': np.zeros(3),
        'femur': rotation @ (mid_pivot - hip_pivot) * 0.001,
        'tibia': rotation @ (end_pivot - mid_pivot) * 0.001,
    }
    # Individual FBXs remain useful editable part sources. Inventory them and
    # record which shapes can be identified in the assembled reference by area.
    inventory = {}
    for path in sorted(SOURCE.glob('*.fbx')):
        if path.stem == 'completeLEG':
            continue
        mesh = read_cad_fbx(path)
        matches = [i for i, part in enumerate(components)
                   if abs(part.area - mesh.area) < 0.01]
        inventory[path.name] = {'triangles': len(mesh.faces),
                                'reference_components_with_matching_area': matches}
    return meshes, offsets, pivots, inventory


def make_body(count, mount):
    body = trimesh.creation.cylinder(radius=0.085, height=0.022, sections=count * 8)
    body.visual.face_colors = np.array((*COLORS['base'], 1)) * 255
    parts = [body]
    for index in range(count):
        angle = math.tau * index / count
        copy = mount.copy()
        copy.apply_transform(transform(rz(angle), rz(angle) @ np.array([RADIUS, 0, 0])))
        copy.visual.face_colors = np.array((*COLORS['mount'], 1)) * 255
        parts.append(copy)
    return trimesh.util.concatenate(parts)


def geometry_tree(count, meshes, offsets):
    body = make_body(count, meshes['mount'])
    nodes = [{'name': 'base', 'parent': '', 'mesh': 'base', 'position': [0, 0, 0],
              'yaw': 0, 'mass': 0.5, 'axis': ''}]
    for index in range(count):
        angle = math.tau * index / count
        parent = 'base'
        for segment in ('hip', 'femur', 'tibia'):
            name = f'leg_{index}_{segment}'
            position = rz(angle) @ np.array([RADIUS, 0, 0]) if segment == 'hip' else offsets[segment]
            nodes.append({'name': name, 'parent': parent, 'mesh': segment,
                          'position': position.tolist(), 'yaw': angle if segment == 'hip' else 0,
                          'mass': MASS[segment], 'axis': 'Z' if segment == 'hip' else 'Y'})
            parent = name
    return body, nodes


def vec(values):
    return ' '.join(f'{float(v):.9f}' for v in values)


def collision_boxes(node, meshes, count):
    if node['name'] == 'base':
        # Central body and separate mount boxes, rather than one wide box that
        # would enclose the hip links and empty spaces between legs.
        boxes = [(np.zeros(3), np.array([0.150, 0.150, 0.022]), 0.0)]
        mesh = meshes['mount']
        for index in range(count):
            angle = math.tau * index / count
            centre = rz(angle) @ (np.array([RADIUS, 0, 0]) + mesh.bounds.mean(axis=0))
            boxes.append((centre, mesh.extents, angle))
        return boxes
    mesh = meshes[node['mesh']]
    return [(mesh.bounds.mean(axis=0), mesh.extents, 0.0)]


def inertia(mesh, mass):
    # Conservative box approximation; measured COM/inertia are not available.
    x, y, z = mesh.extents
    return mesh.bounds.mean(axis=0), mass / 12 * np.array([y*y+z*z, x*x+z*z, x*x+y*y])


def write_urdf(path, count, meshes, nodes):
    robot = ET.Element('robot', name=f'dksh_spiderbot_{count}leg_cad')
    for node in nodes:
        link = ET.SubElement(robot, 'link', name=node['name'])
        center, diagonal = inertia(meshes[node['mesh']], node['mass'])
        inertial = ET.SubElement(link, 'inertial')
        ET.SubElement(inertial, 'origin', xyz=vec(center), rpy='0 0 0')
        ET.SubElement(inertial, 'mass', value=str(node['mass']))
        ET.SubElement(inertial, 'inertia', ixx=str(diagonal[0]), iyy=str(diagonal[1]),
                      izz=str(diagonal[2]), ixy='0', ixz='0', iyz='0')
        visual = ET.SubElement(link, 'visual')
        geometry = ET.SubElement(visual, 'geometry')
        mesh_path = f'meshes/{node["mesh"]}.obj'
        ET.SubElement(geometry, 'mesh', filename=mesh_path)
        material = ET.SubElement(visual, 'material', name=node['mesh'])
        ET.SubElement(material, 'color', rgba=vec((*COLORS[node['mesh']], 1)))
        for centre, size, yaw in collision_boxes(node, meshes, count):
            collision = ET.SubElement(link, 'collision')
            ET.SubElement(collision, 'origin', xyz=vec(centre), rpy=vec((0, 0, yaw)))
            ET.SubElement(ET.SubElement(collision, 'geometry'), 'box', size=vec(size))
        if node['parent']:
            joint = ET.SubElement(robot, 'joint', name=f'{node["name"]}_joint', type='revolute')
            ET.SubElement(joint, 'parent', link=node['parent'])
            ET.SubElement(joint, 'child', link=node['name'])
            ET.SubElement(joint, 'origin', xyz=vec(node['position']), rpy=vec((0, 0, node['yaw'])))
            ET.SubElement(joint, 'axis', xyz='0 0 1' if node['axis'] == 'Z' else '0 1 0')
            ET.SubElement(joint, 'limit', lower=str(-math.pi/2), upper=str(math.pi/2),
                          effort='0.980665', velocity=str(math.radians(400)))
            ET.SubElement(joint, 'dynamics', damping='0.05', friction='0.01')
    ET.indent(robot, space='  ')
    ET.ElementTree(robot).write(path, encoding='utf-8', xml_declaration=True)


def write_usd(path, count, meshes, nodes):
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, '/Robot')
    stage.SetDefaultPrim(root.GetPrim())
    # Floating articulation root on the root rigid body, compatible with PhysX.
    materials = {}
    for name, color in COLORS.items():
        mat = UsdShade.Material.Define(stage, f'/Robot/Looks/{name}')
        shader = UsdShade.Shader.Define(stage, f'/Robot/Looks/{name}/Shader')
        shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(0.55)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
        materials[name] = mat
    # Store each visual once; repeated legs reference it with instanceable prims.
    visual_stage = Usd.Stage.CreateNew(str(path.with_name('visuals.usdc')))
    UsdGeom.SetStageUpAxis(visual_stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(visual_stage, 1.0)
    for key in ('base', 'hip', 'femur', 'tibia'):
        source = meshes[key]
        mesh = UsdGeom.Mesh.Define(visual_stage, f'/{key}')
        mesh.CreatePointsAttr(source.vertices.astype(np.float32).tolist())
        mesh.CreateFaceVertexCountsAttr([3] * len(source.faces))
        mesh.CreateFaceVertexIndicesAttr(source.faces.ravel().tolist())
        mesh.CreateSubdivisionSchemeAttr('none')
        mesh.CreateDoubleSidedAttr(True)
    visual_stage.GetRootLayer().Save()
    world = {'': np.eye(4)}
    for node in nodes:
        name = node['name']
        world[name] = world[node['parent']] @ transform(rz(node['yaw']), node['position'])
        position = world[name][:3, 3]
        rotation = world[name][:3, :3]
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        body = UsdGeom.Xform.Define(stage, f'/Robot/{name}')
        body.AddTranslateOp().Set(Gf.Vec3d(*position))
        body.AddOrientOp().Set(Gf.Quatf(math.cos(yaw/2), Gf.Vec3f(0, 0, math.sin(yaw/2))))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        if name == 'base':
            UsdPhysics.ArticulationRootAPI.Apply(body.GetPrim())
        mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
        center, diagonal = inertia(meshes[node['mesh']], node['mass'])
        mass.CreateMassAttr(node['mass'])
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*center))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*diagonal))
        mass.CreatePrincipalAxesAttr(Gf.Quatf(1))
        visual = stage.DefinePrim(f'/Robot/{name}/visual')
        visual.GetReferences().AddReference('./visuals.usdc', f'/{node["mesh"]}')
        UsdShade.MaterialBindingAPI.Apply(visual).Bind(materials[node['mesh']])
        visual.SetInstanceable(True)
        for i, (centre, size, box_yaw) in enumerate(collision_boxes(node, meshes, count)):
            box = UsdGeom.Cube.Define(stage, f'/Robot/{name}/collision_{i}')
            box.CreateSizeAttr(1)
            box.AddTranslateOp().Set(Gf.Vec3d(*centre))
            box.AddRotateZOp().Set(math.degrees(box_yaw))
            box.AddScaleOp().Set(Gf.Vec3f(*size))
            box.CreatePurposeAttr('guide')
            UsdPhysics.CollisionAPI.Apply(box.GetPrim())
        if node['parent']:
            joint = UsdPhysics.RevoluteJoint.Define(stage, f'/Robot/joints/{name}_joint')
            joint.CreateBody0Rel().SetTargets([f'/Robot/{node["parent"]}'])
            joint.CreateBody1Rel().SetTargets([f'/Robot/{name}'])
            joint.CreateLocalPos0Attr(Gf.Vec3f(*node['position']))
            joint.CreateLocalPos1Attr(Gf.Vec3f(0))
            joint.CreateLocalRot0Attr(Gf.Quatf(math.cos(node['yaw']/2), Gf.Vec3f(0, 0, math.sin(node['yaw']/2))))
            joint.CreateLocalRot1Attr(Gf.Quatf(1))
            joint.CreateAxisAttr(node['axis'])
            joint.CreateLowerLimitAttr(-90)
            joint.CreateUpperLimitAttr(90)
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'angular')
            drive.CreateTypeAttr('force')
            # USD angular drive gains use degrees; equivalent to 8 Nm/rad.
            drive.CreateStiffnessAttr(8 * math.pi / 180)
            drive.CreateDampingAttr(0.35 * math.pi / 180)
            drive.CreateMaxForceAttr(0.980665)
            drive.CreateTargetPositionAttr(0)
    stage.GetRootLayer().Save()
    return world


def render_preview(scenes, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(15, 8), facecolor='#f4f6f7')
    for column, (count, entries) in enumerate(scenes.items()):
        for row, (elev, azim) in enumerate([(28, -58), (90, -90)]):
            ax = fig.add_subplot(2, 2, row * 2 + column + 1, projection='3d')
            ax.set_facecolor('#f4f6f7')
            for key, mesh in entries:
                ax.add_collection3d(Poly3DCollection(mesh.triangles,
                    facecolors=COLORS[key], linewidths=0, shade=True))
            ax.set(xlim=(-0.26, 0.26), ylim=(-0.26, 0.26), zlim=(-0.115, 0.10))
            ax.set_box_aspect((1, 1, 0.42)); ax.view_init(elev, azim)
            ax.set_axis_off()
            if row == 0:
                ax.set_title(f'{count} LEGS / {count*3} JOINTS', fontsize=16, pad=0)
    fig.subplots_adjust(left=0, right=1, bottom=0.035, top=0.94, wspace=0, hspace=-0.20)
    fig.text(0.5, 0.02, 'Original CAD leg geometry | Provisional central chassis and joint calibration',
             ha='center', fontsize=10, color='#59636b')
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEST)
    parser.add_argument('--no-preview', action='store_true')
    parser.add_argument('--unity', action='store_true', help='Also export Unity meshes, materials and prefabs')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    meshes, offsets, pivots, inventory = prepare_leg()
    scenes = {}
    for count in (8, 6):
        folder = output / f'spiderbot_{count}leg'
        (folder / 'meshes').mkdir(parents=True, exist_ok=True)
        body, nodes = geometry_tree(count, meshes, offsets)
        variant_meshes = dict(meshes, base=body)
        for key in ('base', 'hip', 'femur', 'tibia'):
            variant_meshes[key].export(folder / 'meshes' / f'{key}.obj', include_normals=True)
        write_urdf(folder / f'spiderbot_{count}leg.urdf', count, variant_meshes, nodes)
        world = write_usd(folder / f'spiderbot_{count}leg.usd', count, variant_meshes, nodes)
        scene = trimesh.Scene()
        entries = []
        for node in nodes:
            mesh = variant_meshes[node['mesh']].copy()
            mesh.visual.face_colors = np.array((*COLORS[node['mesh']], 1)) * 255
            scene.add_geometry(mesh, node_name=node['name'], geom_name=node['name'],
                               parent_node_name=node['parent'] or 'world',
                               transform=transform(rz(node['yaw']), node['position']))
            mesh.apply_transform(world[node['name']])
            entries.append((node['mesh'], mesh))
        scene.export(folder / f'spiderbot_{count}leg.glb')
        scenes[count] = entries
        # JSON is also the exact geometry/kinematic handoff to the Unity builder.
        data = {'legCount': count, 'nodes': nodes, 'meshes': [],
                'spawnHeight': float(-min(m.bounds[0, 2] for _, m in entries) + 0.012)}
        for key in ('base', 'hip', 'femur', 'tibia'):
            mesh = variant_meshes[key]
            data['meshes'].append({'name': key, 'vertices': mesh.vertices.reshape(-1).tolist(),
                                    'triangles': mesh.faces.reshape(-1).tolist(), 'color': COLORS[key]})
        for node in data['nodes']:
            centre, diagonal = inertia(variant_meshes[node['mesh']], node['mass'])
            node['centerOfMass'] = centre.tolist()
            node['inertia'] = diagonal.tolist()
            node['colliders'] = [{'position': c.tolist(), 'size': s.tolist(), 'yaw': a}
                                 for c, s, a in collision_boxes(node, variant_meshes, count)]
        # Large transient handoff is not part of the checked-in asset bundle.
        handoff = ROOT / 'outputs/spiderbot-unity'
        handoff.mkdir(parents=True, exist_ok=True)
        (handoff / f'spiderbot_{count}leg.json').write_text(json.dumps(data), encoding='utf-8')
        print(f'BUILT {count} legs: {len(nodes)} rigid bodies, {len(nodes)-1} joints, spawn={data["spawnHeight"]:.5f}m')
    manifest = {'source': str((SOURCE / 'completeLEG.fbx').relative_to(ROOT)),
                'source_sha256': hashlib.sha256((SOURCE/'completeLEG.fbx').read_bytes()).hexdigest(),
                'cad_unit_assumption': 'millimetres', 'cad_to_link_rotation_y_degrees': 45,
                'groups': GROUPS, 'component_triangle_counts': EXPECTED_COMPONENT_FACES,
                'hinge_pivots_cad_mm': {k: v.tolist() for k, v in pivots.items()},
                'joint_offsets_m': {k: v.tolist() for k, v in offsets.items()},
                'standalone_part_inventory': inventory, 'mount_radius_m': RADIUS,
                'calibration_status': 'Provisional axes, mass, COM, collision boxes; verify against hardware CAD.'}
    (output/'assembly_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if not args.no_preview:
        render_preview(scenes, output/'spiderbot_variants.png')
    if args.unity:
        from export_spiderbot_unity import main as export_unity
        export_unity()


if __name__ == '__main__':
    main()
