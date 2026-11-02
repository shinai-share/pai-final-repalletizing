"""Builds the load-collapse recovery cell on a USD stage.

World frame: z up, x from pallet A (source, -x) to pallet B (destination, +x), origin
midway between the pallets on the floor. All geometry comes from the Cell config.
"""
from __future__ import annotations

import math
import numpy as np
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

from ..config import Cell
from . import usd_build as ub
from .. import structure
from ..structure import FORK_SIDES, WRIST_LINK

ROOT = '/World'
CELL = '/World/cell'
SCENERY = '/World/scenery'         # the warehouse around the cell: for people, far outside every map the robot keeps
CASES = '/World/cell/cases'
GANTRY = '/World/cell/gantry'
GRIPPER = '/World/cell/gantry/suction'
CAMERAS = {
    'overhead_a': '/World/cell/cam_overhead_a',
    'overhead_b': '/World/cell/cam_overhead_b',
    'tool': '/World/cell/gantry/tool/cam_tool',
    'down': '/World/cell/gantry/wrist_r/cam_down',       # plate centre, looking along the cups' approach
    'presentation': '/World/cell/cam_presentation',
    'side': '/World/cell/cam_side',                        # for people: the collapse seen from the side, low
    'closeup': '/World/cell/gantry/tool/cam_closeup',      # for people: rides with the gantry, sees cups and wrist
    'under': '/World/cell/cam_under',
}
CUP_COMPRESSION = 0.010           # m a bellows cup compresses; a stop at zero doubled the plate's contact and took the tines' push
GANTRY_JOINTS = ('x_joint', 'y_joint', 'z_joint', 'yaw_joint', 'pitch_joint', 'roll_joint')
# Under-supports: a bar per side moves along tool z (joint value = -extension), its tine along +y.
FORK_JOINTS = ('fork_z_l', 'fork_z_r', 'fork_y_l', 'fork_y_r')


def _physics_scene(stage, cell: Cell):
    scene = UsdPhysics.Scene.Define(stage, ROOT + '/physicsScene')
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)
    px = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    px.CreateTimeStepsPerSecondAttr(int(round(1.0 / cell.physics.dt)))
    px.CreateSolverTypeAttr('TGS')
    px.CreateEnableCCDAttr(True)
    px.CreateEnableStabilizationAttr(True)
    px.CreateBroadphaseTypeAttr('MBP')
    px.CreateEnableGPUDynamicsAttr(False)
    px.CreateGpuFoundLostAggregatePairsCapacityAttr(4096)


def _lights(stage):
    # A warehouse is lit from high-bay fittings above, with daylight from the side: soft fill from the
    # dome, a low sun for shadows that show which case rests on which, and rows of ceiling lights.
    dome = UsdLux.DomeLight.Define(stage, ROOT + '/dome')
    dome.CreateIntensityAttr(180.0)
    dome.CreateColorAttr(Gf.Vec3f(0.92, 0.95, 1.0))
    sun = UsdLux.DistantLight.Define(stage, ROOT + '/sun')
    sun.CreateIntensityAttr(700.0)
    sun.CreateAngleAttr(2.0)
    sun.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.9))
    ub.set_pose(sun.GetPrim(), (0, 0, 5), quat=(0.92, 0.25, 0.25, 0.0))
    for i, x in enumerate((-3.0, 0.0, 3.0)):
        for j, y in enumerate((-2.0, 1.5)):
            lamp = UsdLux.RectLight.Define(stage, f'{ROOT}/highbay_{i}_{j}')
            lamp.CreateWidthAttr(1.2)
            lamp.CreateHeightAttr(0.3)
            lamp.CreateIntensityAttr(9000.0)
            lamp.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.94))
            ub.set_pose(lamp.GetPrim(), (x, y, 6.0))          # a rect light shines along its -z: down


def _floor(stage, cell: Cell):
    mat = ub.material(stage, CELL + '/mat/floor', cell.appearance.floor_color, friction=(0.7, 0.6), roughness=0.35)
    prim = ub.rigid_body(stage, CELL + '/floor', (0, 0, -0.05), kinematic=True)
    ub.box(stage, CELL + '/floor/geom', (16.0, 14.0, 0.1), mat=mat)
    # Painted walkway lines around the robot's working area, as on a warehouse floor.
    paint = ub.material(stage, CELL + '/mat/floor_line', (0.80, 0.62, 0.02), roughness=0.5)
    p = cell.pallet
    x0, x1 = p.a_center[0] - p.size[0] / 2 - 0.55, p.b_center[0] + p.size[0] / 2 + 0.55
    y0, y1, w = -1.30, 1.30, 0.08
    for name, size, pos in (('s', (x1 - x0 + w, w, 0.002), ((x0 + x1) / 2, y0, 0.001)),
                            ('n', (x1 - x0 + w, w, 0.002), ((x0 + x1) / 2, y1, 0.001)),
                            ('w', (w, y1 - y0, 0.002), (x0, 0.0, 0.001)),
                            ('e', (w, y1 - y0, 0.002), (x1, 0.0, 0.001))):
        ub.box(stage, f'{CELL}/floor_line_{name}', size, pos=pos, mat=paint, collide=False)
    return prim


def _scenery(stage):
    """Walls and storage racks well outside the cell, for people only (no collision)."""
    UsdGeom.Scope.Define(stage, SCENERY)
    wall = ub.material(stage, SCENERY + '/mat/wall', (0.62, 0.63, 0.62), roughness=0.9)
    upright = ub.material(stage, SCENERY + '/mat/rack_upright', (0.04, 0.12, 0.40), roughness=0.5, metallic=0.3)
    beam = ub.material(stage, SCENERY + '/mat/rack_beam', (0.80, 0.30, 0.03), roughness=0.5, metallic=0.3)
    stock = ub.material(stage, SCENERY + '/mat/stock', (0.46, 0.29, 0.13), roughness=0.9)
    ub.box(stage, SCENERY + '/wall_back', (16.0, 0.2, 7.0), pos=(0.0, 6.0, 3.5), mat=wall, collide=False)
    ub.box(stage, SCENERY + '/wall_east', (0.2, 14.0, 7.0), pos=(7.5, 0.0, 3.5), mat=wall, collide=False)
    ub.box(stage, SCENERY + '/wall_west', (0.2, 14.0, 7.0), pos=(-7.5, 0.0, 3.5), mat=wall, collide=False)
    # Pallet racking against the back wall: four bays, three beam levels, stock on the lower two.
    for b, x in enumerate((-5.4, -2.7, 0.0, 2.7)):
        for u, (dx, y) in enumerate(((0.0, 4.6), (0.0, 5.6), (2.7, 4.6), (2.7, 5.6))):
            ub.box(stage, f'{SCENERY}/rack_{b}_upright_{u}', (0.08, 0.08, 4.5), pos=(x + dx, y, 2.25), mat=upright, collide=False)
        for k, z in enumerate((0.15, 1.6, 3.05)):
            for f, y in enumerate((4.6, 5.6)):
                ub.box(stage, f'{SCENERY}/rack_{b}_beam_{k}_{f}', (2.7, 0.06, 0.12), pos=(x + 1.35, y, z), mat=beam, collide=False)
            if k < 2:
                ub.box(stage, f'{SCENERY}/rack_{b}_stock_{k}', (2.2, 0.9, 1.0), pos=(x + 1.35, 5.1, z + 0.56), mat=stock, collide=False)


def _pallets(stage, cell: Cell):
    p = cell.pallet
    plastic = ub.material(stage, CELL + '/mat/pallet', cell.appearance.pallet_color, friction=p.friction, roughness=0.55)
    pocket = ub.material(stage, CELL + '/mat/pallet_pocket', (0.015, 0.015, 0.017), roughness=0.9)
    lx, ly, h = p.size
    bodies = []
    for name, c in (('pallet_a', p.a_center), ('pallet_b', p.b_center)):
        body = ub.rigid_body(stage, f'{CELL}/{name}', (c[0], c[1], h / 2), kinematic=True)
        ub.box(stage, f'{CELL}/{name}/geom', p.size, mat=plastic)
        # Fork openings on all four sides, visual only: the collision shape stays the solid block,
        # and the openings sit below the deck, where no camera the robot uses looks.
        for k, s in enumerate((-0.27, 0.27)):
            for e, sign in enumerate((-1, 1)):
                ub.box(stage, f'{CELL}/{name}/pocket_x{k}{e}', (0.002, ly * 0.26, h * 0.55),
                       pos=(sign * (lx / 2 + 0.001), s * ly, -h * 0.08), mat=pocket, collide=False)
                ub.box(stage, f'{CELL}/{name}/pocket_y{k}{e}', (lx * 0.26, 0.002, h * 0.55),
                       pos=(s * lx, sign * (ly / 2 + 0.001), -h * 0.08), mat=pocket, collide=False)
        bodies.append(body)
    a, b = bodies
    mark = ub.material(stage, CELL + '/mat/handover', (0.85, 0.62, 0.12))
    for i, (hx, hy) in enumerate(cell.handover.spots):
        ub.box(stage, f'{CELL}/handover_mark_{i}', (0.30, 0.42, 0.004), pos=(hx, hy, 0.002), mat=mark, collide=False)
    st = cell.staging
    steel = ub.material(stage, CELL + '/mat/staging', (0.30, 0.34, 0.40), friction=(0.6, 0.5))
    for i, (tx, ty) in enumerate(st.tables):
        ub.rigid_body(stage, f'{CELL}/staging_{i}', (tx, ty, st.table_size[2] / 2), kinematic=True)
        ub.box(stage, f'{CELL}/staging_{i}/geom', st.table_size, mat=steel)
    return a, b


def _cases(stage, cell: Cell, spawns):
    UsdGeom.Scope.Define(stage, CASES)
    mats = {}
    palette = cell.appearance.case_colors
    band = ub.material(stage, CELL + '/mat/case_band', cell.appearance.case_print_color, roughness=0.7)
    label = ub.material(stage, CELL + '/mat/case_label', (0.93, 0.93, 0.90), roughness=0.8)
    for i, s in enumerate(spawns):
        sku = cell.sku(s.sku_id)
        key = (sku.id, i % len(palette))
        if key not in mats:
            mats[key] = ub.material(stage, f'{CELL}/mat/case_{sku.id.replace("-", "_")}_{key[1]}', palette[key[1]],
                                    friction=sku.friction, roughness=0.92)
        mass = s.mass if s.mass is not None else sku.mass
        prim = ub.rigid_body(stage, f'{CASES}/{s.case_id}', s.pos, mass=mass, yaw_deg=s.yaw_deg, quat=s.quat)
        ub.box(stage, f'{CASES}/{s.case_id}/geom', sku.size, mat=mats[key])
        # Where the contents sit, and the inertia that goes with it (box about its centre of mass,
        # shifted by the offset). Physics only: the robot finds the centre of mass by a test lift.
        w, d, h = sku.size
        cx, cy, cz = s.com
        m = UsdPhysics.MassAPI(prim)
        m.CreateCenterOfMassAttr(Gf.Vec3f(float(cx), float(cy), float(cz)))
        m.CreateDiagonalInertiaAttr(Gf.Vec3f(float(mass * ((d * d + h * h) / 12 + cy * cy + cz * cz)),
                                             float(mass * ((w * w + h * h) / 12 + cx * cx + cz * cz)),
                                             float(mass * ((w * w + d * d) / 12 + cx * cx + cy * cy))))
        m.CreatePrincipalAxesAttr(Gf.Quatf(1, 0, 0, 0))
        # A dark band along the long edge of the top face and a label on one end: which face is up
        # and how the case lies can be seen at a glance.
        ub.box(stage, f'{CASES}/{s.case_id}/band', (w * 0.9, 0.035, 0.002), pos=(0, 0, h / 2 + 0.001), mat=band, collide=False)
        ub.box(stage, f'{CASES}/{s.case_id}/label', (0.002, d * 0.55, h * 0.45), pos=(w / 2 + 0.001, 0, 0), mat=label, collide=False)
        prim.CreateAttribute('shinai:sku', Sdf.ValueTypeNames.String).Set(sku.id)
        prim.CreateAttribute('shinai:layer', Sdf.ValueTypeNames.Int).Set(s.layer)
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rb.CreateMaxDepenetrationVelocityAttr(1.0)
        rb.CreateSleepThresholdAttr(0.0)         # a sleeping case would ignore the transport gravity
        rb.CreateSolverPositionIterationCountAttr(cell.physics.position_iterations)
        rb.CreateSolverVelocityIterationCountAttr(cell.physics.velocity_iterations)


def wrap_cases(stage, cell: Cell, spawns, wrapped_layers):
    """Stretch film: wrapped layers are bound to pallet A as one unit load until the film is cut.

    Joint frames are set from the as-built poses, so binding does not move anything."""
    UsdGeom.Scope.Define(stage, CELL + '/wrap')
    pallet_pos = np.array([cell.pallet.a_center[0], cell.pallet.a_center[1], cell.pallet.size[2] / 2])
    paths = []
    for s in spawns:
        if s.layer not in wrapped_layers:
            continue
        path = f'{CELL}/wrap/{s.case_id}'
        j = UsdPhysics.FixedJoint.Define(stage, path)
        j.CreateBody0Rel().SetTargets([Sdf.Path(CELL + '/pallet_a')])
        j.CreateBody1Rel().SetTargets([Sdf.Path(f'{CASES}/{s.case_id}')])
        j.CreateLocalPos0Attr(Gf.Vec3f(*(np.array(s.pos) - pallet_pos)))
        j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        half = math.radians(s.yaw_deg) / 2
        j.CreateLocalRot0Attr(Gf.Quatf(math.cos(half), 0, 0, math.sin(half)))
        j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
        # Jointed bodies stop colliding by default; keep contact so cutting the film
        # does not let the case sink through the pallet.
        j.CreateCollisionEnabledAttr(True)
        paths.append(path)
    return paths


def cut_wrap(stage, paths):
    """The operator cuts the stretch film before unloading: disable every wrap joint."""
    for path in paths:
        stage.GetPrimAtPath(path).GetAttribute('physics:jointEnabled').Set(False)


def _gantry(stage, cell: Cell):
    g = cell.gantry
    z0 = g.origin_height
    origin = (0.0, 0.0, z0)
    frame_mat = ub.material(stage, CELL + '/mat/frame', cell.appearance.frame_color, roughness=0.4, metallic=0.2)
    tool_mat = ub.material(stage, CELL + '/mat/tool', (0.12, 0.13, 0.15), friction=(1.0, 0.9))
    cup_mat = ub.material(stage, CELL + '/mat/cup', (0.86, 0.38, 0.18))
    base = ub.rigid_body(stage, GANTRY + '/base', origin, mass=500.0, gravity=False)
    UsdPhysics.ArticulationRootAPI.Apply(base)
    art = PhysxSchema.PhysxArticulationAPI.Apply(base)
    art.CreateEnabledSelfCollisionsAttr(False)
    art.CreateSleepThresholdAttr(0.0)          # a sleeping articulation ignores new drive targets
    art.CreateStabilizationThresholdAttr(0.0)
    art.CreateSolverPositionIterationCountAttr(32)
    ub.fixed_joint(stage, GANTRY + '/base/anchor', None, Sdf.Path(GANTRY + '/base'), local0=origin)
    span_x = g.x_range[1] - g.x_range[0] + 0.6
    span_y = g.y_range[1] - g.y_range[0] + 0.6
    top = g.beam_height - z0
    for sgn in (-1, 1):   # side rails and legs (visual only, out of reach)
        ub.box(stage, f'{GANTRY}/base/rail_{"np"[sgn > 0]}', (span_x, 0.12, 0.16), pos=(0, sgn * span_y / 2, top), mat=frame_mat, collide=False)
        for sx in (-1, 1):
            ub.box(stage, f'{GANTRY}/base/leg_{"np"[sgn > 0]}{"np"[sx > 0]}', (0.14, 0.14, g.beam_height), pos=(sx * span_x / 2, sgn * span_y / 2, top - g.beam_height / 2), mat=frame_mat, collide=False)
    for name in ('bridge', 'trolley', 'ram', 'tool', 'wrist_p', 'wrist_r'):
        ub.rigid_body(stage, f'{GANTRY}/{name}', origin, gravity=False,
                      mass={'bridge': 80.0, 'trolley': 30.0, 'ram': structure.ram_mass(cell), 'tool': 3.0,
                            'wrist_p': 2.0, 'wrist_r': 3.0}[name])
    ub.box(stage, GANTRY + '/bridge/beam', (0.18, span_y, 0.20), pos=(0, 0, top - 0.18), mat=frame_mat, collide=False)
    # The trolley guides the ram, and the ram is long enough to stay in it at every height (structure.py). The
    # column ends a wrist link above the cup face: a case turned a quarter rises up to half its height above the
    # face (0.135 m, plus any grip offset) and must clear it.
    for b in structure.gantry_boxes(cell):
        ub.box(stage, f'{GANTRY}/{b.body}/{b.name}', b.size, pos=b.pos, mat=frame_mat, collide=b.collide)
    # The plate turns with the wrist; the wrist axes pass through the centre of the cup face.
    ub.box(stage, GANTRY + '/wrist_r/plate', g.tool_size, pos=(0, 0, g.tool_size[2] / 2), mat=tool_mat)
    ub.box(stage, GANTRY + '/tool/wrist_link', (0.08, 0.08, WRIST_LINK - g.tool_size[2]),
           pos=(0, 0, (WRIST_LINK + g.tool_size[2]) / 2), mat=frame_mat, collide=False)
    # Vacuum head: a separate body rigidly fixed to the tool, so the suction attachment
    # joints (head -> tool) never constrain the gantry's own axes.
    ub.rigid_body(stage, GANTRY + '/head', origin, mass=1.0, gravity=False)
    for i, (cx, cy) in enumerate(g.cup_offsets):
        ub.box(stage, f'{GANTRY}/head/cup_{i}', (0.07, 0.07, 0.012), pos=(cx, cy, -0.006), mat=cup_mat, collide=False)
    ub.fixed_joint(stage, GANTRY + '/head_mount', Sdf.Path(GANTRY + '/wrist_r'), Sdf.Path(GANTRY + '/head'))
    lim = g.limits
    k, d = g.drive_stiffness, g.drive_damping
    ub.drive_joint(stage, GANTRY + '/x_joint', 'prismatic', Sdf.Path(GANTRY + '/base'), Sdf.Path(GANTRY + '/bridge'), 'X', lim[0], k, d)
    ub.drive_joint(stage, GANTRY + '/y_joint', 'prismatic', Sdf.Path(GANTRY + '/bridge'), Sdf.Path(GANTRY + '/trolley'), 'Y', lim[1], k, d)
    ub.drive_joint(stage, GANTRY + '/z_joint', 'prismatic', Sdf.Path(GANTRY + '/trolley'), Sdf.Path(GANTRY + '/ram'), 'Z', lim[2], k, d)
    ub.drive_joint(stage, GANTRY + '/yaw_joint', 'revolute', Sdf.Path(GANTRY + '/ram'), Sdf.Path(GANTRY + '/tool'), 'Z', lim[3], k, d)
    ub.drive_joint(stage, GANTRY + '/pitch_joint', 'revolute', Sdf.Path(GANTRY + '/tool'), Sdf.Path(GANTRY + '/wrist_p'), 'Y', lim[4], k, d)
    ub.drive_joint(stage, GANTRY + '/roll_joint', 'revolute', Sdf.Path(GANTRY + '/wrist_p'), Sdf.Path(GANTRY + '/wrist_r'), 'X', lim[5], k, d)
    dc = cell.down_camera
    ub.camera(stage, CAMERAS['down'], dc.offset, (1.0, 0.0, 0.0, 0.0), focal_mm=dc.focal_mm, clip=(0.005, 3.0))   # USD cameras look along -z
    tc = cell.tool_camera
    pitch = math.radians(tc.pitch_deg)
    ub.camera(stage, CAMERAS['tool'], tc.offset, (math.cos(pitch / 2), math.sin(pitch / 2), 0, 0), focal_mm=tc.focal_mm, h_aperture_mm=tc.aperture_mm)


def _fork(stage, cell: Cell):
    """Two L-shaped under-supports carried by the wrist. At joint zero a tine's top is level with the
    cup face and the tine lies behind its bar; the bar joint moves along tool z (value = -extension)
    and the tine joint slides the tine along +y under the held case. Each joint has a force-type drive
    with its own cap, and each side is a separate joint, so the load on each tine is read apart."""
    f = cell.fork
    g = cell.gantry
    origin = (0.0, 0.0, g.origin_height)
    steel = ub.material(stage, CELL + '/mat/fork_bar', (0.55, 0.57, 0.60), roughness=0.35, metallic=0.6)
    pad = ub.material(stage, CELL + '/mat/fork_pad', (0.10, 0.10, 0.11), friction=f.friction, roughness=0.8)
    pm = PhysxSchema.PhysxMaterialAPI.Apply(pad.GetPrim())
    pm.CreateFrictionCombineModeAttr('min')              # the pad's low friction governs, not the board's
    pm.CreateCompliantContactStiffnessAttr(float(f.pad_stiffness))
    pm.CreateCompliantContactDampingAttr(float(f.pad_damping))
    # The yoke the guides stand on, bolted over the plate's -y face (structure.fork_mounts, review 2026-10-31).
    for b in structure.fork_mounts(cell):
        if not b.name.startswith('fork_guide_'):
            ub.box(stage, f'{GANTRY}/{b.body}/{b.name}', b.size, pos=b.pos, mat=steel, collide=b.collide)
    for side, _ in FORK_SIDES:
        bar, tine = f'{GANTRY}/fork_bar_{side}', f'{GANTRY}/fork_tine_{side}'
        ub.rigid_body(stage, bar, origin, mass=f.part_mass, gravity=False)
        ub.rigid_body(stage, tine, origin, mass=f.part_mass, gravity=False)
        # The bar runs through its guide on the yoke; the tine through a sleeve at the bar's foot (structure.py). Each
        # stays in its guide over all of its travel.
        for b in structure.fork_boxes(cell, side):
            path = f'{GANTRY}/{b.body}/{b.name}' if b.name != 'tine' else tine + '/geom'
            ub.box(stage, path, b.size, pos=b.pos, mat=pad if b.name == 'tine' else steel, collide=b.collide)
        e0, e1 = f.extension_range
        ub.drive_joint(stage, f'{GANTRY}/fork_z_{side}', 'prismatic', Sdf.Path(GANTRY + '/wrist_r'), Sdf.Path(bar), 'Z',
                       (-e1, -e0), f.drive_stiffness, f.drive_damping, drive_type='force', max_force=f.z_force_limit)
        ub.drive_joint(stage, f'{GANTRY}/fork_y_{side}', 'prismatic', Sdf.Path(bar), Sdf.Path(tine), 'Y',
                       f.insert_range, f.drive_stiffness, f.drive_damping, drive_type='force', max_force=f.y_force_limit)
        for name in ('fork_z_', 'fork_y_'):
            stage.GetPrimAtPath(f'{GANTRY}/{name}{side}').GetAttribute('drive:linear:physics:targetPosition').Set(
                float(-e0) if name == 'fork_z_' else 0.0)


def _suction(stage, cell: Cell):
    """Isaac Surface Gripper with one D6 attachment point per vacuum cup."""
    import usd.schema.isaac.robot_schema as rs
    gr = cell.gripper
    rs.CreateSurfaceGripper(stage, GRIPPER)
    prim = stage.GetPrimAtPath(GRIPPER)
    UsdGeom.Scope.Define(stage, GRIPPER + '_points')
    points = []
    for i, (cx, cy) in enumerate(cell.gantry.cup_offsets):
        path = f'{GRIPPER}_points/cup_{i}'
        jp = stage.DefinePrim(path, 'PhysicsJoint')
        rs.ApplyAttachmentPointAPI(jp)
        for axis in ('transX', 'transY', 'transZ', 'rotX', 'rotY', 'rotZ'):
            UsdPhysics.LimitAPI.Apply(jp, axis)
        for axis in ('rotX', 'rotY', 'rotZ', 'transZ'):
            UsdPhysics.DriveAPI.Apply(jp, axis)
        # Free axes are expressed with low > high, as in the Isaac example scene.
        for axis in ('transX', 'transY'):
            jp.GetAttribute(f'limit:{axis}:physics:low').Set(1.0)
            jp.GetAttribute(f'limit:{axis}:physics:high').Set(-1.0)
        jp.GetAttribute('limit:transZ:physics:low').Set(-CUP_COMPRESSION)    # bellows give way: no rigid stop where the plate already stops the case
        jp.GetAttribute('limit:transZ:physics:high').Set(0.01)
        tilt = cell.gripper.max_face_tilt_deg
        for axis in ('rotX', 'rotY', 'rotZ'):
            jp.GetAttribute(f'limit:{axis}:physics:low').Set(-tilt)
            jp.GetAttribute(f'limit:{axis}:physics:high').Set(tilt)
        # Bellows cups conform to a tilted face; a case held on its side sags on them by its own
        # moment (measured by scripts/probe_wrist.py and turned out by the executive).
        jp.GetAttribute('drive:rotX:physics:stiffness').Set(float(gr.cup_tilt_stiffness))
        jp.GetAttribute('drive:rotY:physics:stiffness').Set(float(gr.cup_tilt_stiffness))
        jp.GetAttribute('drive:rotZ:physics:stiffness').Set(float(gr.cup_twist_stiffness))
        jp.GetAttribute('drive:transZ:physics:stiffness').Set(20000.0)
        jp.GetAttribute('drive:transZ:physics:damping').Set(500.0)
        jp.CreateAttribute('isaac:forwardAxis', Sdf.ValueTypeNames.Token).Set('Z')
        jp.CreateAttribute('isaac:clearanceOffset', Sdf.ValueTypeNames.Float).Set(0.002)
        jp.CreateRelationship('physics:body0').SetTargets([Sdf.Path(GANTRY + '/head')])
        jp.CreateRelationship('physics:body1').SetTargets([Sdf.Path(GANTRY + '/wrist_r')])
        jp.CreateAttribute('physics:excludeFromArticulation', Sdf.ValueTypeNames.Bool).Set(True)
        jp.CreateAttribute('physics:jointEnabled', Sdf.ValueTypeNames.Bool).Set(True)
        jp.CreateAttribute('physics:localPos0', Sdf.ValueTypeNames.Point3f).Set(Gf.Vec3f(cx, cy, 0.0))
        jp.CreateAttribute('physics:localPos1', Sdf.ValueTypeNames.Point3f).Set(Gf.Vec3f(cx, cy, 0.0))
        flip = Gf.Quatf(0.0, 1.0, 0.0, 0.0)          # forward axis +Z of the joint points down
        jp.CreateAttribute('physics:localRot0', Sdf.ValueTypeNames.Quatf).Set(flip)
        jp.CreateAttribute('physics:localRot1', Sdf.ValueTypeNames.Quatf).Set(flip)
        points.append(Sdf.Path(path))
    prim.GetRelationship(rs.Relations.ATTACHMENT_POINTS.name).SetTargets(points)
    prim.GetAttribute(rs.Attributes.MAX_GRIP_DISTANCE.name).Set(gr.max_grip_distance)
    prim.GetAttribute(rs.Attributes.COAXIAL_FORCE_LIMIT.name).Set(gr.coaxial_force_limit)
    prim.GetAttribute(rs.Attributes.SHEAR_FORCE_LIMIT.name).Set(gr.shear_force_limit)
    prim.GetAttribute(rs.Attributes.RETRY_INTERVAL.name).Set(gr.retry_interval)


def _fixed_cameras(stage, cell: Cell):
    p = cell.pallet
    # Both overhead cameras are turned 90 deg: the long side of the image spans y, where A's spill
    # and the handover strips beside B lie.
    r = math.sqrt(0.5)
    for key, c, q in (('overhead_a', p.a_center, (r, 0, 0, r)), ('overhead_b', p.b_center, (r, 0, 0, r))):
        ub.camera(stage, CAMERAS[key], (c[0], c[1], cell.overhead_camera.height), q, focal_mm=cell.overhead_camera.focal_mm)
    uc = cell.under_camera
    ub.camera(stage, CAMERAS['under'], uc.position, (0.0, 1.0, 0.0, 0.0), focal_mm=uc.focal_mm)   # looks straight up
    ub.camera(stage, CAMERAS['presentation'], (0.0, -5.2, 3.3), _look_quat((0.0, -5.2, 3.3), (0.0, 0.0, 0.7)), focal_mm=16.0)
    a = p.a_center
    side_eye = (a[0] - 0.4, a[1] - 2.9, 0.75)
    ub.camera(stage, CAMERAS['side'], side_eye, _look_quat(side_eye, (a[0] + 0.25, a[1], 0.35)), focal_mm=18.0)
    # In the tool frame (the gantry's yaw body): beside and above the cups, looking at the case they hold.
    close_eye = (0.62, -0.62, 0.22)
    ub.camera(stage, CAMERAS['closeup'], close_eye, _look_quat(close_eye, (0.0, 0.0, -0.14)), focal_mm=14.0)


def _look_quat(eye, target):
    """USD camera looks along -Z with +Y up; build a quaternion (w, x, y, z) looking at target."""
    f = np.array(target, float) - np.array(eye, float)
    f /= np.linalg.norm(f)
    z = -f
    x = np.cross((0, 0, 1), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    q = Gf.Matrix3d(*x, *y, *z).ExtractRotation().GetQuat()   # rows are the camera axes in world
    return (q.GetReal(), *q.GetImaginary())


def build(stage: Usd.Stage, cell: Cell, spawns) -> dict:
    ub.define_stage_units(stage)
    UsdGeom.Xform.Define(stage, ROOT)
    UsdGeom.Xform.Define(stage, CELL)
    _physics_scene(stage, cell)
    _lights(stage)
    _floor(stage, cell)
    _scenery(stage)
    _pallets(stage, cell)
    _cases(stage, cell, spawns)
    _gantry(stage, cell)
    _fork(stage, cell)
    _suction(stage, cell)
    _fixed_cameras(stage, cell)
    return {'cases': [s.case_id for s in spawns]}
