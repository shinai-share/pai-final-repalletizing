"""Small USD authoring helpers (pxr only, no Kit runtime needed beyond pxr)."""
from __future__ import annotations

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


def set_pose(prim, pos=(0, 0, 0), yaw_deg=0.0, quat=None):
    x = UsdGeom.Xformable(prim)
    x.ClearXformOpOrder()
    x.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*pos))
    if quat is not None:
        x.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(*quat))
    elif yaw_deg:
        x.AddRotateZOp().Set(float(yaw_deg))


def material(stage, path, color, friction=None, restitution=0.0, roughness=0.6, metallic=0.0):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + '/Shader')
    sh.CreateIdAttr('UsdPreviewSurface')
    sh.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(roughness)
    if metallic:
        sh.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(metallic)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), 'surface')
    if friction is not None:
        pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        pm.CreateStaticFrictionAttr(friction[0])
        pm.CreateDynamicFrictionAttr(friction[1])
        pm.CreateRestitutionAttr(restitution)
    return mat


def box(stage, path, size, pos=(0, 0, 0), yaw_deg=0.0, mat=None, collide=True, offset=(0, 0, 0)):
    """A visual+collision box child. The parent decides whether it is a rigid body."""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    x = UsdGeom.Xformable(cube)
    x.ClearXformOpOrder()
    x.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*(p + o for p, o in zip(pos, offset))))
    if yaw_deg:
        x.AddRotateZOp().Set(float(yaw_deg))
    x.AddScaleOp().Set(Gf.Vec3f(*size))
    if collide:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    if mat is not None:
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat, UsdShade.Tokens.weakerThanDescendants, 'physics')
        UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(mat)
    return cube


def rigid_body(stage, path, pos=(0, 0, 0), mass=None, kinematic=False, yaw_deg=0.0, quat=None, gravity=True):
    xf = UsdGeom.Xform.Define(stage, path)
    set_pose(xf.GetPrim(), pos, yaw_deg, quat)
    rb = UsdPhysics.RigidBodyAPI.Apply(xf.GetPrim())
    if kinematic:
        rb.CreateKinematicEnabledAttr(True)
    if mass is not None:
        UsdPhysics.MassAPI.Apply(xf.GetPrim()).CreateMassAttr(float(mass))
    if not gravity:
        PhysxSchema.PhysxRigidBodyAPI.Apply(xf.GetPrim()).CreateDisableGravityAttr(True)
    return xf.GetPrim()


def fixed_joint(stage, path, body0, body1, local0=(0, 0, 0), local1=(0, 0, 0)):
    """With body0=None the joint anchors body1 to the world; local0 is then a world position."""
    j = UsdPhysics.FixedJoint.Define(stage, path)
    if body0:
        j.CreateBody0Rel().SetTargets([body0])
    j.CreateBody1Rel().SetTargets([body1])
    j.CreateLocalPos0Attr(Gf.Vec3f(*local0))
    j.CreateLocalPos1Attr(Gf.Vec3f(*local1))
    return j


def drive_joint(stage, path, kind, body0, body1, axis, limits, stiffness, damping, drive_type='acceleration',
                max_force=None):
    """Prismatic or revolute joint with a position drive: acceleration-type (gains independent of
    the moving mass) or force-type with a force cap (the drive pushes no harder than `max_force`).

    Bodies share their origin at joint zero, so local frames are identity."""
    if kind == 'prismatic':
        j = UsdPhysics.PrismaticJoint.Define(stage, path)
        drive_name = 'linear'
    else:
        j = UsdPhysics.RevoluteJoint.Define(stage, path)
        drive_name = 'angular'
    j.CreateAxisAttr(axis)
    j.CreateBody0Rel().SetTargets([body0])
    j.CreateBody1Rel().SetTargets([body1])
    j.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    lo, hi = limits
    if kind == 'revolute':        # USD revolute limits are degrees
        import math
        lo, hi = math.degrees(lo), math.degrees(hi)
    j.CreateLowerLimitAttr(float(lo))
    j.CreateUpperLimitAttr(float(hi))
    if kind == 'revolute':        # USD angular drive gains are per degree
        import math
        stiffness, damping = stiffness * math.pi / 180, damping * math.pi / 180
    d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), drive_name)
    d.CreateTypeAttr(drive_type)
    if max_force is not None:
        d.CreateMaxForceAttr(float(max_force))
    d.CreateStiffnessAttr(float(stiffness))
    d.CreateDampingAttr(float(damping))
    d.CreateTargetPositionAttr(0.0)
    return j


def filter_pair(prim_a, prim_b):
    UsdPhysics.FilteredPairsAPI.Apply(prim_a).CreateFilteredPairsRel().AddTarget(prim_b.GetPath())


def camera(stage, path, pos, quat_wxyz, focal_mm=18.0, h_aperture_mm=20.955, clip=(0.05, 20.0)):
    cam = UsdGeom.Camera.Define(stage, path)
    cam.CreateFocalLengthAttr(focal_mm)
    cam.CreateHorizontalApertureAttr(h_aperture_mm)
    cam.CreateClippingRangeAttr(Gf.Vec2f(*clip))
    set_pose(cam.GetPrim(), pos, quat=quat_wxyz)
    return cam


def define_stage_units(stage: Usd.Stage):
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
