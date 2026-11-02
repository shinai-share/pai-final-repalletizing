"""Typed cell configuration loaded from TOML. Geometry is validated so that a wrong
file fails loudly before any simulation starts."""
from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .appearance import delta_e

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CELL = REPO_ROOT / 'configs' / 'cell.toml'


@dataclass(frozen=True)
class Sku:
    id: str
    name: str
    size: tuple
    mass: float
    friction: tuple
    com_uncertainty: float = 0.0     # m: how far the contents may put the centre of mass from the box centre (product knowledge)


@dataclass(frozen=True)
class Pallet:
    size: tuple
    a_center: tuple
    b_center: tuple
    gap: float
    friction: tuple

    @property
    def top(self) -> float:
        return self.size[2]


@dataclass(frozen=True)
class Gantry:
    origin_height: float
    x_range: tuple
    y_range: tuple
    z_range: tuple
    yaw_range: tuple
    pitch_range: tuple
    roll_range: tuple
    v_max: tuple
    a_max: tuple
    beam_height: float
    tool_size: tuple
    cup_offsets: tuple
    travel_clearance: float
    slow_speed: float
    slow_accel: float
    touch_speed: float
    fine_touch_speed: float
    drive_stiffness: float
    drive_damping: float
    wrist_torque_limit: float
    park_z: float                 # z joint value of the park pose (between the pallets, high)

    @property
    def limits(self):
        return (self.x_range, self.y_range, self.z_range, self.yaw_range, self.pitch_range, self.roll_range)


@dataclass(frozen=True)
class ToolCamera:
    offset: tuple
    pitch_deg: float
    focal_mm: float
    aperture_mm: float
    resolution: tuple


@dataclass(frozen=True)
class DownCamera:
    offset: tuple
    focal_mm: float
    resolution: tuple
    every_s: float
    latency_s: float


@dataclass(frozen=True)
class OverheadCamera:
    height: float
    focal_mm: float
    resolution: tuple
    resolution_b: tuple = ()      # B's camera checks placements to a degree: finer pixels than A's


@dataclass(frozen=True)
class UnderCamera:
    position: tuple
    focal_mm: float
    resolution: tuple


@dataclass(frozen=True)
class Gripper:
    coaxial_force_limit: float      # where the simulated cup lets go (physics)
    shear_force_limit: float
    planning_coaxial_limit: float   # what the robot plans on (leakage allowed for)
    planning_shear_limit: float
    max_grip_distance: float
    retry_interval: float
    rated_payload: float
    max_face_tilt_deg: float
    cup_tilt_stiffness: float
    cup_twist_stiffness: float


@dataclass(frozen=True)
class Layout:
    cols: int
    rows: int
    layers: int
    interlock: bool


@dataclass(frozen=True)
class Physics:
    dt: float
    position_iterations: int
    velocity_iterations: int


@dataclass(frozen=True)
class Transport:
    brake_decel: tuple
    lateral_amp: tuple
    lateral_freq: tuple
    vertical_amp: tuple
    vertical_freq: tuple
    duration: float
    wrap_failure_layers: tuple


@dataclass(frozen=True)
class Handover:
    spots: tuple
    spot_yaw_deg: float


@dataclass(frozen=True)
class Variation:
    com_xy: float
    com_z: float
    mass: tuple


@dataclass(frozen=True)
class Staging:
    tables: tuple
    table_size: tuple
    com_allowance: float


@dataclass(frozen=True)
class Fork:
    """Two L-shaped under-supports beside the cups (tool frame: x the plate's long side, y its short
    side, z up from the cup face). Each bar comes down on the -y side and its tine slides in +y under
    the held case. Lengths in m, forces in N."""
    bar_x: float                  # bars at x = +-bar_x
    bar_y: float                  # both bars at this y (negative: the side away from the hand camera)
    bar_size: tuple               # x, y, z of a bar
    tine_size: tuple              # x (width), y (length), z (thickness) of a tine
    extension_range: tuple        # tine top below the cup face: retracted (negative, above) to fully down
    insert_range: tuple           # tine travel along +y from the bar
    grip_gap: float               # held case's top face below the cup face
    clearance: float              # tine top below the case bottom before it slides in
    z_force_limit: float          # per bar drive
    y_force_limit: float          # per tine drive
    drive_stiffness: float        # N/m (force-type drive)
    drive_damping: float
    support_fraction: float       # share of the measured weight the tines take when they push up
    tine_load_limit: float        # one tine
    insert_force_limit: float     # resistance that means the tine met something while sliding in
    pull_force_limit: float       # resistance while pulling out that means the tines drag the case
    z_speed: float                # m/s of a bar
    y_speed: float                # m/s of a tine
    friction: tuple               # tine pad on board: static, dynamic
    pad_stiffness: float          # compliant contact of the tine pad (N/m)
    pad_damping: float
    part_mass: float              # each bar and each tine


@dataclass(frozen=True)
class Structure:
    """Where the gantry's sliding parts run in guides (shinai_pi/structure.py). Lengths in m."""
    ram_length: float
    ram_section: float            # square steel tube, outside
    ram_wall: float
    ram_min_engagement: float     # of the ram that stays in the trolley at every height
    trolley_size: tuple           # x, y, z: its height is the ram's guide
    yoke_height: float            # the yoke bolted face to face on the plate's -y side, from the cup face up
    fork_guide_height: float      # each bar runs through a guide fixed to the yoke's face, from the cup face up
    fork_guide_wall: float
    fork_min_engagement: float    # of the bar that stays in its guide at every extension
    tine_sleeve: tuple            # y from the bar's centre, behind and in front: the sleeve each tine runs through
    tine_sleeve_lid: float        # its top above the tine (nothing below the tine: it slides just over surfaces)
    tine_min_engagement: float
    steel_density: float


@dataclass(frozen=True)
class Contact:
    cup_load_gain: float           # N per unit of the up-down axis reading, for a load hanging on the cups
    link_force_gain: float         # N per unit, for a force on the plate (a touch); 0 = not calibrated
    touch_detect: float            # N at which the empty cups count a face as touched
    touch_static: float
    touch_peak: float
    landing_static: float
    landing_peak: float
    collision: float


@dataclass(frozen=True)
class Appearance:
    """Colours people see (linear RGB, 0 to 1). The robot judges by depth and labels, never colour."""
    pallet_color: tuple
    floor_color: tuple
    case_colors: tuple
    case_print_color: tuple
    frame_color: tuple


@dataclass(frozen=True)
class Rendering:
    """Pictures for people: the presentation view and the two insets beside it."""
    people_resolution: tuple
    inset_resolution: tuple
    antialiasing: int             # Isaac: 0 off, 1 TAA, 2 FXAA, 3 DLSS, 4 RTXAA
    people_subframes: int         # render passes per saved picture: lets shadows and edges settle


@dataclass(frozen=True)
class Cell:
    pallet: Pallet
    handover: Handover
    staging: Staging
    variation: Variation
    gantry: Gantry
    tool_camera: ToolCamera
    down_camera: DownCamera
    overhead_camera: OverheadCamera
    under_camera: UnderCamera
    gripper: Gripper
    layout: Layout
    skus: tuple
    physics: Physics
    transport: Transport
    fork: Fork
    structure: Structure
    contact: Contact
    appearance: Appearance
    rendering: Rendering
    source: str = field(default='')

    def sku(self, sku_id: str) -> Sku:
        for s in self.skus:
            if s.id == sku_id:
                return s
        raise KeyError(f'unknown sku {sku_id}')


def _tuple(v):
    return tuple(_tuple(x) for x in v) if isinstance(v, list) else v


def _make(cls, d):
    return cls(**{k: _tuple(v) for k, v in d.items()})


MIN_CASE_PALLET_DELTA_E = 25.0     # CIELAB: an obvious difference at a glance


def validate(cell: Cell) -> None:
    p = cell.pallet
    edge_gap = abs(p.b_center[0] - p.a_center[0]) - p.size[0]
    if not math.isclose(edge_gap, p.gap, abs_tol=1e-6):
        raise ValueError(f'pallet edge gap is {edge_gap:.3f} m, config says {p.gap} m')
    g = cell.gantry
    for lo, hi in g.limits:
        if not lo < hi:
            raise ValueError('gantry ranges must be increasing')
    reach_x = (min(p.a_center[0], p.b_center[0]) - p.size[0] / 2, max(p.a_center[0], p.b_center[0]) + p.size[0] / 2)
    if reach_x[0] < g.x_range[0] or reach_x[1] > g.x_range[1]:
        raise ValueError('gantry x range does not cover both pallets')
    if p.size[1] / 2 > g.y_range[1] or -p.size[1] / 2 < g.y_range[0]:
        raise ValueError('gantry y range does not cover the pallets')
    if not len(g.v_max) == len(g.a_max) == len(g.limits):
        raise ValueError('every gantry axis needs a velocity and an acceleration limit')
    if any(v <= 0 for v in (*g.v_max, *g.a_max)):
        raise ValueError('velocity and acceleration limits must be positive')
    if not g.z_range[0] <= g.park_z <= g.z_range[1]:
        raise ValueError('the park height lies outside the z travel')
    for s in cell.skus:
        if any(d <= 0 for d in s.size) or s.mass <= 0:
            raise ValueError(f'sku {s.id} has non-positive size or mass')
    gr = cell.gripper
    if gr.planning_coaxial_limit > gr.coaxial_force_limit or gr.planning_shear_limit > gr.shear_force_limit:
        raise ValueError('the robot cannot plan on more than a cup holds')
    f = cell.fork
    tallest = max(sk.size[2] for sk in cell.skus)          # only upright cases ride on the tines (executive)
    if f.extension_range[1] < tallest + f.grip_gap + f.clearance:
        raise ValueError('the fork cannot reach under the tallest upright case')
    if -f.extension_range[0] - f.tine_size[2] < g.tool_size[2]:
        raise ValueError('retracted, the fork tines hang below the back of the plate')
    held_depth = max(sorted(sk.size, reverse=True)[1] for sk in cell.skus)
    if abs(f.bar_y) - f.bar_size[1] / 2 <= held_depth / 2 + 0.02:
        raise ValueError('the fork bar would touch the long face of a held case')
    from .structure import check
    loose = check(cell)
    if loose:
        raise ValueError(f'gantry structure: {loose}')
    look = cell.appearance
    colours = [look.pallet_color, look.floor_color, look.case_print_color, look.frame_color, *look.case_colors]
    if any(len(c) != 3 or not all(0.0 <= v <= 1.0 for v in c) for c in colours):
        raise ValueError('a colour must be three values from 0 to 1 (linear RGB)')
    if any(delta_e(c, look.pallet_color) < MIN_CASE_PALLET_DELTA_E for c in look.case_colors):
        raise ValueError('the pallet colour is too close to a case colour for people to tell them apart')
    lay = cell.layout
    s0 = cell.skus[0]
    if lay.cols * s0.size[0] > p.size[0] + 1e-6 or lay.rows * s0.size[1] > p.size[1] + 1e-6:
        raise ValueError('layout overhangs the pallet')


def load_cell(path: Path | str = DEFAULT_CELL) -> Cell:
    path = Path(path)
    d = tomllib.loads(path.read_text(encoding='utf-8'))
    cell = Cell(
        pallet=_make(Pallet, d['pallet']),
        handover=_make(Handover, d['handover']),
        staging=_make(Staging, d['staging']),
        variation=_make(Variation, d['variation']),
        gantry=_make(Gantry, d['gantry']),
        tool_camera=_make(ToolCamera, d['tool_camera']),
        down_camera=_make(DownCamera, d['down_camera']),
        overhead_camera=_make(OverheadCamera, d['overhead_camera']),
        under_camera=_make(UnderCamera, d['under_camera']),
        gripper=_make(Gripper, d['gripper']),
        layout=_make(Layout, d['layout']),
        skus=tuple(_make(Sku, s) for s in d['sku']),
        physics=_make(Physics, d['physics']),
        transport=_make(Transport, d['transport']),
        fork=_make(Fork, d['fork']),
        structure=_make(Structure, d['structure']),
        contact=_make(Contact, d['contact']),
        appearance=_make(Appearance, d['appearance']),
        rendering=_make(Rendering, d['rendering']),
        source=path.name,
    )
    validate(cell)
    return cell
