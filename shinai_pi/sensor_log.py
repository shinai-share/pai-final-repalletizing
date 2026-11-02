"""What the robot's own cameras saw, kept with the run, and its estimate drawn over them (owner 2026-10-31: show how the
camera picture is recognised by the robot).

Each capture the robot makes is written as the picture, the segmentation (which pixels belong to which object: in
the simulator this comes from the renderer, not from a learned detector), the depth in millimetres, and the camera.
overlay() draws the robot's own estimate over a picture: the top face it fitted to each case, coloured by how likely
taking it is to bring others down, and the case it chose."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def project(frame, pts):
    """Pixel (u, v) of world points and whether each lies in front of the camera (CameraFrame conventions)."""
    p = np.hstack([np.asarray(pts, float), np.ones((len(pts), 1))]) @ frame.view
    z = -p[:, 2]
    front = z > 1e-9
    zs = np.where(front, z, 1.0)
    u = frame.fx * p[:, 0] / zs + frame.width / 2 - 0.5
    v = -frame.fy * p[:, 1] / zs + frame.height / 2 - 0.5
    return u, v, front


def _colour(key: str):
    if not key.startswith('case_'):
        return (70, 70, 70)
    h = hashlib.sha256(key.encode()).digest()
    return tuple(int(60 + c % 180) for c in h[:3])


def segmentation_picture(frame) -> np.ndarray:
    out = np.zeros((frame.height, frame.width, 3), np.uint8)
    for i, key in frame.instance_map.items():
        out[frame.instance == i] = _colour(key)
    return out


class SensorLog:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    def write(self, frame) -> str:
        name = f'{int(round(frame.stamp * 1000)):08d}_{frame.name}'
        rgb = frame.extra.get('rgb')
        if rgb is not None:
            Image.fromarray(np.asarray(rgb, np.uint8)).save(self.folder / f'{name}.jpg', quality=88)
        Image.fromarray(segmentation_picture(frame)).save(self.folder / f'{name}_seg.png')
        d = np.asarray(frame.depth, float)
        mm = np.where(np.isfinite(d) & (d > 0), np.clip(d * 1000.0, 0, 65535), 0).astype(np.uint16)
        Image.fromarray(mm).save(self.folder / f'{name}_depth.png')
        meta = {'camera': frame.name, 't': round(float(frame.stamp), 3), 'fx': frame.fx, 'fy': frame.fy,
                'width': frame.width, 'height': frame.height, 'view': np.asarray(frame.view).tolist(),
                'objects': {str(k): v for k, v in frame.instance_map.items()}}
        (self.folder / f'{name}.json').write_text(json.dumps(meta, ensure_ascii=False), encoding='utf-8')
        return name


STATE_COLOURS = {'safe': (70, 200, 110), 'risky': (240, 170, 40), 'blocked': (230, 70, 60)}   # inspector_feed's states


def overlay(rgb, frame, boxes, chosen=None, width: int = 2) -> Image.Image:
    """The picture with each estimated case's top face outlined by the robot's own call on it (green: safe to take,
    amber: takeable but may bring others down, red: not takeable now, grey: no call) and the chosen case thicker in
    white."""
    img = Image.fromarray(np.asarray(rgb, np.uint8)).convert('RGB')
    draw = ImageDraw.Draw(img)
    for b in boxes:
        poly = np.asarray(b['poly'], float)
        pts = np.hstack([poly, np.full((len(poly), 1), float(b['top']))])
        u, v, front = project(frame, pts)
        if not front.all():
            continue
        xy = list(zip(u.tolist(), v.tolist()))
        is_chosen = b['id'] == chosen
        draw.polygon(xy, outline=(255, 255, 255) if is_chosen else STATE_COLOURS.get(b.get('state'), (160, 160, 160)),
                     width=width * 2 if is_chosen else width)
    return img
