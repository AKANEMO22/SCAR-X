"""Prove exactly which preprocessing step destroys colour, and how much is lost.

The handover asks to prove which step loses colour and to separate
"input lacks information" from "neurons failed to learn". This isolates the
pipeline step by step on the 36 owned RGB fixtures:

  S0  raw RGB                        128x128x3
  S1  after convert('L') only        128x128      (grayscale, no resize)
  S2  grayscale resized to 64x64     64x64        (the intermediate `small`)
  S3  grayscale resized to 8x8       8x8 = 64     (the actual model input)

At each stage it measures how many distinct scenes remain distinguishable, and
for the colour questions specifically, whether two scenes that differ ONLY in
colour still differ at that stage. That is the information available to the
network — a ceiling no learning rule can exceed.

Also reports an RGB-preserving alternative as a CEILING REFERENCE ONLY. It is not
an approved architecture, adds no model, and is computed with numpy on the same
pixels the core already receives.
"""
import base64
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

DATA = ROOT / 'data' / 'candidates' / '20261110T142721Z-adaptive-curriculum-grounding-v1-a7c646' / 'dataset.json'
data = json.loads(DATA.read_text(encoding='utf-8'))
scenes = data['scenes']
imgdir = DATA.parent / 'images'

print('=== PIPELINE, STEP BY STEP ===')
print('  Source: src/scarx/app/pilots.py :: PilotLab.pixels')
print("    gray = image.convert('L')                 <-- colour discarded HERE")
print('    small = gray.resize((64,64))')
print('    pixels = gray.resize((8,8))/255*16        <-- 8x8 = 64 numbers')
print()

stages = {k: {} for k in ('S0_rgb', 'S1_gray', 'S2_gray64', 'S3_gray8', 'S3_rgb8')}


def stage_keys(path):
    raw = path.read_bytes()
    with Image.open(io.BytesIO(raw)) as image:
        rgb = image.convert('RGB')
        gray = rgb.convert('L')
        s0 = np.asarray(rgb, np.uint8).tobytes()
        s1 = np.asarray(gray, np.uint8).tobytes()
        s2 = np.asarray(gray.resize((64, 64), Image.BILINEAR), np.uint8).tobytes()
        g8 = np.asarray(gray.resize((8, 8), Image.BILINEAR), np.float32) / 255 * 16
        s3 = g8.tobytes()
        # RGB-preserving reference: 8x8 per channel = 192 numbers (ceiling only)
        r8 = np.asarray(rgb.resize((8, 8), Image.BILINEAR), np.uint8).tobytes()
    return {'S0_rgb': s0, 'S1_gray': s1, 'S2_gray64': s2,
            'S3_gray8': s3, 'S3_rgb8': r8}


by_name = {}
for png in sorted(imgdir.glob('*.png')):
    by_name[png.name] = stage_keys(png)

for stage in stages:
    groups = defaultdict(list)
    for name, keys in by_name.items():
        groups[keys[stage]].append(name)
    distinct = len(groups)
    print(f'  {stage:12s} distinct representations: {distinct:2d} / 36 images')

print()
print('=== COLOUR-ONLY PAIRS: does the stage still tell them apart? ===')


def scene_by_file(name):
    for s in scenes:
        if Path(s['file']).name == name or s['file'] == name:
            return s
    return None


pair_total = 0
pair_broken = {k: 0 for k in stages}
for group in {s['counterfactual_group'] for s in scenes}:
    members = [s for s in scenes if s['counterfactual_group'] == group]
    if len(members) != 2:
        continue
    a, b = members
    geometry_a = json.dumps([(o['shape'], o['x'], o['y'], o['halfsize']) for o in a['objects']])
    geometry_b = json.dumps([(o['shape'], o['x'], o['y'], o['halfsize']) for o in b['objects']])
    if geometry_a != geometry_b:
        continue  # not a pure colour swap
    na, nb = Path(a['file']).name, Path(b['file']).name
    if na not in by_name or nb not in by_name:
        continue
    pair_total += 1
    for stage in stages:
        if by_name[na][stage] == by_name[nb][stage]:
            pair_broken[stage] += 1

print(f'  pure colour-swap pairs found: {pair_total}')
for stage in stages:
    lost = pair_broken[stage]
    print(f'  {stage:12s} indistinguishable: {lost:2d}/{pair_total}'
          f'   {"<-- COLOUR DESTROYED" if lost == pair_total else ""}')

print()
print('=== ATTRIBUTION ===')
if pair_broken['S1_gray'] == pair_total:
    print("  convert('L') alone already merges every colour pair.")
    print('  => the loss happens at the RGB->grayscale conversion, BEFORE any resize.')
else:
    print("  convert('L') keeps some pairs apart; resize contributes further loss.")
print()
print(f"  S3_gray8 (actual model input) distinguishes {len({v['S3_gray8'] for v in by_name.values()})}/36 scenes")
print(f"  S3_rgb8 (colour kept, 192 numbers) distinguishes {len({v['S3_rgb8'] for v in by_name.values()})}/36 scenes")
print()
print('  This is an INPUT-INFORMATION result, not a neuron result. No learning rule')
print('  can recover a distinction that is absent from the numbers it receives.')
print('  S3_rgb8 is a ceiling reference only; it is not an approved encoder.')

out = ROOT / 'local' / 'color-loss-stages.json'
out.write_text(json.dumps({
    'distinct_per_stage': {s: len({v[s] for v in by_name.values()}) for s in stages},
    'colour_pairs': pair_total,
    'colour_pairs_merged': pair_broken,
    'note': 'input-information ceiling; not model accuracy; rgb8 is reference only',
}, indent=2), encoding='utf-8')
print(f'\nwrote {out}')
