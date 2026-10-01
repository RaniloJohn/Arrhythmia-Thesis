"""
Pre-print audit of the exported STLs.

    python printcheck.py

build.py checks that the DESIGN is right. This checks that the FILES are
printable, which is a different question and the one a print shop will fail
you on. It reads the exported meshes, not the CAD solids, because the mesh is
what actually gets sliced.

Checks, in the order a shop would hit them:

  1. Watertight  - every edge shared by exactly two triangles. A mesh with
                   holes slices into garbage or is rejected outright.
  2. Orientation - consistent winding, no inverted normals.
  3. Degenerate  - zero-area facets, which some slicers choke on.
  4. Bed fit     - against a small FDM bed and a desktop SLA vat.
  5. Overhangs   - downward-facing area steeper than the support threshold,
                   reported per part so the print orientation can be chosen.
  6. Thin walls  - the narrowest deliberate feature, against the nozzle.
"""

import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from stl import mesh as stlmesh

import params as P

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

# Parts that actually go to the shop. ref_* are reference solids for viewing
# the assembly and must never be printed.
PRINTABLE = ["body", "lid", "shim_0p4mm", "shim_0p8mm", "shim_1p2mm",
             "board_gauge",
             "fit_gauge"]

SUPPORT_ANGLE = 45.0          # degrees from horizontal
FDM_BED = (220.0, 220.0, 250.0)
SLA_VAT = (143.0, 89.0, 175.0)


class Result:
    def __init__(self):
        self._fail, self._warn, self._ok = [], [], []

    def fail(self, m):
        self._fail.append(m)

    def warn(self, m):
        self._warn.append(m)

    def ok(self, m):
        self._ok.append(m)

    def report(self, title):
        print(f"\n{title}")
        for m in self._ok:
            print(f"   [ok]   {m}")
        for m in self._warn:
            print(f"   [warn] {m}")
        for m in self._fail:
            print(f"   [FAIL] {m}")
        return not self._fail


def audit(name, path, r):
    m = stlmesh.Mesh.from_file(path)
    tris = m.vectors
    n_tri = len(tris)

    # --- watertight -------------------------------------------------
    # Quantise vertices before hashing: STL stores float32, so two facets
    # meeting at a corner can disagree in the last bit and look like a hole
    # that is not there.
    q = np.round(tris.reshape(-1, 3).astype(np.float64), 4)
    verts = {}
    idx = np.empty(len(q), dtype=np.int64)
    for i, v in enumerate(map(tuple, q)):
        if v not in verts:
            verts[v] = len(verts)
        idx[i] = verts[v]
    idx = idx.reshape(-1, 3)

    edges = defaultdict(int)
    directed = defaultdict(int)
    for a, b, c in idx:
        for u, v in ((a, b), (b, c), (c, a)):
            edges[(min(u, v), max(u, v))] += 1
            directed[(u, v)] += 1

    boundary = [e for e, n in edges.items() if n == 1]
    nonmanifold = [e for e, n in edges.items() if n > 2]

    if boundary:
        r.fail(f"{name}: not watertight - {len(boundary)} open edge(s)")
    elif nonmanifold:
        r.fail(f"{name}: non-manifold - {len(nonmanifold)} edge(s) shared by "
               f">2 facets")
    else:
        r.ok(f"{name}: watertight, {n_tri} facets")

    # --- winding ----------------------------------------------------
    bad = [e for e, n in directed.items() if n > 1]
    if bad:
        r.fail(f"{name}: inconsistent winding on {len(bad)} edge(s) - "
               f"some normals are inverted")

    # --- degenerate facets ------------------------------------------
    cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    areas = np.linalg.norm(cross, axis=1) / 2.0
    zero = int((areas < 1e-9).sum())
    if zero:
        r.warn(f"{name}: {zero} zero-area facet(s)")

    # --- bed fit ----------------------------------------------------
    pts = tris.reshape(-1, 3)
    size = pts.max(axis=0) - pts.min(axis=0)
    # (size is used by the first-layer check further down)
    dims = sorted(size)
    for label, bed in (("FDM bed", FDM_BED), ("SLA vat", SLA_VAT)):
        if all(d <= b for d, b in zip(dims, sorted(bed))):
            pass
        else:
            r.fail(f"{name}: does not fit the {label}")

    # --- overhangs --------------------------------------------------
    # Facet normals pointing downward more than the support threshold.
    lens = np.linalg.norm(cross, axis=1)
    lens[lens == 0] = 1.0
    nz = cross[:, 2] / lens
    thresh = -math.cos(math.radians(90.0 - SUPPORT_ANGLE))
    steep = nz < thresh

    # A part's own bed-contact face points straight down and is steep by any
    # measure, but it rests ON the plate and needs no support. Counting it
    # made every flat part - lid, shims, gauge - look like it needed 40%
    # support, which is exactly the wrong instruction to hand a print shop.
    zmin = pts[:, 2].min()
    centroid_z = tris[:, :, 2].mean(axis=1)
    on_bed = (centroid_z - zmin) < 0.05
    needs_support = steep & ~on_bed

    total = float(areas.sum())
    sup_area = float(areas[needs_support].sum())
    bed_area = float(areas[on_bed & (nz < -0.9)].sum())
    pct = 100.0 * sup_area / total if total else 0.0

    # First-layer footprint. A tall part balanced on a small contact patch
    # detaches or warps; the fix is a brim, but the shop has to be told.
    if size[2] > 5.0 and bed_area < 150.0:
        r.warn(f"{name}: only {bed_area:.0f} mm2 touches the bed in this "
               f"orientation, on a part {size[2]:.1f} mm tall - specify a brim")

    if pct > 20.0:
        r.warn(f"{name}: {pct:.0f}% of surface needs support "
               f"({sup_area:.0f} mm2) - review the print orientation")
    elif sup_area > 1.0:
        r.ok(f"{name}: {pct:.0f}% needs support ({sup_area:.0f} mm2); "
             f"{bed_area:.0f} mm2 rests on the bed")
    else:
        r.ok(f"{name}: no supported overhangs; "
             f"{bed_area:.0f} mm2 rests on the bed")

    return size


def main():
    print()
    print("=" * 70)
    print("  PRE-PRINT AUDIT")
    print("=" * 70)

    missing = [n for n in PRINTABLE
               if not os.path.exists(os.path.join(OUT, n + ".stl"))]
    if missing:
        print(f"\n  Missing STLs: {', '.join(missing)}")
        print("  Run:  python build.py --assembly\n")
        return 1

    r = Result()
    sizes = {}
    for n in PRINTABLE:
        sizes[n] = audit(n, os.path.join(OUT, n + ".stl"), r)
    ok = r.report("MESH INTEGRITY AND PRINTABILITY")

    # --- deliberate thin features -----------------------------------
    # Vertical walls are limited by extrusion width; a flat part's thickness
    # is limited by layer height instead. Judging a 0.4 mm shim against the
    # nozzle would condemn a feature that is simply two layers tall.
    print("\nTHINNEST DELIBERATE FEATURES")
    nozzle, layer = 0.4, 0.2
    walls = [
        ("floor under the sensor", P.FLOOR),
        ("side wall", P.WALL),
        ("lid", P.LID_T),
        ("boss wall around insert", (P.BOSS_D - P.INSERT_D) / 2),
        ("strap-bearing wall", 1.75),
        ("light-seal ring", P.SEAL_RING_W),
    ]
    print("   vertical walls          (limited by the 0.40 mm nozzle)")
    for label, v in sorted(walls, key=lambda t: t[1]):
        lanes = v / nozzle
        note = "OK" if lanes >= 2.0 else "UNDER 2 EXTRUSION WIDTHS"
        print(f"     {label:<26} {v:5.2f} mm   {lanes:4.1f} x nozzle   {note}")
    print("   flat thicknesses        (limited by the 0.20 mm layer)")
    for t in sorted(P.SHIM_THICKNESSES):
        layers = t / layer
        note = "OK" if layers >= 2.0 else "UNDER 2 LAYERS"
        print(f"     {'shim ' + format(t, '.1f') + ' mm':<26} {t:5.2f} mm   "
              f"{layers:4.1f} layers    {note}")

    # --- orientation guidance ---------------------------------------
    print("\nRECOMMENDED PRINT ORIENTATION")
    print("   body        contact face DOWN on the bed.")
    print("               Puts the sensor boss and seal ring on the build")
    print("               surface where they come out sharpest, and lays the")
    print("               strap slots' load path across layers rather than")
    print("               along them.")
    print("   lid         outside face DOWN. The OLED window edge stays crisp.")
    print("   shims       flat, any orientation. Print all three.")
    print("   fit_gauge   flat. Print this FIRST and on its own.")

    print("\nBUILD SIZES")
    for n in PRINTABLE:
        s = sizes[n]
        print(f"   {n:<14} {s[0]:6.2f} x {s[1]:6.2f} x {s[2]:6.2f} mm")

    print()
    print("=" * 70)
    print("  MESH AUDIT PASSED" if ok else "  MESH AUDIT FAILED")
    print("=" * 70)
    print()
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        code = main()
    finally:
        sys.stdout.flush()
    os._exit(code)
