"""
Shaded PNG previews of the exported STLs.

CadQuery's own SVG export is an unshaded wireframe - technically correct and
almost unreadable for a part with this many internal features. This renders
the actual triangle mesh with simple Lambertian shading so the boss, the
seal ring, the USB-C opening and the strap lugs are all legible at a glance.

    python render.py

Writes out/render_<part>_<view>.png. Preview only - the STEP files are the
deliverable, this is just so a human can see what was built.
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from stl import mesh as stlmesh

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

# elev, azim, and a label. The skin view matters most - it is the face that
# carries the sensor boss and the light seal.
VIEWS = {
    "iso": (26, -58),
    "skin": (-88, -90),
    "top": (88, -90),
}

LIGHT = np.array([-0.4, -0.7, 0.6])
LIGHT = LIGHT / np.linalg.norm(LIGHT)


def render(stl_path, out_path, elev, azim, base=(0.30, 0.55, 0.85)):
    m = stlmesh.Mesh.from_file(stl_path)
    tris = m.vectors

    # Lambertian term per facet, lifted off zero so back faces stay visible.
    n = m.normals
    lens = np.linalg.norm(n, axis=1, keepdims=True)
    lens[lens == 0] = 1.0
    n = n / lens
    shade = np.clip(n @ LIGHT, 0.0, 1.0)
    shade = 0.35 + 0.65 * shade

    colors = np.zeros((len(tris), 4))
    colors[:, 0] = base[0] * shade
    colors[:, 1] = base[1] * shade
    colors[:, 2] = base[2] * shade
    colors[:, 3] = 1.0

    fig = plt.figure(figsize=(9, 7), dpi=130)
    ax = fig.add_subplot(111, projection="3d")
    coll = Poly3DCollection(tris, facecolors=colors, linewidths=0.08,
                            edgecolors=(0, 0, 0, 0.18))
    ax.add_collection3d(coll)

    pts = tris.reshape(-1, 3)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    ctr = (lo + hi) / 2
    span = (hi - lo).max() / 2 * 1.05
    ax.set_xlim(ctr[0] - span, ctr[0] + span)
    ax.set_ylim(ctr[1] - span, ctr[1] + span)
    ax.set_zlim(ctr[2] - span, ctr[2] + span)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    size = hi - lo
    ax.set_title(
        f"{os.path.basename(stl_path)}   "
        f"{size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f} mm",
        fontsize=10, color="#333",
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return size


def main():
    targets = [a for a in sys.argv[1:]] or ["body", "lid"]
    for name in targets:
        stl = os.path.join(OUT, f"{name}.stl")
        if not os.path.exists(stl):
            print(f"  skip {name}: no STL (run build.py first)")
            continue
        for view, (elev, azim) in VIEWS.items():
            png = os.path.join(OUT, f"render_{name}_{view}.png")
            size = render(stl, png, elev, azim)
            print(f"  {os.path.basename(png):28s} "
                  f"{size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f} mm")


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────
#  Assembly view - the enclosure with the real boards in place
# ─────────────────────────────────────────────────────────────────────

PALETTE = {
    "body": (0.62, 0.66, 0.72),
    "max30102": (0.85, 0.35, 0.30),
    "esp32c3": (0.25, 0.65, 0.40),
    "ssd1306": (0.30, 0.45, 0.85),
}


def assembly(out_path, elev, azim, alpha_body=0.22):
    """Body shown ghosted so the stacked boards inside stay visible."""
    import cadquery as cq
    import layout as L
    import parts
    import mockups

    lo = L.compute()
    shapes = {"body": parts.body(lo)}
    shapes.update(mockups.all_components(lo))

    fig = plt.figure(figsize=(10, 8), dpi=130)
    ax = fig.add_subplot(111, projection="3d")
    allpts = []

    for name, shp in shapes.items():
        tmp = os.path.join(OUT, f"_tmp_{name}.stl")
        cq.exporters.export(shp, tmp, tolerance=0.02, angularTolerance=0.2)
        m = stlmesh.Mesh.from_file(tmp)
        os.remove(tmp)

        n = m.normals
        lens = np.linalg.norm(n, axis=1, keepdims=True)
        lens[lens == 0] = 1.0
        shade = 0.4 + 0.6 * np.clip((n / lens) @ LIGHT, 0.0, 1.0)

        base = PALETTE[name]
        cols = np.zeros((len(m.vectors), 4))
        for i in range(3):
            cols[:, i] = base[i] * shade
        cols[:, 3] = alpha_body if name == "body" else 1.0

        ax.add_collection3d(Poly3DCollection(
            m.vectors, facecolors=cols, linewidths=0.05,
            edgecolors=(0, 0, 0, 0.12)))
        allpts.append(m.vectors.reshape(-1, 3))

    pts = np.vstack(allpts)
    lo_, hi_ = pts.min(axis=0), pts.max(axis=0)
    ctr = (lo_ + hi_) / 2
    span = (hi_ - lo_).max() / 2 * 1.05
    for setter, c in ((ax.set_xlim, ctr[0]), (ax.set_ylim, ctr[1]),
                      (ax.set_zlim, ctr[2])):
        setter(c - span, c + span)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title("Assembly - red: MAX30102   green: ESP32-C3   blue: SSD1306",
                 fontsize=10, color="#333")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
