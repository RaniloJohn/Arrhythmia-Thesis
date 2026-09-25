# Wrist Enclosure — Radial-Artery PPG Node

Parametric CAD source for the wearable acquisition node's housing. Every
dimension lives in `params.py`; nothing downstream hard-codes a number.

Full design rationale, wiring and the print-shop brief:
[[05 - Claude Notes/2026-09-11 - Wrist Enclosure Design Specification|Wrist Enclosure Design Specification]]

## Requirements

CadQuery 2.8 on Python 3.12 (installed at
`%LOCALAPPDATA%\Programs\Python\Python312\python.exe`):

```
python -m pip install cadquery
```

## Build

```
python build.py            # everything (needs MAX_VARIANT set)
python build.py --gauge    # Stage-1 fit gauge only
python build.py --check    # run checks, export nothing
```

Outputs land in `out/` as both `.step` and `.stl`:

- **STEP** — for the print shop, and opens as an editable solid in Fusion 360
- **STL** — for the slicer

`out/` is gitignored. The source of truth is `params.py`, not an exported mesh.

## Inspecting the fit in Fusion 360

```
python build.py --assembly      # exports the parts AND the reference solids
```

Then in Fusion: **Utilities → Scripts and Add-Ins → Scripts → green +**, point it
at `fusion/`, select **ArrhythmiaEnclosure**, press **Run**.

It imports every printed part plus reference solids for the three boards, the
mated USB-C plug and the strap band — each a named, colour-coded component.
They arrive pre-aligned, because CadQuery exports them in a shared origin with
(0, 0, 0) at the centre of the skin-contact face, so nothing needs joining or
moving.

Use **Inspect → Section Analysis** to check any clearance directly rather than
trusting a written figure. Components prefixed `REF` are not printable parts —
hide them before exporting for manufacture.

Fusion has no headless mode and its API only runs inside the running
application, so this script cannot be driven from outside. It is deterministic
though: re-run it after any `build.py` and the assembly rebuilds.

## Files

| File | Role |
|---|---|
| `params.py` | Every raw measurement. The only file you normally edit. |
| `layout.py` | Derived envelope, Z stack-up, component placement |
| `parts.py` | Solid geometry: `body`, `lid`, `shims`, `fit_gauge` |
| `build.py` | Export driver plus the pre-flight geometric checks |

## Current state

`MAX_VARIANT` is set to `"mhetlive"` and every check passes, so the parts
export. Two things are still open:

- The MAX30102's **aperture position** has not been measured, so the optical
  window is opened up by `AP_UNCERTAINTY` (±1.5 mm) to be sure the chip is
  exposed wherever it actually sits. Measure `AP_DX` / `AP_DY` and set
  `AP_UNCERTAINTY = 0` to recover the tight window.
- `TARGET_TOTAL_H = 35.0` makes the watch **35 mm tall**, on request, to get
  an 18 mm wiring plenum between the ESP32-C3 and the display. The strap
  tunnel grew with it and the band now exits at the lid seat; `build.py`
  warns about this on every build. See the design note before printing.

Set `TARGET_TOTAL_H = None` for the original 18.2 mm stack.

The **fit gauge does not depend on any of this** and can be ordered
immediately — the aperture is a property of the MAX30102 chip package, not
the breakout.

## Coordinate system

Origin at the centre of the skin-contact face.

```
+X  along the forearm, toward the ELBOW   (cable exits this way)
+Y  across the wrist,  toward the THUMB   (radial / lateral side)
+Z  away from the skin, toward the OLED
```

## Why the checks exist

A CAD boolean that silently did nothing still exports a clean-looking file.
Each check in `build.py` corresponds to a mistake that is invisible on screen
and expensive after printing. Two of them caught real defects during the
first build of this model:

- the lid's alignment lip landing on all four screw posts, and
- `body_h` including the lid's own thickness, so body and lid occupied the
  same 1.6 mm and the assembly stood a lid too tall.

Do not delete a check to make a build pass.
