# Enclosure Print Order Brief

**Date:** 2026-09-11
**For:** the 3D printing service
**Files:** `08 - Hardware/enclosure/out/` — send the `.step` files, keep `.stl` as backup
**Design record:** [[2026-09-11 - Wrist Enclosure Design Specification]]

---

## Order in two stages

**Do not order Stage 2 until Stage 1 has been checked against the real modules.** There is no
iteration loop — the hardware is at a teammate's location and the team has no physical access —
so a wrong clearance discovered on the finished enclosure means starting over.

### Stage 1 — two gauges (order these first, on their own)

| File | Size (mm) | Qty |
|---|---|---|
| `fit_gauge.step` | 72.0 × 46.0 × 3.0 | 1 |
| `board_gauge.step` | 87.8 × 44.3 × 2.5 | 1 |

**`fit_gauge`** carries the USB-C opening, the sensor aperture, the heat-set insert bore and a
screw clearance hole, **each repeated at three clearances (0.15 / 0.25 / 0.35 mm)**, plus one
strap slot. Whichever column gives a firm slip fit sets `CLEARANCE` for Stage 2.

**`board_gauge`** is a go/no-go plate with one through-pocket per board, each cut to exactly
the clearance the cavity gives it. Drop each real module through. Pockets are identified by
drilled dots: **1 = MAX30102, 2 = ESP32-C3, 3 = SSD1306.**

This second gauge exists because of a specific exposure. Every board sits in the cavity on
**0.25 mm of margin**, and every one of those outlines is a catalogue figure rather than a
caliper reading:

| Board | Assumed outline | Margin | Source |
|---|---|---|---|
| SSD1306 | 27.3 × 27.8 | 0.25 mm | catalogue — **and it sets the whole cavity** |
| ESP32-C3 | 22.5 × 18.0 | 0.25 mm | catalogue |
| MAX30102 | 21.0 × 16.0 | 0.25 mm | catalogue |

A board 0.3 mm larger than its datasheet in the wrong axis does not go in, and the enclosure is
scrap. Two cheap flat plates remove that exposure entirely.

### Stage 2 — the enclosure

| File | Size (mm) | Qty |
|---|---|---|
| `body.step` | 40.4 × 34.6 × 33.4 | 1 |
| `lid.step` | 40.4 × 31.7 × 1.6 | 1 |
| `shim_0p4mm.step` | 15.0 × 12.7 × 0.4 | 1 |
| `shim_0p8mm.step` | 15.0 × 12.7 × 0.8 | 1 |
| `shim_1p2mm.step` | 15.0 × 12.7 × 1.2 | 1 |

The body is deliberately tall: the assembled watch stands **35.0 mm** from the skin-contact
seal ring to the top of the lid, because the team asked for hand-wiring room between the
boards. It is not a slicing error and it should not be scaled down — scaling would take every
hole in the shell with it.

Print **all three shims**. They tune the sensor's skin contact pressure, which is the single
largest determinant of PPG signal quality, and they exist precisely because the team cannot
iterate on hardware.

---

## Material

**Preferred:** PETG, ABS, or PA12 nylon (MJF / SLS).

**If the shop quotes SLA resin** — common and cheapest in the Philippines — two conditions:

1. **Specify full post-cure.** Incompletely cured resin is a skin irritant, and this part is
   worn against skin for extended recordings.
2. **Tell us first.** Resin is brittle, and the wall outboard of each strap slot is 1.75 mm
   and carries the entire strap tension. For resin that wants raising to ~2.25 mm, which is a
   one-line parameter change and a re-export.

---

## Print settings and orientation

Audited with `printcheck.py` against the exported meshes.

| Part | Orientation | Notes |
|---|---|---|
| **body** | Contact face **down** | **Add a brim.** Only 44 mm² touches the bed — the light-seal ring lands first and the rest of the 16.6 mm-tall part sits above it. 13% of the surface (968 mm²) needs support, mostly the USB recess roof. |
| **lid** | Outside face **down** | No supported overhangs; 984 mm² flat on the bed. Keeps the display window edge crisp. |
| **shims** | Flat | No supports. 0.4 mm is two layers at 0.20 mm. |
| **fit_gauge** | Flat | 3045 mm² on the bed. |

- Nozzle 0.4 mm, layer 0.20 mm, **4 perimeters** (walls are dimensioned for this).
- Contact face and seal ring take priority for surface finish — they bear on skin.

### Mesh integrity

All six meshes are **watertight**, correctly wound, with no degenerate facets and no
non-manifold edges. Every part fits both a 220 mm FDM bed and a 143 × 89 mm SLA vat.

### Thinnest deliberate features

| Feature | mm | vs 0.40 mm nozzle |
|---|---|---|
| Boss wall around insert | 0.90 | 2.2× |
| Floor under the sensor | 1.20 | 3.0× |
| Light-seal ring | 1.20 | 3.0× |
| Side wall / lid | 1.60 | 4.0× |
| Strap-bearing wall | 1.75 | 4.4× |

Nothing sits under two extrusion widths.

---

## Hardware to buy separately

| Item | Spec | Qty |
|---|---|---|
| Brass heat-set inserts | M2, 3.2 mm OD, ≥4 mm long | 4 |
| Machine screws | M2 × 6 mm, pan or cap head | 4 |
| Elastic velcro band | **16 mm** wide, ≤3 mm thick | 1 |

The 16 mm band is narrower than a watch strap deliberately — the strap slot and the two screw
bosses share the body's width. `layout.py` computes the boss position from `STRAP_W` and raises
a descriptive error rather than silently producing a part where they collide.

---

## The one thing still assumed

The MAX30102's **optical aperture position** has not been measured; the photograph the team
supplied shows the back of the board. Rather than block the order, the aperture has been
**opened up by ±1.5 mm** (`AP_UNCERTAINTY`) so the chip is exposed wherever it actually sits
on the PCB. The window is now 9.10 × 6.80 mm instead of 6.10 × 3.80 mm.

This costs little: a larger window does not increase LED-to-photodiode leakage through the
housing, the seal ring still surrounds it, and the module's 21 × 16 mm PCB still rests on
plenty of floor around the opening. It is a deliberate trade of a slightly looser optical
window for the ability to print now.

**Set `AP_UNCERTAINTY = 0.0` once `AP_DX` / `AP_DY` are measured** to recover the tight window.

Six other dimensions remain catalogue values rather than caliper readings — the full list is in
the design specification. Stage 1 is what converts them.

---

## Assembly

This is **not** a snap-together kit. Someone has to build it, and on this project that someone
is whoever physically holds the hardware — so the steps below are written to be handed over.

**The circuit already exists.** The acquisition chain is live and validated on real sensor
data, so the three modules are already wired to each other. This is a housing job, not a
build-the-circuit job.

### After Stage 1

1. Drop each real module through its pocket in `board_gauge`. All three must pass.
2. Try a USB-C plug, an M2 screw and the strap through `fit_gauge`; note which clearance
   column gives a firm slip fit.
3. Set `CLEARANCE` in `params.py`, re-run `python build.py --assembly` and
   `python printcheck.py`, then order Stage 2.

### After Stage 2

4. **Fit the fasteners.** With `FASTENER_MODE = "insert"`, heat-set the four brass inserts
   into the end blocks from the lid seat, pressing straight down with a soldering iron. With
   `"selftap"`, skip this entirely — the screws cut their own thread.
5. **Check the wire lengths before stacking.** The gaps between boards are **1.2 mm**. Long
   Dupont jumpers will not fit and will hold the stack apart; the boards need short lengths of
   30 AWG silicone wire soldered flat. This is the step most likely to need rework, so check
   it before committing to an assembly session.
6. **Stack from the bottom.** MAX30102 face-down over the aperture, then the ESP32-C3 with its
   USB-C facing the port opening, then the SSD1306 face-up. A thin adhesive foam pad between
   each pair sets the spacing and damps vibration — the enclosure deliberately provides no
   internal ledges, because the boards fill the cavity to 0.25 mm and a ledge would foul them.
7. **Close and fasten.** Four M2 screws through the lid.
8. **Thread the 16 mm band** through both end slots.
9. **Tune the contact pressure.** Start with the 0.8 mm shim under the sensor and adjust from
   the live SQI reading — up if the trace is weak, down if it looks occluded.

### Choosing the fastener

| | `insert` | `selftap` |
|---|---|---|
| Tooling | soldering iron + inserts | screwdriver only |
| Open/close cycles | many | roughly 5&ndash;10 |
| Boss wall | 0.90 mm | **1.65 mm** |
| Parts to buy | 4 inserts + 4 screws | 4 screws |

`insert` is the default because the shims are meant to be swapped while tuning, and that means
repeated opening. Choose `selftap` if the assembler does not have a soldering iron they trust
for heat-setting — then tune the shim stack first and do the final assembly once.
