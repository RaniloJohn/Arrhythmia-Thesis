# Print-ready parts

Hand these to the 3D printing service. Everything here is a real part — the
reference solids used for checking the assembly in Fusion live in
`../enclosure/out/` and must **never** be sent to a printer.

Regenerated automatically by `python ../enclosure/build.py --assembly`, so this
folder cannot drift from the model. Send the `.step` files; keep `.stl` as a
fallback for shops that only accept mesh.

Full brief, including material, orientation and the hardware to buy:
[[05 - Claude Notes/2026-09-11 - Enclosure Print Order Brief|Enclosure Print Order Brief]]

---

## Order stage 1 first, on its own

**`stage-1-gauges/`** — two flat plates, about an hour of print time between
them.

| File | Size (mm) | What it settles |
|---|---|---|
| `board_gauge` | 87.8 × 44.3 × 2.5 | Whether each real module actually fits the cavity. Drop each board through its pocket. Dots identify them: **1** MAX30102, **2** ESP32-C3, **3** SSD1306. |
| `fit_gauge` | 72.0 × 46.0 × 3.0 | The right clearance. Carries the USB-C opening, sensor aperture, insert bore and screw hole at **0.15 / 0.25 / 0.35 mm** each, plus a strap slot. |

Every board currently sits on **0.25 mm of margin against a catalogue
dimension**, not a caliper reading. A board 0.3 mm larger than its datasheet in
the wrong axis does not go in and the enclosure is scrap. These two plates cost
almost nothing and remove that exposure.

Then set `CLEARANCE` in `params.py` from whichever column fits, re-run
`build.py --assembly` and `printcheck.py`, and order stage 2.

## Then stage 2

**`stage-2-enclosure/`** — the enclosure itself.

| File | Size (mm) | Qty | Orientation |
|---|---|---|---|
| `body` | 40.4 × 34.6 × 16.6 | 1 | Contact face **down**, **with a brim** — only 44 mm² touches the bed |
| `lid` | 40.4 × 31.7 × 1.6 | 1 | Outside face down |
| `shim_0p4mm` | 15.0 × 12.7 × 0.4 | 1 | Flat |
| `shim_0p8mm` | 15.0 × 12.7 × 0.8 | 1 | Flat |
| `shim_1p2mm` | 15.0 × 12.7 × 1.2 | 1 | Flat |

Print **all three shims**. Only one ends up in the device; they exist to tune
the sensor's skin contact pressure, which is the largest single determinant of
PPG signal quality.

Nozzle 0.4 mm, layer 0.20 mm, 4 perimeters. PETG, ABS or PA12 nylon preferred —
if the shop quotes SLA resin, specify **full post-cure** and tell the team
first, because the strap wall wants thickening for a brittle material.

---

## How the two halves join

There are no clips, pins or alignment lips — **four M2 screws are the only
connection**, and that is a measured constraint rather than an omission. The
boards leave 0.25 mm inside the cavity and the end blocks are packed with the
strap slots and screw bosses; the widest free strip anywhere is 1.75 mm, and a
2 mm locating pin needs about 3 mm.

To compensate, one diagonal pair of lid holes is a **close fit** so those two
screws act as dowels, holding lateral play to roughly ±0.05 mm instead of
±0.15 mm. The other two stay clearance holes: four close-fit holes in a printed
part will not all line up, and the lid would bind instead of locating.

## Assembly order

1. Heat-set four M2 brass inserts into the body's end blocks, from the lid seat.
2. Check wire lengths — the gaps between boards are **1.2 mm**, so Dupont
   jumpers will not fit; short 30 AWG silicone wire soldered flat will.
3. Stack: MAX30102 face-down over the aperture, ESP32-C3 with its USB-C toward
   the side opening, SSD1306 face-up. A thin adhesive foam pad between each
   pair sets the spacing — there are deliberately no internal ledges, because
   the boards fill the cavity too completely for one to clear them.
4. Fit one shim under the sensor, starting with 0.8 mm.
5. Close with four M2 × 6 mm screws.
6. Thread a **16 mm** elastic velcro band through both end slots.
7. Tune the shim from the live SQI reading — thicker if the trace is weak,
   thinner if it looks occluded.
