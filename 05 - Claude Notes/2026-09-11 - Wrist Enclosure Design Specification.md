# Wrist Enclosure Design Specification

**Date:** 2026-09-11
**Component:** Wearable acquisition node housing — `08 - Hardware/enclosure/`
**Status:** Geometry complete and validated; blocked on one measurement before release

---

## Why this exists

The thesis describes a **wrist-worn** PPG node, but until now the ESP32-C3, MAX30102 and
SSD1306 were loose modules on a bench. Without a housing there is no repeatable
sensor-to-skin coupling, so PPG signal quality varies between recordings for reasons nobody
can control or document. That undermines the ISO/IEC 25010 **Reliability** and **Usability**
evaluations in Chapter 5, and it makes the eventual validation dataset inconsistent once the
teammate's Colab training lands.

This note is the design record for the enclosure that fixes that.

## Constraints that shaped the design

1. **Nobody can touch the hardware.** It lives at a teammate's location, reached only over
   Tailscale. Every adjustment must be possible without a redesign.
2. **Commercial print shop, no iteration loop.** One shot. The design must tolerate being
   slightly wrong, and tolerance-critical features are proven on a cheap coupon first.
3. **USB-C tether only.** No battery, matching how the system actually runs today —
   the C3 enumerates on the Pi as `/dev/ttyACM0`.

## Form factor decision

Sensor over the **radial artery** on the volar (palm-side) wrist, thumb side. This drives
four things that a generic project box would get wrong:

| Driver | Consequence in the model |
|---|---|
| Contact pressure dominates reflectance-PPG quality | Sensor sits on a boss proud of the contact face; boss height tunable by printed shims |
| The artery is **lateral** of the flexor tendons | Sensor offset `SENSOR_OFFSET_Y = 6.0 mm` toward the thumb — centring it would read tendon |
| LED-to-photodiode crosstalk through the housing | Raised light-seal ring around the aperture |
| The strap needs both end blocks | USB-C exits **−Y**, the ulnar side, with a strain-relief collar — see below |

### The port cannot share an end block with the strap

Raised by the team and confirmed by measurement. The strap slots are vertical through-slots in
the ±X end blocks, spanning the full height, so **the band physically occupies the middle of
each end block**. With the USB-C also at +X the two overlapped by **357 mm³** — the strap would
thread straight through the space the plug needs:

```
band-volume INTERSECT plug-volume = 357.0 mm3
  overlap spans x 14.95..18.45  y -8.25..4.50  z 4.00..12.00
```

No clearance tweak fixes this. It is a topology conflict, not a tolerance one, and every
geometric check passed at the time because none of them modelled the band as a solid object —
only the slot as a hole.

**Fix: the ESP32-C3 is rotated 90°** so its USB-C faces −Y. Smartwatches resolve the same
conflict identically: straps at 12 and 6, ports and buttons at 3. The cable now leaves from the
little-finger side, away from the thumb-side sensor boss, and is routed along the forearm from
there.

Three things improved as a side effect:

- The plug now crosses only a `WALL`-thick side wall instead of a 6.3 mm end block, so the
  insertion-depth problem disappears entirely.
- Recess-to-boss clearance went from 0.70 mm to **7.45 mm** — the port sits at x = 0 while the
  bosses are at ±16.7.
- Antenna keep-out clearance improved to 1.10 mm.

The cost is a **3 mm strain-relief collar** on the −Y face, since a 1.6 mm side wall grips a
plug overmould far less well than a 6.3 mm end block did. Overall width becomes 34.6 mm at the
collar; the body proper stays 31.7 mm.

The body is kept **symmetric** in Y even though the sensor is not. An asymmetric body
rotates under strap tension, and rotation moves the aperture off the artery.

### The R30 wrist curve was not implemented as specified

The brief called for an R30 transverse curve matching the volar wrist. Implemented literally,
that removes ~6 mm of material at the body edge, which breaches the 1.2 mm floor and opens
the cavity. The ergonomic goal — that the rim cannot dig in — is met instead with a **flat
contact face and a 2.0 mm perimeter chamfer**. Flat also guarantees the sensor boss seats
squarely, which matters more for signal quality than conformity to the wrist.

## Envelope

With the MH-ET LIVE MAX30102 (21 × 16 mm):

| | Before packing pass | After packing pass | **Current (35 mm build)** |
|---|---|---|---|
| Footprint | 55.4 × 31.7 mm | 40.4 × 31.7 mm | **40.4 × 31.7 mm** |
| Assembled height | 18.5 mm | 17.0 mm | **33.8 mm** |
| Overall height (with boss) | 19.7 mm | 18.2 mm | **35.0 mm** |
| Internal cavity | 27.8 × 28.5 × 15.7 mm | 27.8 × 28.5 × 14.2 mm | **27.8 × 28.5 × 31.0 mm** |

The height column changed on purpose and late — see *Raising the watch to 35 mm* below. The
footprint and every hole in the shell are unaffected.

### Where the 33% came from

A waste audit showed the enclosure was only 50% cavity by volume, with **7.2 mm of solid end
material per side** purely to carry screw bosses — and then the strap lugs bolted a further
6.6 mm per side onto that. Two blocks, two jobs, stacked end to end.

1. **Merged the strap slots into the boss end blocks (−14.4 mm).** The insert is bored 4 mm
   down from the lid seat; the strap slot is a vertical through-slot. They can share the same
   X if they are separated in **Y** instead. The cost is strap width — the slot and the two
   bosses now share the body's Y extent, so `STRAP_W` drops to 16 mm. `layout.py` computes the
   boss Y position from `STRAP_W` and raises a descriptive error if the two cannot coexist,
   so this constraint cannot be broken silently.
2. **Deleted the protruding USB strain-relief collar (−4.0 mm)** and replaced it with a plug
   recess cut into the end block. See the correction below — the first attempt at this was
   wrong and stopped the plug seating.
3. **Tightened the Z stack (−1.5 mm).** `WIRE_GAP` 1.5 → 1.2 mm, `C3_TALL_BOT` 1.0 → 0.6 mm
   (the Super Mini carries almost nothing on its underside), MH-ET LIVE back clearance
   1.5 → 1.0 mm, `BOSS_D` 5.6 → 5.0 mm.

Three new checks guard the merge, since none of it is visible in a render: the strap slot must
clear the cavity (1.05 mm), clear the insert bores (2.10 mm), and leave enough material
outboard to carry strap tension (**1.75 mm — a standing warning**, fine in PETG or PA12, thin
for brittle SLA resin; raise `BOSS_EDGE` if the shop quotes resin).

### Why it is 42 mm along the forearm

Fitting the real boards forced the screw bosses out of the cavity. The SSD1306 is
27.3 × 27.8 mm inside a 27.8 × 28.5 mm cavity — **0.25 mm of margin all round** — so a
5.6 mm screw post cannot stand anywhere inside without passing through a PCB. The first
assembly check caught exactly that: all three boards colliding with all four posts.

The bosses now sit outside the cavity, buried in solid material at the two **ends**. That
choice grows the body along the forearm, where a watch case has room, instead of across the
wrist, where width is what makes a device uncomfortable. Across-wrist width stays at 31.7 mm.
Lug-to-lug is 55.4 mm, which is normal for a large watch.

The alternative — bosses in the four corners — works out at roughly 38 × 39 mm, trading
7 mm of extra wrist width for 4 mm less length. That is the worse trade for a wearable, but
it is a single change to `screw_x`/`screw_y` in `layout.py` if the team prefers it.

### Correction — the USB-C plug could not seat

Flagged by the team, and confirmed by measurement. Moving the screw bosses into the end blocks
made the material in front of the USB receptacle **6.3 mm thick**. The collar was then deleted
on the reasoning that this depth was free strain relief. It is not: it is material the plug has
to get *through*.

```
counterbore depth (overmould)    4.00 mm
tight bore depth (shell only)    2.30 mm   <- a 12 x 7 overmould cannot pass
tongue engagement into receptacle 4.20 mm  (needs ~6.5 for full mating)
VERDICT: PLUG WILL NOT FULLY SEAT (short by 2.30 mm)
```

A USB-C plug's metal shell inserts about **6.5 mm**, with its overmould immediately behind it.
So every millimetre in front of the receptacle that is too narrow for the overmould is a
millimetre stolen from engagement. The part exported perfectly and looked right in every
render.

**Fix.** The end block only needs full thickness *where the bosses are* — they sit at
y = ±11.9 while the port sits at y = −2. A recess is cut between them, sized generously for an
overmould (13 × 8 mm, since cable overmoulds vary from about 8 to 16 mm). The recess walls are
the strain relief: the overmould is captured on all four sides, so the cable cannot lever the
connector.

**`USB_FACE_WALL` must be zero**, and this took three attempts to get right. The tempting move
is to leave a thin face wall with a shell-sized slot so the port looks neat and dust is kept
out. It cannot work: a fully mated plug's overmould front face ends up *at* the receptacle
face, so the overmould sweeps the entire wall depth. Any face wall is subtracted directly from
insertion depth, and the tongue is only 6.5 mm to begin with. If a neat faceplate is ever
wanted, the way to get it is a panel-mount USB-C pigtail inside the enclosure, not a thinner
wall.

Two further traps found on the way, both invisible in a render:

- **The recess corner fillet.** At 1.0 mm the rounded vertical corners clipped a rectangular
  overmould by ~1 mm³ — enough to stop the plug seating. Reduced to 0.5 mm.
- **The fillet at the inner end.** The fillet rounds the vertical corners along the *whole*
  cut, so a recess stopping level with the plug's inner face puts rounded corners exactly
  where the plug's corners go. The cut now overshoots into the cavity, which removes nothing
  extra because that volume is already void.

### The check that caused this: never assert a parameter back at yourself

The original guard computed `engage = USB_PLUG_TONGUE - USB_FACE_WALL` and compared it to a
minimum. That is arithmetic on parameters — it restates the inputs and cannot observe the
solid. It cheerfully reported a healthy 5.30 mm while 111.6 mm³ of plastic physically blocked
the plug, and it would have reported the same had the recess boolean failed entirely.

It is replaced by a real interference test: `mockups.usb_plug()` places a plug solid where a
mated plug sits, and the check intersects it with the body. Non-zero volume means blocked, and
the message names the obstructing span in X. The same principle applies to every geometric
check in this project — **a check that cannot fail independently of the geometry is not a
check.** The plug-path check also earns its keep beyond fit: if it fails, the node cannot be
powered *or* reflashed, so the whole remote-firmware workflow in the runbook dies with it.

### The board mounting holes do not shrink the enclosure

Both the SSD1306 and the MH-ET LIVE MAX30102 carry four corner mounting holes (confirmed by
the team, 2026-09-11). The obvious hope was to move the screw bosses inside the OLED's hole
pattern and drop ~9 mm of length. **Measured, it does not work:**

```
5.0 mm boss (post with heat-set insert):
  (+11.65, -11.90)  C3 -1.60   MAX30102 +7.47
  (+11.65, +11.90)  C3 +2.40   MAX30102 -1.35
  -> worst clearance -1.60 mm

2.3 mm bare M2 through-bolt:
  (+11.65, -11.90)  C3 -0.25   MAX30102 +0.00
  -> worst clearance -0.25 mm
```

**The constraint was never the OLED.** The ESP32-C3 (22.5 × 18) and the MAX30102 (21 × 16)
occupy the middle of the cavity, so no fastener column can run floor-to-lid anywhere inside
it regardless of what holes the top board has. Even a bare M2 bolt is 0.25 mm short — and it
would still need something to thread into, at which point the 5 mm boss returns and collides
by 1.6 mm.

40.4 mm is therefore the floor for this board set. The only remaining lever is a physically
smaller display: a 0.91" OLED (≈30 × 12 mm) would collapse both the length and ~1.6 mm of
height.

### What the holes are actually worth

Positive location, which is a real gain in the place it matters most.

- **`MAX_HOLES`** — two diagonal locating pins through the sensor's own holes replace the
  printed fence. The fence only constrains the module to within the fit clearance on each
  side; the aperture has to register against a 5.6 × 3.3 mm optical window, so a millimetre of
  slop is the difference between reading the artery and reading tendon. This is the
  highest-stakes tolerance in the design.
- **`OLED_HOLES`** — two diagonal pegs descending from the lid. The display otherwise floats,
  with 0.3 mm of air above and a wire gap below, and nothing stopping it shifting out from
  behind its window.

Two pins rather than four in both cases: four would over-constrain a printed part and the
board would rock on whichever pair seated first.

Both are implemented and **gated on measurement** — `OLED_HOLES` and `MAX_HOLES` in
`params.py` are `None`, and the fence is used as the fallback. Supply each as
`(pitch_x, pitch_y, hole_diameter)`. Do not guess: an off-pitch pin is worse than no pin,
because the board then will not seat flat at all.

### There is deliberately no lid alignment lip

With 0.25 mm of board margin, any lip dropping into the cavity mouth lands on the OLED glass,
and a lip that cleared the glass would be a 0.3 mm wide rim — too thin to print. The four
screws locate the lid to well inside the tolerance a lip would have given. A 0.30 mm gap is
held above the glass so print tolerance cannot put the display in compression.

The stack itself comes to 18.2 mm — chunky-watch sized, a G-Shock being about 18 mm thick —
and its tallest single contributor is the C3's USB-C shell with the OLED sitting above it.
The build now ships taller than that on purpose; see *Raising the watch to 35 mm*.

### Z stack (from the skin-contact face, MH-ET LIVE variant, 35 mm build)

```
  skin  →  MAX30102 (face down)  →  ESP32-C3  →  [ plenum ]  →  SSD1306  →  lid

   1.20  inner floor / sensor optical face
   3.40  MAX30102 back
   5.20  ESP32-C3 PCB
   9.80  ESP32-C3 tallest component (USB-C shell)
         ( 18.00 mm wiring plenum )
  27.80  SSD1306 PCB
  31.90  SSD1306 glass / lid underside
  33.80  top of lid
```

Everything that has to line up with a hole in the shell — the optical aperture at z ≈ 0, the
USB-C port at z ≈ 8, the strap slots — sits **below** the plenum and did not move when the
watch grew.

## Raising the watch to 35 mm

Asked for on 2026-09-21: more room inside for hand-wiring. `TARGET_TOTAL_H = 35.0` in
`params.py` states the overall height directly — measured from the light-seal ring, the
lowest point against the skin, to the top of the lid, which is what a caliper across the
finished watch reads. `layout.py` subtracts the natural stack (18.2 mm, every gap already at
its minimum) and spends the entire 16.8 mm difference in **one** place: the gap between the
ESP32-C3 and the display.

(37 mm was tried first and judged too tall; the only edit needed to come back to 35 was that
one number, which is the point of keeping the height as a parameter rather than a dimension.)

One tall void beats two roomier gaps. Eight hand-soldered 30 AWG conductors previously had to
live in two 1.2 mm slots, with nowhere for slack to go and no way to get an iron near a joint
once the lower board was down. An 18 mm plenum can be reached from above with the lid off,
wire can be dressed and coiled in it, and the sensor's four conductors can run up alongside
the C3 (3.4 mm clear on each ±X side) instead of being pinched under it.

Setting `TARGET_TOTAL_H = None` returns the tight 18.2 mm stack; asking for less than the
boards allow raises a descriptive error rather than producing a quietly impossible part.

### The display needed something to stand on

Below ~18 mm total the display simply rested on the C3. With 18 mm of air under it, it would
have been hanging from four solder joints. `parts.py` therefore grows two **support rails**
up the ±X cavity walls, ending at the display's underside, so the board lands on a ledge that
underlaps each of its ±X edges by 1.25 mm — 69 mm² of bearing area.

Rails, not a full perimeter shelf: the C3 is pushed hard against the −Y wall so its USB-C
port can reach the outside, and a rail on that wall would sit exactly where the board has to
pass on the way down. In X both lower boards have millimetres to spare, so the cavity is just
narrower in X below the display (24.8 mm instead of 27.8 mm) and every board still drops
through. This is generated only when the plenum exceeds `PLENUM_SHELF_THRESHOLD`, so the
slim build is unchanged.

Two new checks prove it against the solid rather than against the parameters:

- each board's footprint is **swept vertically** up through the real body and must hit
  nothing — the check that would have caught a rail placed where a board has to pass;
- the display's underside is intersected with the body and must find **≥ 20 mm²** of bearing
  area — a rail that missed would still export a perfectly valid solid.

### What this costs, stated plainly

- **35 mm is about twice a G-Shock.** It is a wearable brick, appropriate for a bench and
  short clinical recordings, not for daily wear. The body is now nearly cubic: 40.4 × 34.6 ×
  33.4 mm including the USB collar.
- **The strap tunnel is now 33 mm deep**, because it is a vertical through-slot in the end
  blocks and the end blocks grew with the body. The band therefore emerges at the lid seat,
  32 mm above the skin, so strap tension pulls near the *top* of the device and levers it off
  the wrist — and rotation moves the aperture off the radial artery, the one thing this
  design cannot afford. `build.py` now raises this as a standing warning on every build.
  **Recommended before printing:** rework the end blocks into low side lugs (tunnel axis
  along Y, near the contact face) so the band stays at skin level regardless of body height.
  That is a change of strap topology, not of height, so it was not folded into this pass.
- More print time and material: the body is 15.4 cm³, up from about 9 cm³.

The OLED faces the palm side and is read by supinating the forearm — the same arrangement
as a clinical pulse-oximeter watch.

## Wiring

Both peripherals share **one I2C bus**. Confirmed against
`03 - ML/firmware/arduino/ArrhythmiaNode/ArrhythmiaNode.ino:243`.

| Signal | ESP32-C3 Super Mini | MAX30102 (8-pin) | SSD1306 OLED |
|---|---|---|---|
| 3V3 | `3V3` | `VIN` | `VCC` |
| Ground | `GND` | `GND` | `GND` |
| I2C data | `GPIO 8` | `SDA` | `SDA` |
| I2C clock | `GPIO 9` | `SCL` | `SCL` |

**Unused on the MAX30102:** `INT`, `RD`, `IRD`, and the second `GND`. The firmware polls the
sensor (`particleSensor.check()` / `.available()`, `ArrhythmiaNode.ino:306-307`) and never
uses the interrupt line. So despite the 8-pin header, the internal wire channel carries only
**4 conductors per device**.

**Addresses do not collide:** MAX30102 at `0x57`, SSD1306 at `0x3C`.

**Pull-ups:** both breakouts carry their own ~4.7 kΩ pull-ups. In parallel that is ~2.4 kΩ,
which is comfortably above the ~1.1 kΩ floor set by the 3 mA sink spec at 3.3 V. No change
needed, but do not add a third pull-up device to this bus without rechecking.

## Manufacturing

- **Assembly:** M2 screws into brass heat-set inserts. Chosen over snap-fits deliberately —
  a snap-fit cannot be tuned after printing and there is exactly one attempt.
- **Clearance:** `CLEARANCE = 0.25 mm` per side, the forgiving FDM value. Drop to 0.15 only
  after the Stage-1 coupon proves the shop holds it.
- **Material:** ask for **PETG/ABS (FDM)** or **PA12 nylon (MJF/SLS)**. If the shop quotes
  **SLA resin** — common and cheapest in PH — specify **full post-cure**: incompletely cured
  resin is a skin irritant and this part is worn against skin for extended recordings. Resin
  is also brittle, so strap lugs would need thickening.

### Order in two stages

**Stage 1 — fit gauge.** A 72 × 46 × 3 mm coupon carrying the USB-C opening, the optical
aperture, the heat-set insert bore and a screw clearance hole, each repeated at three
clearances (0.15 / 0.25 / 0.35), plus one strap slot. Whichever column gives a firm slip fit
sets `CLEARANCE` for the real build.

**This can be ordered now.** It does not depend on the unresolved sensor question, because
the optical aperture is a property of the MAX30102 *chip package* (5.6 × 3.3 mm), not of the
breakout it is mounted on.

**Stage 2 — the enclosure**, at the clearance the coupon proved.

## The sensor — MH-ET LIVE MAX30102

Identified 2026-09-11 from the MakerLab Electronics PH product photo. `MAX_VARIANT` is set to
`"mhetlive"` and the enclosure now builds.

- **PCB 21 × 16 mm**
- Two rows of four pads: `GND` `RD` `IRD` `INT` on one edge, `VIN` `SDA` `SCL` `GND` on the
  other
- Four corner mounting holes plus two elongated plated slots on the left and right edges
- `1V8` and `3V3` solder pads on the right edge — **a pull-up selector, see below**
- The optical window is on the **opposite face** to the silkscreen

### The 1V8 / 3V3 pads are a trap worth knowing about

On MH-ET LIVE boards the onboard ~4.7 kΩ I²C pull-ups can be strapped either to the module's
internal **1.8 V** rail or to **3.3 V**. Strapped to 1.8 V, the sensor **never appears in an
I²C scan** from a 3.3 V microcontroller like the ESP32-C3 — the bus simply never reaches a
valid logic high.

The current board is evidently strapped correctly, since the acquisition chain is live on real
sensor data. Record it anyway: if a spare or replacement module is ever fitted and the firmware
starts reporting a missing sensor, **check this jumper before touching any code.** It is a
five-minute hardware fix that can otherwise burn a day in the driver.

### Still outstanding — the aperture position

The board outline came from a catalogue. The **aperture position did not**, and it is the
design datum: the boss is placed from it and the light seal is built around it. It is currently
assumed centred on the PCB, flagged `AP_PROVISIONAL=True`, and `build.py` emits a loud warning
on every build.

**The current export is for review only and must not go to the print shop.** Getting this wrong
puts the LEDs over the flexor tendons instead of the radial artery, which presents as a flat PPG
trace and gets misdiagnosed as a firmware fault.

Needed — a photo of the **front** (sensor side) flat next to a ruler answers all of it:

1. Aperture centre — X/Y from one named corner
2. Aperture window size
3. Tallest component on the back face
4. Whether the 8-pin headers can be desoldered (they add ~8 mm if not)

## Validation

`build.py` runs pre-flight checks before exporting, because a CAD boolean that silently did
nothing still produces a clean-looking file. Two of them caught real defects on the first
build of this model:

- the lid's alignment lip landing squarely on all four screw posts;
- `body_h` including the lid's own thickness, so body and lid occupied the same 1.6 mm and
  the assembly stood a lid-thickness too tall.

Adding mock solids for the three real boards (`mockups.py`) then caught three more:

- **all four screw posts passed through all three PCBs** — the structural finding above;
- **the lid's "alignment lip" was a solid block, not a rim**, so it filled the cavity mouth
  and bore directly on the OLED glass;
- **the contact-face chamfer had never applied at all.** `faces("<Z")` selects the seal-ring
  underside, not the contact face, and a 2 mm chamfer on a 1.2 mm-wide ring fails — which an
  `except: pass` swallowed. The part exported clean and would have printed with a sharp 90°
  rim bearing on the wrist. The exception handler is gone and a cross-section check now
  guards it. Note the chamfer must stay strictly below `FILLET_OUTER` or OCCT raises
  `StdFail_NotDone`; it is derived as a fraction of the fillet so the two cannot diverge.

All fixed. Current status: every check passes, with one standing warning — the provisional
aperture position.

### That antenna warning is currently harmless

The firmware **never brings the radio up** — there is no `WiFi.h`, no BLE and no networking
anywhere in `03 - ML/firmware/`. Telemetry leaves over USB CDC via `Serial.write()`
(`ArrhythmiaNode.ino:359`). An unused antenna cannot be detuned.

The check is kept rather than deleted because the thesis is framed as an **IoT** system. The
moment anyone makes this node wireless, the warning becomes a real defect — and a detuned
antenna presents as flaky range, which is miserable to diagnose after the part is printed.

## Related — not an enclosure problem

`ArrhythmiaNode.ino:278-279` drives both LEDs at `0x1F` (~6.4 mA), which is tuned for
**fingertip transmission**. Wrist *reflectance* through thicker tissue typically needs
substantially more drive current. At 6.4 mA the wrist trace may look near-flat no matter how
good the enclosure is.

Recording it here so that, once the housing is printed, a weak signal is not misdiagnosed as
a mechanical fit problem. This is a firmware change and is **not** in scope for the enclosure.

---

**Source:** `08 - Hardware/enclosure/` — `params.py` is the single source of truth.
**Related:** [[2026-09-10 - Raspberry Pi Deployment Architecture and Runbook]] ·
[[2026-09-10 - Model Training Plan and Project Roadmap]]
