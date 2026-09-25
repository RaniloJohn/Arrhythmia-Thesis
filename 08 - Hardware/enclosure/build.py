"""
Render every part to STEP and STL, with pre-flight geometric checks.

    python build.py            # build everything it can
    python build.py --gauge    # Stage-1 fit gauge only (no sensor data needed)
    python build.py --check    # run checks, export nothing

STEP is the format the print shop and Fusion 360 want; STL is what the slicer
wants. Both are written to out/, which is gitignored - the source of truth is
params.py, not the exported mesh.

The checks exist because a boolean that silently did nothing still exports a
clean-looking file. Every one of them has caught a real class of mistake:
an aperture that never reached the outside, a USB-C cutout that missed the
wall, or a brass insert planted in the C3's antenna keep-out.
"""

import argparse
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cadquery as cq

import params as P
import layout as L
import parts
import mockups

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


# ─────────────────────────────────────────────────────────────────────
#  Checks
# ─────────────────────────────────────────────────────────────────────

class CheckResult:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.passed = []

    def fail(self, msg):
        self.errors.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)

    def ok(self, msg):
        self.passed.append(msg)

    def report(self):
        for m in self.passed:
            print(f"  [ok]   {m}")
        for m in self.warnings:
            print(f"  [warn] {m}")
        for m in self.errors:
            print(f"  [FAIL] {m}")
        return not self.errors


def _solid_check(r, name, shape):
    val = shape.val()
    if not val.isValid():
        r.fail(f"{name}: shape is not a valid solid")
        return
    vol = val.Volume()
    if vol <= 0:
        r.fail(f"{name}: non-positive volume ({vol:.2f})")
        return
    r.ok(f"{name}: valid solid, {vol / 1000.0:.2f} cm3")


def _overlap_1d(a0, a1, b0, b1):
    """Signed overlap of two intervals. Positive means they intersect."""
    return min(a1, b1) - max(a0, b0)


def check_all(lo, built) -> CheckResult:
    r = CheckResult()

    for name, shape in built.items():
        _solid_check(r, name, shape)

    if "body" not in built:
        return r

    body = built["body"]

    # --- The aperture must actually pierce the contact face ----------
    # If the boolean missed, the sensor sees only plastic and the trace is
    # flat - a failure mode indistinguishable from a dead sensor in software.
    probe = (
        cq.Workplane("XY")
        .box(lo.aperture.l * 0.5, lo.aperture.w * 0.5,
             P.BOSS_PROUD + P.SEAL_RING_PROUD + P.FLOOR + 2.0)
        .translate((lo.aperture.cx, lo.aperture.cy,
                    -P.BOSS_PROUD - P.SEAL_RING_PROUD - 1.0))
    )
    if body.intersect(probe).val().Volume() > 0.01:
        r.fail("optical aperture is blocked - it does not reach the contact face")
    else:
        r.ok("optical aperture is clear through boss and floor")

    # --- The USB-C cutout must actually breach the -Y wall -----------
    wall_probe = (
        cq.Workplane("XY")
        .box(P.C3_USB_W * 0.5, P.WALL * 0.5, P.C3_USB_H * 0.5)
        .translate((lo.c3.cx, -lo.cav_w / 2 - P.WALL / 2,
                    lo.z_c3_pcb + P.C3_PCB_T + P.C3_USB_H / 2))
    )
    if body.intersect(wall_probe).val().Volume() > 0.01:
        r.fail("USB-C cutout did not breach the -Y wall - the plug cannot seat")
    else:
        r.ok("USB-C cutout breaches the -Y wall")

    # --- Brass inserts vs the C3 antenna keep-out --------------------
    # Plastic near a ceramic chip antenna is fine; a brass heat-set insert
    # detunes the 2.4 GHz element and costs range.
    #
    # This is a WARNING, not a failure, because the firmware never brings the
    # radio up: there is no WiFi.h, no BLE and no networking anywhere in
    # 03 - ML/firmware/, and telemetry leaves over USB CDC via Serial.write()
    # at ArrhythmiaNode.ino:359. An unused antenna cannot be detuned.
    #
    # The check is kept rather than deleted because the thesis is framed as an
    # IoT system. The moment anyone makes the node wireless, this silently
    # becomes a real defect - and a detuned antenna presents as flaky range,
    # which is miserable to diagnose after the part is already printed.
    ko = L.antenna_keepout(lo)
    worst = None
    for (x, y) in lo.screw_xy:
        rad = P.INSERT_D / 2
        ox = _overlap_1d(x - rad, x + rad, ko.x_min, ko.x_max)
        oy = _overlap_1d(y - rad, y + rad, ko.y_min, ko.y_max)
        if ox > 0 and oy > 0:
            r.warn(f"brass insert at ({x:+.1f}, {y:+.1f}) sits inside the C3 "
                   f"antenna keep-out by {min(ox, oy):.2f} mm - harmless while "
                   f"the node is USB-tethered, must be moved before adding WiFi")
        else:
            clear = -min(ox, oy)
            worst = clear if worst is None else min(worst, clear)
    if worst is not None and not any("antenna" in w for w in r.warnings):
        r.ok(f"all brass inserts clear the antenna keep-out "
             f"(tightest {worst:.2f} mm)")

    # --- Lid must not foul the screw bosses --------------------------
    if "lid" in built:
        if built["lid"].intersect(body).val().Volume() > 0.01:
            r.fail("lid and body interfere - check LIP_H against cavity depth")
        else:
            r.ok("lid seats on the body without interference")

    # --- The contact-face relief must actually be there --------------
    # A chamfer that fails to apply leaves a valid, clean-looking solid with
    # a sharp 90-degree rim bearing on the wrist. Nothing else catches it:
    # the part exports fine and a shaded render cannot show it. So measure
    # the cross-section instead - a real chamfer makes the section grow with
    # height over the first CONTACT_CHAMFER millimetres.
    def _area_at(z):
        plane = cq.Workplane("XY").box(400, 400, 0.02).translate((0, 0, z))
        inter = body.val().intersect(plane.val())
        return inter.Volume() / 0.02 if inter.Volume() > 1e-9 else 0.0

    # Both samples must sit BELOW z_floor. Above it the cavity opens up and
    # removes ~800 mm2, which swamps the ~125 mm2 the chamfer contributes and
    # makes a perfectly good chamfer read as missing.
    z_hi = min(parts.CONTACT_CHAMFER, lo.z_floor) - 0.1
    grow = _area_at(z_hi) - _area_at(0.1)
    if grow < 5.0:
        r.fail("contact-face perimeter chamfer is missing - the rim is a "
               "sharp 90-degree edge bearing on the wrist")
    else:
        r.ok(f"contact-face relief present (section grows {grow:.0f} mm2 "
             f"between z=0.1 and z={z_hi:.1f})")

    # --- Can a USB-C plug actually reach the receptacle? -------------
    # "The cutout breaches the wall" is not the same as "the plug mates".
    # A plug's shell inserts ~6.5 mm with its overmould right behind it, so
    # every millimetre of enclosure in front of the receptacle that is too
    # narrow for the overmould is a millimetre stolen from engagement. This
    # caught a real defect: after the bosses moved into the end blocks the
    # material in front of the port went to 6.3 mm and the plug fell 2.3 mm
    # short of seating, on a part that otherwise exported perfectly.
    # Put a plug where a mated plug goes and intersect it with the body.
    # The previous version of this check computed engagement arithmetically
    # from USB_FACE_WALL and so simply restated the parameters - it reported
    # a healthy 5.30 mm while a 1.2 mm face wall physically blocked the plug.
    # A check that cannot fail independently of the geometry is not a check.
    plug = mockups.usb_plug(lo)
    blocked = body.intersect(plug).val().Volume()
    if blocked > 0.01:
        bb = body.intersect(plug).val().BoundingBox()
        r.fail(f"USB-C plug is obstructed by {blocked:.1f} mm3 of enclosure "
               f"between x={bb.xmin:.2f} and x={bb.xmax:.2f} - the plug cannot "
               f"seat, so the node cannot be powered or flashed")
    else:
        r.ok("USB-C plug seats fully - no enclosure material in its path")

    # --- The STRAP must not occupy the plug's space ------------------
    # The band is a real object, not just a hole. When the port and the strap
    # slots shared the +X end block this overlapped by 357 mm3 - the band
    # threaded straight through where the plug had to go. Every geometric
    # check passed at the time, because none of them modelled the band.
    #
    # Nothing here is a clearance problem; it is a topology one. Straps go on
    # the ends, the port goes on a side, exactly as a smartwatch arranges it.
    slot_t = P.STRAP_T + 2 * P.CLEARANCE
    slot_w = P.STRAP_W + 2 * P.CLEARANCE
    for sx in (+1, -1):
        band = (
            cq.Workplane("XY")
            .box(slot_t, slot_w, lo.body_h + 6.0, centered=(True, True, False))
            .translate((sx * lo.lug_x, 0, -3.0))
        )
        hit = band.intersect(plug).val()
        v = hit.Volume() if hit.Volume() > 1e-9 else 0.0
        if v > 0.01:
            r.fail(f"the strap band passes through the USB plug "
                   f"({v:.0f} mm3) at the {'+X' if sx > 0 else '-X'} end - "
                   f"the port and the strap cannot share an end block")
    if not any("strap band" in e for e in r.errors):
        r.ok("strap band and USB plug occupy separate space")

    # The plug recess must not eat into a screw boss.
    recess_half = (P.USB_PLUG_W + 2 * P.CLEARANCE) / 2
    gap_boss = min(abs(sx * lo.lug_x - lo.c3.cx) for sx in (+1, -1)) \
        - P.BOSS_D / 2 - recess_half
    if gap_boss < 0.5:
        r.fail(f"USB plug recess comes within {gap_boss:.2f} mm of a screw "
               f"boss - reduce USB_PLUG_W or move the bosses outward")
    else:
        r.ok(f"USB plug recess clears the screw bosses by {gap_boss:.2f} mm")

    # --- Strap slots now share the end blocks with the bosses --------
    # Three ways that merge can go wrong, none of them visible in a render.
    slot_t = P.STRAP_T + 2 * P.CLEARANCE
    slot_half_w = (P.STRAP_W + 2 * P.CLEARANCE) / 2
    slot_x_in = lo.lug_x - slot_t / 2
    slot_x_out = lo.lug_x + slot_t / 2

    # 1. The slot must not break into the cavity and expose the electronics.
    gap_cav = slot_x_in - lo.cav_l / 2
    if gap_cav < 0.6:
        r.fail(f"strap slot is {gap_cav:.2f} mm from the cavity wall - it "
               f"breaks through into the electronics bay")
    else:
        r.ok(f"strap slot clears the cavity by {gap_cav:.2f} mm")

    # 2. The slot must not run into a heat-set insert bore.
    gap_ins = min(abs(y) for (_, y) in lo.screw_xy) - P.INSERT_D / 2 - slot_half_w
    if gap_ins < 0.6:
        r.fail(f"strap slot is {gap_ins:.2f} mm from an insert bore - "
               f"reduce STRAP_W or move the bosses outward in Y")
    else:
        r.ok(f"strap slot clears the insert bores by {gap_ins:.2f} mm")

    # 3. The wall outboard of the slot carries the entire strap tension.
    strap_wall = lo.body_l / 2 - slot_x_out
    if strap_wall < 1.2:
        r.fail(f"only {strap_wall:.2f} mm of material outboard of the strap "
               f"slot - it will tear out under strap tension")
    elif strap_wall < 2.0:
        r.warn(f"strap-bearing wall is {strap_wall:.2f} mm - acceptable in "
               f"PETG or PA12, thin for brittle SLA resin")
    else:
        r.ok(f"strap-bearing wall is {strap_wall:.2f} mm")

    # --- Aperture position provenance --------------------------------
    # The board outline can come from a catalogue; the aperture position
    # cannot. Getting it wrong puts the LEDs over the flexor tendons instead
    # of the radial artery, which presents as a flat trace and gets blamed on
    # the firmware. A provisional build is fine to look at and wrong to print.
    if P.max30102().get("AP_PROVISIONAL"):
        m = P.max30102()
        tight_l = m["AP_L"] + 2 * P.CLEARANCE
        tight_w = m["AP_W"] + 2 * P.CLEARANCE
        r.warn(f"aperture position is PROVISIONAL - the window is opened to "
               f"{lo.aperture.l:.2f} x {lo.aperture.w:.2f} mm (from "
               f"{tight_l:.2f} x {tight_w:.2f}) so the chip is exposed "
               f"anywhere within +/-{P.AP_UNCERTAINTY:.1f} mm. Printable as "
               f"is; measure AP_DX/AP_DY and set AP_UNCERTAINTY = 0 to "
               f"recover the tight window")

    # --- Do the real boards actually fit? ----------------------------
    comps = mockups.all_components(lo)

    # 1. No board may collide with the shell. The USB-C shell is the single
    #    intentional exception: it has to pass through the wall opening.
    for name, c in comps.items():
        clash = body.intersect(c).val().Volume()
        if clash > 0.01:
            r.fail(f"{name} collides with the body ({clash:.1f} mm3 overlap)")
        else:
            r.ok(f"{name} clears the body")

        if "lid" in built:
            lclash = built["lid"].intersect(c).val().Volume()
            if lclash > 0.01:
                r.fail(f"{name} collides with the lid ({lclash:.1f} mm3)")

    # 2. No board may collide with another board.
    names = list(comps)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            v = comps[a].intersect(comps[b]).val().Volume()
            if v > 0.01:
                r.fail(f"{a} and {b} overlap by {v:.1f} mm3 - "
                       f"increase WIRE_GAP in layout.py")
    if not any("overlap" in e for e in r.errors):
        r.ok("no board-to-board collisions")

    # 3. Every board must sit inside the cavity footprint. A board that pokes
    #    through a wall still passes the collision test if the wall happens to
    #    be cut away there, so check containment explicitly.
    for name, c in comps.items():
        bb = c.val().BoundingBox()
        over_x = max(bb.xmax - lo.cav_l / 2, -lo.cav_l / 2 - bb.xmin)
        over_y = max(bb.ymax - lo.cav_w / 2, -lo.cav_w / 2 - bb.ymin)
        # The C3 is allowed past the +X wall: that is the USB-C shell.
        allow = P.C3_USB_PROTRUDE + 3.1 if name == "esp32c3" else 0.05
        if over_x > allow or over_y > 0.05:
            r.fail(f"{name} extends outside the cavity "
                   f"(X by {over_x:+.2f}, Y by {over_y:+.2f} mm)")
        else:
            head = min(lo.cav_l / 2 - bb.xmax, lo.cav_w / 2 - bb.ymax,
                       bb.ymin + lo.cav_w / 2)
            r.ok(f"{name} fits the cavity (tightest margin {head:.2f} mm)")

    # 4. Every board must be droppable into the cavity from above.
    #    "It fits" and "it can be assembled" are different claims: a board
    #    sitting flush against a wall has nowhere to go on the way in, and a
    #    single tolerance the wrong way makes it uninstallable.
    for name, c in comps.items():
        bb = c.val().BoundingBox()
        sides = {
            "-X": bb.xmin + lo.cav_l / 2, "+X": lo.cav_l / 2 - bb.xmax,
            "-Y": bb.ymin + lo.cav_w / 2, "+Y": lo.cav_w / 2 - bb.ymax,
        }
        tight = min(sides, key=sides.get)
        val = sides[tight]
        if val < 0.15:
            r.fail(f"{name} has only {val:.2f} mm at {tight} - it cannot be "
                   f"lowered into the cavity during assembly")
        else:
            r.ok(f"{name} is insertable (tightest {val:.2f} mm at {tight})")

    # 5. Every board must have a clear VERTICAL PATH down to its seat.
    #    Fitting the cavity footprint is not the same as being installable
    #    once the cavity stops being a plain box: the display support rails
    #    narrow it in X for the whole height below the display, and a board
    #    wider than the gap between them can no longer be lowered past. Sweep
    #    each board's own footprint straight up through the real solid and
    #    see whether it hits anything - the rails are geometry, so only
    #    geometry can answer this. The footprint is swept bare, without the
    #    fit clearance: the display already fills the cavity to within that
    #    clearance, so adding it would clip the cavity's corner fillets and
    #    report a board that drops in perfectly well as uninstallable.
    for name, rect, z0 in (("max30102", lo.sensor, lo.z_floor),
                           ("esp32c3", lo.c3, lo.z_c3),
                           ("ssd1306", lo.oled, lo.z_oled)):
        col = (
            cq.Workplane("XY")
            .box(rect.l, rect.w, lo.z_lid + 5.0 - z0,
                 centered=(True, True, False))
            .translate((rect.cx, rect.cy, z0))
        )
        hit = body.intersect(col).val()
        v = hit.Volume() if hit.Volume() > 1e-9 else 0.0
        if v > 0.01:
            bb = hit.BoundingBox()
            r.fail(f"{name} cannot be lowered into place - {v:.1f} mm3 of "
                   f"enclosure sits above its seat between z={bb.zmin:.1f} "
                   f"and z={bb.zmax:.1f}")
        else:
            r.ok(f"{name} has a clear vertical insertion path")

    # 6. A tall plenum leaves the display with nothing under it. Prove the
    #    rails actually reach beneath its edges - a rail that missed would
    #    still export a perfectly valid solid, and the board would be held up
    #    by four solder joints.
    if lo.plenum > P.PLENUM_SHELF_THRESHOLD:
        shelf = (
            cq.Workplane("XY")
            .box(lo.oled.l, lo.oled.w, 0.4, centered=(True, True, False))
            .translate((lo.oled.cx, lo.oled.cy, lo.z_oled - 0.4))
        )
        bearing = body.intersect(shelf).val()
        area = (bearing.Volume() / 0.4) if bearing.Volume() > 1e-9 else 0.0
        if area < 20.0:
            r.fail(f"the display has only {area:.1f} mm2 of support under it "
                   f"across a {lo.plenum:.1f} mm plenum - it is hanging on "
                   f"its wires")
        else:
            r.ok(f"display rests on {area:.0f} mm2 of support rail")

    # 7. Headroom under the lid.
    top = max(c.val().BoundingBox().zmax for c in comps.values())
    gap = lo.z_lid - top
    if gap < -0.01:
        r.fail(f"the stack is {-gap:.2f} mm taller than the cavity")
    elif gap < 0.2:
        r.warn(f"only {gap:.2f} mm between the tallest board and the lid")
    else:
        r.ok(f"{gap:.2f} mm headroom between the top board and the lid")

    # --- Wearability sanity ------------------------------------------
    if lo.total_h > 22.0:
        why = (f" - this is TARGET_TOTAL_H = {P.TARGET_TOTAL_H:.1f} mm, asked "
               f"for deliberately to get {lo.plenum:.1f} mm of hand-wiring "
               f"room") if P.TARGET_TOTAL_H is not None else ""
        r.warn(f"overall height {lo.total_h:.1f} mm is thick for a wrist "
               f"device (a G-Shock is ~18 mm){why}")

    # The strap threads vertically through the end blocks, so the tunnel is
    # as deep as the body is tall. That is fine on a slim body and awkward on
    # a tall one: the band emerges at the lid seat, so strap tension pulls
    # near the TOP of the device and levers it off the wrist - which moves
    # the aperture off the artery, the one thing this design cannot afford.
    if lo.body_h > 20.0:
        r.warn(f"strap tunnel is {lo.body_h + P.BOSS_PROUD:.1f} mm deep and "
               f"the band exits at the lid seat, {lo.body_h:.1f} mm above the "
               f"skin - it will lever the body away from the wrist. Consider "
               f"reworking the end blocks into low side lugs before printing")

    return r


# ─────────────────────────────────────────────────────────────────────
#  Export
# ─────────────────────────────────────────────────────────────────────

def export(name, shape):
    os.makedirs(OUT, exist_ok=True)
    step = os.path.join(OUT, f"{name}.step")
    stl = os.path.join(OUT, f"{name}.stl")
    cq.exporters.export(shape, step)
    cq.exporters.export(shape, stl, tolerance=0.01, angularTolerance=0.1)
    print(f"  wrote {os.path.relpath(step, OUT)} and "
          f"{os.path.relpath(stl, OUT)}")


# ─────────────────────────────────────────────────────────────────────
#  Print-ready folder
# ─────────────────────────────────────────────────────────────────────

PRINT_READY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "print-ready")

# Split by stage, because the order matters more than the file list: the
# gauges exist to settle the tolerances the enclosure is then cut to.
STAGES = {
    "stage-1-gauges": ["fit_gauge", "board_gauge"],
    "stage-2-enclosure": ["body", "lid",
                          "shim_0p4mm", "shim_0p8mm", "shim_1p2mm"],
}


def publish_print_ready():
    """
    Copy only the printable parts into print-ready/, grouped by stage.

    out/ also holds the ref_* reference solids, which must never reach a
    printer. Keeping a separate folder that contains nothing but real parts
    removes the chance of sending the wrong file.
    """
    import shutil

    lines = []
    for stage, names in STAGES.items():
        d = os.path.join(PRINT_READY, stage)
        os.makedirs(d, exist_ok=True)
        for n in names:
            for ext in ("step", "stl"):
                src = os.path.join(OUT, f"{n}.{ext}")
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(d, f"{n}.{ext}"))
        lines.append(f"{stage}/  {len(names)} part(s)")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gauge", action="store_true",
                    help="build only the Stage-1 fit gauge")
    ap.add_argument("--check", action="store_true",
                    help="run checks without exporting")
    ap.add_argument("--assembly", action="store_true",
                    help="also export the board mockups, so the whole "
                         "assembly can be inspected in Fusion 360")
    args = ap.parse_args()

    built = {}

    print("Fit gauge")
    built["fit_gauge"] = parts.fit_gauge()
    print("  built (independent of the MAX30102 variant)")

    if not args.gauge:
        try:
            lo = L.compute()
        except ValueError as e:
            print("\n" + "=" * 68)
            print("Enclosure NOT built - missing input:")
            print("=" * 68)
            print(e)
            print("\nThe fit gauge above does not depend on this and can be "
                  "ordered now.\n")
            if not args.check:
                export("fit_gauge", built["fit_gauge"])
            return 1

        print("\n" + L.describe(lo) + "\n")
        print("Parts")
        built["board_gauge"] = parts.board_gauge(lo)
        built["body"] = parts.body(lo)
        built["lid"] = parts.lid(lo)
        for n, s in parts.shims(lo).items():
            built[n] = s

        if args.assembly:
            # Reference solids, not printable parts. Exported so the fit can
            # be inspected, sectioned and measured in Fusion 360 rather than
            # taken on trust from a report.
            for n, s in mockups.all_components(lo).items():
                built["ref_" + n] = s
            built["ref_usb_plug"] = mockups.usb_plug(lo)
            slot_t = P.STRAP_T + 2 * P.CLEARANCE
            slot_w = P.STRAP_W + 2 * P.CLEARANCE
            band = None
            for sx in (+1, -1):
                b = (cq.Workplane("XY")
                     .box(slot_t, slot_w, lo.body_h + 6.0,
                          centered=(True, True, False))
                     .translate((sx * lo.lug_x, 0, -3.0)))
                band = b if band is None else band.union(b)
            built["ref_strap_band"] = band

        print(f"  built {len(built)} parts")

        print("\nChecks")
        lo_ok = check_all(lo, built).report()
    else:
        lo_ok = True
        print("\nChecks")
        r = CheckResult()
        _solid_check(r, "fit_gauge", built["fit_gauge"])
        lo_ok = r.report()

    if args.check:
        return 0 if lo_ok else 1

    if not lo_ok:
        print("\nChecks failed - refusing to export. Fix params.py and re-run.")
        return 1

    print("\nExport")
    for n, s in built.items():
        export(n, s)

    stage_files = publish_print_ready()
    if stage_files:
        print("\nPrint-ready folder")
        for line in stage_files:
            print("  " + line)

    print("\nStill to confirm with calipers before release:")
    for part, what in P.verify_report():
        print(f"  - {part:10s} {what}")
    return 0


def _exit(code):
    """
    Leave without running interpreter finalization.

    CadQuery's OCCT kernel (OCP) corrupts the heap in its static destructors
    on Windows, so a normal sys.exit() returns 0xC0000374 even when every
    part exported cleanly. That nonzero status is indistinguishable from a
    real build failure to anything scripting this. Flushing and calling
    os._exit skips the faulty teardown; the files are already on disk by
    this point.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    try:
        _exit(main())
    except Exception:
        traceback.print_exc()
        _exit(2)
