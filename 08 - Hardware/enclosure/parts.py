"""
Solid geometry for the wrist enclosure.

Four printable parts:

    body()       lower shell - sensor boss, cavity, USB-C exit, strap lugs
    lid()        upper shell - OLED window, alignment lip, screw holes
    shims()      stackable boss shims for tuning skin contact pressure
    fit_gauge()  Stage-1 coupon that proves the cutouts before the real print

All of them are driven from layout.compute(), which is driven from params.py.
No dimension is written literally in this file.

NOTE ON THE CONTACT FACE
------------------------
The design brief called for an R30 transverse curve to match the volar wrist.
Implemented literally that removes ~6 mm of material at the body edge, which
breaches a 1.2 mm floor and opens the cavity. The ergonomic goal - that the
rim cannot dig into the wrist - is met instead with a flat contact face and a
large perimeter chamfer (CONTACT_CHAMFER). Flat also guarantees the sensor
boss seats squarely, which matters more for signal quality than conformity.
"""

import cadquery as cq

import params as P
import layout as L

# Perimeter relief on the skin-contact face. Replaces the R30 curve; see the
# module docstring.
#
# Must stay strictly below FILLET_OUTER: the chamfer runs around the perimeter
# and has to resolve against the filleted vertical corners, and OCCT throws
# StdFail_NotDone when the two are equal. Measured on this geometry: 1.8 mm is
# the last value that builds against a 2.0 mm fillet. Capped at 75% of the
# fillet so changing FILLET_OUTER cannot silently break the build.
_CONTACT_CHAMFER_WANTED = 1.5
CONTACT_CHAMFER = min(_CONTACT_CHAMFER_WANTED, P.FILLET_OUTER * 0.75)

# Lid alignment lip - drops into the cavity so the halves cannot shift.
LIP_H = 0.0  # no alignment lip; see lid() for why

# Strap lug height, measured up from the contact face. Kept low so the band
# emerges close to the skin and the body does not lever away from the wrist.
LUG_H = 5.0


def _lug_l() -> float:
    """Lug block length along X - must clear the band plus its side walls."""
    return max(P.LUG_L, P.STRAP_T + 2 * P.LUG_WALL)


def _rounded_block(l, w, h, r, centre_z=False):
    """A block with filleted vertical edges, sitting on z = 0 by default."""
    wp = cq.Workplane("XY")
    if centre_z:
        wp = wp.box(l, w, h)
    else:
        wp = wp.box(l, w, h, centered=(True, True, False))
    if r > 0:
        wp = wp.edges("|Z").fillet(r)
    return wp


# ─────────────────────────────────────────────────────────────────────
#  Body
# ─────────────────────────────────────────────────────────────────────

def body(lo: L.Layout = None):
    m = P.max30102()
    lo = lo or L.compute()

    part = _rounded_block(lo.body_l, lo.body_w, lo.body_h, P.FILLET_OUTER)

    # --- Contact-face perimeter relief -------------------------------
    # Applied to the bare block FIRST, while "<Z" unambiguously means the
    # contact face. Do this after the boss is unioned on and "<Z" resolves to
    # the seal-ring underside instead - a ~1.2 mm wide face that a 2 mm
    # chamfer cannot be applied to, so the operation fails and the wrist gets
    # a sharp 90-degree rim.
    part = part.faces("<Z").chamfer(CONTACT_CHAMFER)

    # --- Cavity ------------------------------------------------------
    part = part.cut(
        _rounded_block(
            lo.cav_l, lo.cav_w, lo.cav_h + 1.0,
            max(P.FILLET_OUTER - P.WALL, 0.5),
        ).translate((0, 0, lo.z_floor))
    )

    # --- Display support rails ---------------------------------------
    # With a tall wiring plenum the display has nothing underneath it: the
    # C3 is now 20 mm below, and the lid alone cannot hold a board that
    # cannot reach it. Two rails up the +/-X cavity walls give it a ledge to
    # land on, so the stack is supported top and bottom instead of dangling
    # from the wires.
    #
    # They are rails rather than a full perimeter shelf on purpose. The C3 is
    # pushed hard against the -Y wall so its USB-C port can reach the outside,
    # and a rail there would sit exactly where the board has to pass on the
    # way down. In X both lower boards have millimetres to spare, so the
    # cavity can simply be narrower in X below the display without making any
    # board uninstallable - build.py proves that by sweeping each board
    # upward through the real solid rather than by trusting this paragraph.
    if lo.plenum > P.PLENUM_SHELF_THRESHOLD:
        rail_x_in = lo.oled.l / 2 - P.OLED_SHELF_W
        rail_l = lo.cav_l / 2 - rail_x_in
        for sx in (+1, -1):
            part = part.union(
                cq.Workplane("XY")
                .box(rail_l, lo.cav_w, lo.z_oled - lo.z_floor,
                     centered=(True, True, False))
                .translate((sx * (rail_x_in + rail_l / 2), 0, lo.z_floor))
            )

    # --- Sensor boss, proud of the contact face ----------------------
    # Footprint is the aperture plus a seal ring plus wall on each side.
    boss_l = lo.aperture.l + 2 * P.SEAL_RING_W + 2 * P.WALL
    boss_w = lo.aperture.w + 2 * P.SEAL_RING_W + 2 * P.WALL
    boss = (
        _rounded_block(boss_l, boss_w, P.BOSS_PROUD, 1.0)
        .translate((lo.aperture.cx, lo.aperture.cy, -P.BOSS_PROUD))
    )
    part = part.union(boss)

    # --- Light seal ring ---------------------------------------------
    # Stops the red/IR LEDs coupling through the housing straight into the
    # photodiode. Without it the trace is DC-dominated with no usable AC.
    ring_outer = (
        _rounded_block(
            lo.aperture.l + 2 * P.SEAL_RING_W,
            lo.aperture.w + 2 * P.SEAL_RING_W,
            P.SEAL_RING_PROUD, 0.4,
        )
        .translate((lo.aperture.cx, lo.aperture.cy,
                    -P.BOSS_PROUD - P.SEAL_RING_PROUD))
    )
    part = part.union(ring_outer)

    # --- Optical aperture, straight through the boss and floor -------
    part = part.cut(
        cq.Workplane("XY")
        .box(lo.aperture.l, lo.aperture.w,
             P.BOSS_PROUD + P.SEAL_RING_PROUD + P.FLOOR + 1.0,
             centered=(True, True, False))
        .translate((lo.aperture.cx, lo.aperture.cy,
                    -P.BOSS_PROUD - P.SEAL_RING_PROUD - 0.5))
    )

    # --- Sensor location ----------------------------------------------
    # Two options, and the pins are much the better one when the hole pitch
    # is known: a fence only constrains the module to within the fit
    # clearance on each side, while pins through real holes fix it. The
    # aperture has to line up with a 5.6 x 3.3 mm optical window, so a
    # millimetre of slop here is the difference between reading the artery
    # and reading tendon.
    if P.MAX_HOLES:
        px, py, hd = P.MAX_HOLES
        pin_d = hd - 2 * P.PIN_CLEARANCE
        pin_h = m["T"] + 1.0
        # Two diagonal pins, not four: four would over-constrain a printed
        # part and the module would rock on whichever pair seated first.
        for (sx, sy) in ((+1, +1), (-1, -1)):
            part = part.union(
                cq.Workplane("XY")
                .circle(pin_d / 2)
                .extrude(pin_h)
                .translate((lo.sensor.cx + sx * px / 2,
                            lo.sensor.cy + sy * py / 2, lo.z_floor))
            )
    else:
        # Fallback fence. Prints without support and needs no measurement,
        # but only locates the module to +/- CLEARANCE.
        fence_h = m["T"] + 0.6
        outer = _rounded_block(
            lo.sensor.l + 2 * P.CLEARANCE + 2 * 1.0,
            lo.sensor.w + 2 * P.CLEARANCE + 2 * 1.0,
            fence_h, 0.5,
        ).translate((lo.sensor.cx, lo.sensor.cy, lo.z_floor))
        inner = cq.Workplane("XY").box(
            lo.sensor.l + 2 * P.CLEARANCE,
            lo.sensor.w + 2 * P.CLEARANCE,
            fence_h + 1.0, centered=(True, True, False),
        ).translate((lo.sensor.cx, lo.sensor.cy, lo.z_floor - 0.5))
        part = part.union(outer.cut(inner))

    # --- USB-C exit, +X end block (toward the elbow) -----------------
    # A STEPPED bore, not a hole plus an external collar.
    #
    # Since the bosses moved into the end blocks there is now ~6 mm of solid
    # material between the cavity wall and the outer face. That depth is the
    # strain relief: the plug's metal shell passes through a close-fitting
    # inner bore while its overmould noses into a wider outer counterbore.
    # The old protruding collar added 4 mm to the overall length to do a job
    # the end block already does.
    usb_z = lo.z_c3_pcb + P.C3_PCB_T + P.C3_USB_H / 2

    # Plug recess: cut between the two screw bosses, back to USB_FACE_WALL of
    # the receptacle, so the overmould can follow the shell in and the plug
    # actually latches. The recess walls are the strain relief - the overmould
    # is captured on all four sides, so the cable cannot lever the connector.
    recess_l = P.USB_PLUG_W + 2 * P.CLEARANCE    # along X now
    recess_h = P.USB_PLUG_H + 2 * P.CLEARANCE

    # Strain-relief collar. The -Y side wall is only WALL thick, so unlike
    # the old end-block position there is almost nothing gripping the plug's
    # overmould. A short external collar restores that. It is a local bump on
    # the little-finger side, not a change to the overall width.
    # The fillet runs on the vertical edges, so it is bounded by the collar's
    # Y depth, not by its X length: at r = USB_COLLAR_L/2 the two corner arcs
    # meet and OCCT throws StdFail_NotDone.
    collar_r = min(1.0, P.USB_COLLAR_L / 2 - 0.3)
    collar = (
        _rounded_block(recess_l + 2 * P.WALL, P.USB_COLLAR_L,
                       recess_h + 2 * P.WALL, collar_r, centre_z=True)
        .translate((lo.c3.cx,
                    -lo.body_w / 2 - P.USB_COLLAR_L / 2 + 0.1, usb_z))
    )
    part = part.union(collar)

    # Span the recess by its two END POSITIONS rather than a depth plus an
    # offset. The offset form is easy to get backwards, and a cut placed
    # outward instead of inward still yields a perfectly valid solid - it
    # just leaves the wall in place.
    #
    # Corner fillet is kept to 0.5 mm: at 1.0 mm the rounded corners intrude
    # on a rectangular overmould. And the inner end overshoots into the
    # cavity, because the fillet rounds the vertical corners along the whole
    # cut - stopping level with the plug's inner face would put those rounded
    # corners exactly where the plug's corners need to be. Overshooting
    # removes nothing extra; that volume is already void.
    y_outer = -lo.body_w / 2 - P.USB_COLLAR_L - 0.5   # past the collar face
    y_inner = -lo.cav_w / 2 + 2.5                     # past the cavity wall
    part = part.cut(
        _rounded_block(recess_l, y_inner - y_outer, recess_h, 0.5,
                       centre_z=True)
        .translate((lo.c3.cx, (y_outer + y_inner) / 2, usb_z))
    )

    # --- Heat-set insert bores ----------------------------------------
    # No free-standing posts: the bosses live in the solid end material
    # outside the cavity, so these are simply bored down from the lid seat.
    for (x, y) in lo.screw_xy:
        part = part.cut(
            cq.Workplane("XY")
            .circle(P.INSERT_D / 2)
            .extrude(P.INSERT_DEPTH)
            .translate((x, y, lo.z_lid - P.INSERT_DEPTH))
        )

    # --- Strap slots, cut through the end blocks ----------------------
    # No separate lugs. The slot is a vertical through-slot sharing the same
    # X as the screw boss, offset from it in Y. The insert only reaches
    # INSERT_DEPTH down from the lid seat, and the slot sits between the two
    # bosses, so the two features never meet.
    slot_w = P.STRAP_W + 2 * P.CLEARANCE
    slot_t = P.STRAP_T + 2 * P.CLEARANCE
    for sx in (+1, -1):
        cx = sx * lo.lug_x
        slot = (
            cq.Workplane("XY")
            .box(slot_t, slot_w, lo.body_h + P.BOSS_PROUD + 4.0,
                 centered=(True, True, False))
            .edges("|Z").fillet(min(slot_t, slot_w) / 2 - 0.01)
            .translate((cx, 0, -P.BOSS_PROUD - 2.0))
        )
        part = part.cut(slot)

    return part


# ─────────────────────────────────────────────────────────────────────
#  Lid
# ─────────────────────────────────────────────────────────────────────

def lid(lo: L.Layout = None):
    lo = lo or L.compute()

    # A flat plate, located by the four screws.
    #
    # There is deliberately no alignment lip. The boards fill the cavity to
    # within 0.25 mm, so any lip dropping into the cavity mouth lands on the
    # OLED glass - and a lip that clears the glass would be a 0.3 mm wide rim,
    # too thin to print. The screws locate the lid to well within the
    # clearance a lip would have provided anyway.
    part = _rounded_block(lo.body_l, lo.body_w, P.LID_T, P.FILLET_OUTER)

    # OLED window. Sized from the ACTIVE area, not the module outline, and
    # offset because the lit area does not sit centred on the glass.
    win_l = P.OLED_ACTIVE_L + 2 * P.OLED_WINDOW_MARGIN
    win_w = P.OLED_ACTIVE_W + 2 * P.OLED_WINDOW_MARGIN
    part = part.cut(
        cq.Workplane("XY")
        .box(win_l, win_w, P.LID_T + 2.0)
        .translate((lo.oled.cx, lo.oled.cy + P.OLED_ACTIVE_OFF_Y, 0))
    )

    # OLED retention pegs. The module otherwise floats in the cavity with
    # 0.3 mm of air above it and a wire gap below - nothing locates it, so
    # it can shift and take the display out from behind its window. Two
    # diagonal pegs descending into its own mounting holes fix that.
    if P.OLED_HOLES:
        px, py, hd = P.OLED_HOLES
        peg_d = hd - 2 * P.PIN_CLEARANCE
        peg_h = (lo.z_lid - lo.z_oled) - 0.2
        for (sx, sy) in ((+1, -1), (-1, +1)):
            part = part.union(
                cq.Workplane("XY")
                .circle(peg_d / 2)
                .extrude(peg_h)
                .translate((lo.oled.cx + sx * px / 2,
                            lo.oled.cy + sy * py / 2, -peg_h))
            )

    # Screw through-holes, counterbored for the M2 head.
    #
    # Two of the four are a CLOSE fit so their screws double as dowels.
    # There is nowhere on this part for a lip or an alignment pin - the
    # boards leave 0.25 mm inside the cavity and the end blocks are packed
    # with the strap slots and bosses - so the screws are the only
    # registration available. Tightening two of them costs nothing and takes
    # the lid's lateral play from about +/-0.15 mm to +/-0.05 mm.
    #
    # Two, not four: four close-fit holes in a printed part will not all line
    # up, and the lid would bind rather than locate.
    dowels = {lo.screw_xy[0], lo.screw_xy[3]}   # one diagonal pair
    for (x, y) in lo.screw_xy:
        slack = 0.05 if (x, y) in dowels else 0.15
        part = part.cut(
            cq.Workplane("XY")
            .circle(P.SCREW_D / 2 + slack)
            .extrude(P.LID_T + 2.0)
            .translate((x, y, -1.0))
        )
        part = part.cut(
            cq.Workplane("XY")
            .circle(P.SCREW_HEAD_D / 2)
            .extrude(1.2)
            .translate((x, y, P.LID_T - 1.2))
        )

    return part.translate((0, 0, lo.z_lid))


# ─────────────────────────────────────────────────────────────────────
#  Boss shims
# ─────────────────────────────────────────────────────────────────────

def shims(lo: L.Layout = None):
    """
    Stackable shims that raise the sensor boss without a reprint.

    Contact pressure is the dominant variable in reflectance PPG and the team
    cannot iterate on hardware, so the boss height is made adjustable in the
    field instead of being fixed at design time.
    """
    lo = lo or L.compute()
    out = {}
    for t in P.SHIM_THICKNESSES:
        outer = _rounded_block(
            lo.aperture.l + 2 * P.SEAL_RING_W + 2 * P.WALL + 2 * P.SHIM_CLEARANCE,
            lo.aperture.w + 2 * P.SEAL_RING_W + 2 * P.WALL + 2 * P.SHIM_CLEARANCE,
            t, 1.0,
        )
        hole = cq.Workplane("XY").box(
            lo.aperture.l, lo.aperture.w, t + 2.0, centered=(True, True, False),
        ).translate((0, 0, -1.0))
        out[f"shim_{t:.1f}mm".replace(".", "p")] = outer.cut(hole)
    return out


# ─────────────────────────────────────────────────────────────────────
#  Stage-1 fit gauge
# ─────────────────────────────────────────────────────────────────────

def board_gauge(lo: L.Layout = None):
    """
    Go / no-go gauge for the three board outlines.

    Every board sits in the cavity on 0.25 mm of margin, and every one of
    those outlines is a catalogue figure rather than a caliper reading. A
    board 0.3 mm larger than its datasheet in the wrong axis does not go in,
    and the enclosure is scrap.

    This is a flat plate with a through-pocket per board, each cut to exactly
    the clearance the cavity gives it. Drop the real module in: if it passes
    through, the cavity will take it. If it does not, measure it and change
    params.py before ordering the enclosure.

    Identified by drilled dots beside each pocket - 1 sensor, 2 MCU,
    3 display - rather than embossed text, which is unreliable at this size.
    """
    m = P.max30102()
    lo = lo or L.compute()

    boards = [
        (m["L"], m["W"], 1),                 # MAX30102
        (P.C3_W, P.C3_L, 2),                 # ESP32-C3, as rotated
        (P.OLED_L, P.OLED_W, 3),             # SSD1306
    ]
    t = 2.5
    pad = 5.0
    dot_d, dot_gap = 1.6, 3.0

    widths = [b[0] + 2 * P.CLEARANCE for b in boards]
    plate_l = sum(widths) + pad * (len(boards) + 1)
    plate_w = max(b[1] + 2 * P.CLEARANCE for b in boards) + 2 * pad + 6.0

    part = _rounded_block(plate_l, plate_w, t, 2.0)

    x = -plate_l / 2 + pad
    for (bl, bw, dots), w in zip(boards, widths):
        h = bw + 2 * P.CLEARANCE
        cx = x + w / 2
        cy = 3.0
        part = part.cut(
            cq.Workplane("XY")
            .box(w, h, t + 2.0, centered=(True, True, False))
            .translate((cx, cy, -1.0))
        )
        # identifying dots, below the pocket
        dy = cy - h / 2 - 3.5
        x0 = cx - dot_gap * (dots - 1) / 2
        for i in range(dots):
            part = part.cut(
                cq.Workplane("XY")
                .circle(dot_d / 2)
                .extrude(t + 2.0)
                .translate((x0 + i * dot_gap, dy, -1.0))
            )
        x += w + pad

    return part


def fit_gauge():
    """
    A cheap coupon that proves the shop's tolerances before the real part.

    Does NOT depend on the MAX30102 variant: the optical aperture is a
    property of the chip package, and the USB-C, strap slot and heat-set
    insert are all independent of which breakout is fitted. So this can be
    ordered immediately, in parallel with getting the sensor measured.

    Each feature repeats at three clearances. Whichever column gives a firm
    slip fit sets CLEARANCE in params.py for the real build.
    """
    trials = (0.15, 0.25, 0.35)
    pitch = 20.0
    plate_l = pitch * len(trials) + 12.0
    plate_w = 46.0
    t = 3.0

    part = _rounded_block(plate_l, plate_w, t, 2.0)
    x0 = -pitch * (len(trials) - 1) / 2

    for i, c in enumerate(trials):
        x = x0 + i * pitch

        # USB-C shell opening
        part = part.cut(
            cq.Workplane("XY").box(
                P.C3_USB_W + 2 * c, P.C3_USB_H + 2 * c, t + 2.0,
            ).translate((x, 16.0, 0))
        )

        # MAX30102 optical aperture (5.6 x 3.3 chip package)
        part = part.cut(
            cq.Workplane("XY").box(5.6 + 2 * c, 3.3 + 2 * c, t + 2.0)
            .translate((x, 6.0, 0))
        )

        # Heat-set insert bore
        part = part.cut(
            cq.Workplane("XY").circle(P.INSERT_D / 2 + c - 0.25)
            .extrude(t + 2.0).translate((x, -3.0, -1.0))
        )

        # Screw clearance hole
        part = part.cut(
            cq.Workplane("XY").circle(P.SCREW_D / 2 + c)
            .extrude(t + 2.0).translate((x, -10.0, -1.0))
        )

    # Strap slot - one only, the band is a single known width.
    part = part.cut(
        cq.Workplane("XY")
        .box(P.STRAP_W + 2 * P.CLEARANCE, P.STRAP_T, t + 2.0)
        .translate((0, -18.0, 0))
    )

    return part
