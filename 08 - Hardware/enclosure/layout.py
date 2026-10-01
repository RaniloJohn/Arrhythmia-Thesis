"""
Derived geometry for the wrist enclosure.

params.py holds the raw measurements; this module turns them into the
envelope, the Z stack-up and the component placements that the part files
consume. Keeping the arithmetic here means body and lid can never disagree
about where the cavity is.

Every dimension here is COMPUTED. If a number looks wrong, fix its input in
params.py rather than patching a value in this file.
"""

from dataclasses import dataclass

import params as P

# Vertical room for hookup wire between stacked boards. Four conductors of
# 30 AWG silicone wire plus the solder fillet on the pads underneath.
WIRE_GAP = 1.2

# Clearance between the top of the OLED glass and the underside of the lid.
TOP_GAP = 0.3


@dataclass(frozen=True)
class Rect:
    """An axis-aligned footprint, centred at (cx, cy)."""
    cx: float
    cy: float
    l: float   # along X
    w: float   # along Y

    @property
    def x_min(self): return self.cx - self.l / 2

    @property
    def x_max(self): return self.cx + self.l / 2

    @property
    def y_min(self): return self.cy - self.w / 2

    @property
    def y_max(self): return self.cy + self.w / 2


@dataclass(frozen=True)
class Layout:
    # Footprints
    sensor: Rect
    c3: Rect
    oled: Rect

    # Cavity (inner) extents
    cav_l: float
    cav_w: float
    cav_h: float

    # Outer body, excluding the strap lugs
    body_l: float
    body_w: float
    body_h: float          # the LOWER SHELL only - its top face is the lid seat

    # Assembled and overall
    assembly_h: float      # body + lid, excluding the proud boss
    total_l: float
    total_h: float

    # Z levels, measured from the nominal skin-contact face at z = 0
    z_floor: float         # inner floor - the sensor's optical face rests here
    z_sensor_top: float
    z_c3: float            # underside of the C3's bottom-side components
    z_c3_pcb: float        # the C3 PCB itself
    z_c3_top: float
    z_oled: float
    z_oled_top: float
    z_lid: float
    plenum: float          # C3 top -> display underside: the hand-wiring void

    # Feature placements
    lug_x: float           # lug block centre, +/- this in X
    screw_xy: tuple        # (x, y) of each of the four screw bosses
    aperture: Rect         # optical window in the contact face


def compute() -> Layout:
    """Resolve the full layout. Raises if the MAX30102 variant is unset."""
    m = P.max30102()

    # --- Footprints ------------------------------------------------------
    # The sensor is pushed toward the thumb so it lands over the radial
    # artery rather than the flexor tendons. The C3 is nudged the opposite
    # way to keep the stack balanced and clear of the sensor's wiring.
    sensor = Rect(0.0, P.SENSOR_OFFSET_Y, m["L"], m["W"])
    oled = Rect(0.0, 0.0, P.OLED_L, P.OLED_W)

    # --- The C3 is rotated 90 degrees so its USB-C faces -Y ---------------
    #
    # It cannot face +X. The strap slots have to live in the ±X end blocks,
    # and they are vertical through-slots spanning the full height, so the
    # band physically occupies the middle of each end block. With the port
    # also at +X the two overlap by 357 mm3 — the strap would thread straight
    # through the space the plug needs. No clearance tweak fixes that; it is
    # a topology conflict, not a tolerance one.
    #
    # Smartwatches resolve it the same way: straps at 12 and 6, ports and
    # buttons at 3. So the board turns 90 degrees and the cable leaves from
    # the ulnar (little-finger) side, away from the thumb-side sensor boss,
    # and is routed along the forearm from there.
    c3_l = P.C3_W          # 18.0 now lies along X
    c3_w = P.C3_L          # 22.5 now lies along Y

    half_l = max(P.OLED_L, c3_l, m["L"]) / 2 + P.CLEARANCE

    # --- Cavity ----------------------------------------------------------
    # The body stays symmetric in Y even though the sensor is not, so the
    # strap pulls evenly across the wrist. That costs a few millimetres of
    # width and is worth it: an asymmetric body rotates under tension, and
    # rotation moves the aperture off the artery.
    #
    # The rotated C3 is shorter in Y than the sensor and OLED spans, so it
    # cannot widen the cavity and can be placed after half_w is known.
    half_w = max(max(abs(p.y_min), abs(p.y_max))
                 for p in (sensor, oled)) + P.CLEARANCE

    # Push the C3 toward -Y until its USB-C shell is CLEARANCE short of the
    # inner wall face.
    #
    # Not flush. Flush means zero clearance, and the board has to be lowered
    # into the cavity past that wall during assembly - at 0.00 mm it scrapes,
    # and one print tolerance the wrong way and it simply will not go in.
    # The 0.25 mm costs nothing at the port: the plug recess is bored to
    # overmould size through both the wall and the collar, so the overmould
    # still reaches the receptacle face and engagement stays well past the
    # minimum.
    c3 = Rect(0.0,
              -half_w + P.C3_USB_PROTRUDE + P.CLEARANCE + c3_w / 2,
              c3_l, c3_w)

    parts = (sensor, c3, oled)

    cav_l = 2 * half_l
    cav_w = 2 * half_w

    # --- Z stack ---------------------------------------------------------
    #   skin | floor | MAX30102 (face down) | C3 | SSD1306 (face up) | lid
    z_floor = P.FLOOR
    z_sensor_top = z_floor + m["T"] + m["BACK"]
    z_c3 = z_sensor_top + WIRE_GAP
    z_c3_pcb = z_c3 + P.C3_TALL_BOT
    z_c3_top = z_c3_pcb + P.C3_PCB_T + P.C3_TALL_TOP
    # --- Wiring plenum ---------------------------------------------------
    # The gap between the C3 and the display is the one place in the stack
    # that can absorb extra height without moving anything that has to line
    # up with a hole in the shell: the optical aperture, the USB-C port and
    # the strap slots are all below it. So if params.py asks for a taller
    # watch, the whole difference is spent here, as hand-wiring room.
    natural_z_lid = z_c3_top + WIRE_GAP + P.OLED_TOTAL_T + TOP_GAP
    natural_total = natural_z_lid + P.LID_T + P.BOSS_PROUD + P.SEAL_RING_PROUD

    plenum = WIRE_GAP
    if P.TARGET_TOTAL_H is not None:
        extra = P.TARGET_TOTAL_H - natural_total
        if extra < -1e-9:
            raise ValueError(
                f"TARGET_TOTAL_H = {P.TARGET_TOTAL_H:.1f} mm is shorter than "
                f"the boards themselves allow ({natural_total:.1f} mm with "
                f"every gap already at its minimum).\n"
                f"Raise TARGET_TOTAL_H, "
                f"or set it to None to take whatever the stack comes to."
            )
        plenum += extra

    z_oled = z_c3_top + plenum
    z_oled_top = z_oled + P.OLED_TOTAL_T

    # A hair of clearance above the OLED glass. At exactly zero the lid bears
    # directly on the display, and print tolerance alone would then put it in
    # compression - glass does not forgive that.
    z_lid = z_oled_top + TOP_GAP

    cav_h = z_lid - z_floor

    # --- Screw bosses, and the body length they force ---------------------
    # The boards fill the cavity almost exactly - the OLED leaves 0.25 mm all
    # round - so a screw post cannot stand anywhere inside it without passing
    # through a PCB. The bosses therefore sit OUTSIDE the cavity, buried in
    # solid material at the two ends, which also makes them far stronger than
    # free-standing posts.
    #
    # Putting them at the ends rather than the corners is deliberate: it grows
    # the body along the forearm, where a watch has room, instead of across
    # the wrist, where width is what makes a device uncomfortable.
    boss_r = P.BOSS_D / 2
    screw_x = cav_l / 2 + boss_r + P.BOSS_GAP

    body_l = 2 * (screw_x + boss_r + P.BOSS_EDGE)
    body_w = cav_w + 2 * P.WALL

    # --- Strap slots share the end blocks with the bosses -----------------
    # Previously the strap lugs were separate blocks bolted onto the ends,
    # so the enclosure paid for the boss material AND the lug material — two
    # solid blocks doing two jobs at 7.2 mm and 6.6 mm per side.
    #
    # They can occupy the same X. The strap slot is a vertical through-slot;
    # the insert is bored down from the lid seat and only reaches 4 mm deep.
    # Separating them in Y instead of X lets one block do both, which is
    # where ~14 mm of the old length came from.
    #
    # The cost is strap width: the slot and the two bosses have to share the
    # body's Y extent. STRAP_W is therefore narrower than a watch strap.
    slot_half = (P.STRAP_W + 2 * P.CLEARANCE) / 2
    screw_y_min = slot_half + boss_r + 0.8          # clear the strap slot
    screw_y_max = body_w / 2 - boss_r - P.BOSS_EDGE  # stay inside the body
    if screw_y_min > screw_y_max:
        raise ValueError(
            f"STRAP_W={P.STRAP_W} mm leaves no room for the screw bosses.\n"
            f"  the slot needs the boss centre at |y| >= {screw_y_min:.2f}\n"
            f"  the body wall needs it at        |y| <= {screw_y_max:.2f}\n"
            f"Reduce STRAP_W by about "
            f"{2 * (screw_y_min - screw_y_max):.1f} mm, or widen the body."
        )
    screw_y = (screw_y_min + screw_y_max) / 2

    # The lower shell stops at the lid seat. Including LID_T here would put
    # body and lid in the same 1.6 mm of space - they would interfere, and
    # the assembly would stand a lid-thickness too tall.
    body_h = z_lid
    assembly_h = z_lid + P.LID_T

    # --- Strap slots, inside the end blocks ------------------------------
    # Centred on the same X as the bosses, so they add nothing to the length.
    lug_x = screw_x
    total_l = body_l
    # The lowest point is the seal ring, not the boss it sits on - the ring
    # stands proud of the boss face. Counting only BOSS_PROUD here under-reports
    # the real overall height by SEAL_RING_PROUD.
    total_h = assembly_h + P.BOSS_PROUD + P.SEAL_RING_PROUD

    screw_xy = ((screw_x, screw_y), (screw_x, -screw_y),
                (-screw_x, screw_y), (-screw_x, -screw_y))

    # --- Optical aperture ------------------------------------------------
    # Placed from the sensor's IC position, not from the PCB centre. The
    # MAX30102 die aperture is a property of the chip package, so AP_L/AP_W
    # are fixed; AP_DX/AP_DY are what differ between breakout variants.
    # While the aperture position is provisional the window is opened up by
    # AP_UNCERTAINTY so the chip is exposed wherever it actually sits. Once
    # measured, clear AP_PROVISIONAL and the tight window returns.
    slop = P.AP_UNCERTAINTY if m.get("AP_PROVISIONAL") else 0.0
    aperture = Rect(
        sensor.cx + m["AP_DX"],
        sensor.cy + m["AP_DY"],
        m["AP_L"] + 2 * P.CLEARANCE + 2 * slop,
        m["AP_W"] + 2 * P.CLEARANCE + 2 * slop,
    )

    return Layout(
        sensor=sensor, c3=c3, oled=oled,
        cav_l=cav_l, cav_w=cav_w, cav_h=cav_h,
        body_l=body_l, body_w=body_w, body_h=body_h,
        assembly_h=assembly_h, total_l=total_l, total_h=total_h,
        z_floor=z_floor, z_sensor_top=z_sensor_top,
        z_c3=z_c3, z_c3_pcb=z_c3_pcb, z_c3_top=z_c3_top,
        z_oled=z_oled, z_oled_top=z_oled_top, z_lid=z_lid, plenum=plenum,
        lug_x=lug_x, screw_xy=screw_xy, aperture=aperture,
    )


def antenna_keepout(lo: Layout) -> Rect:
    """
    Volume that must stay free of solid material and metal.

    The C3's ceramic antenna sits on the short edge opposite its USB-C. The
    USB-C faces +X (toward the elbow), so the antenna is at the -X end.
    """
    return Rect(
        lo.c3.x_min - P.C3_ANT_KEEPOUT / 2,
        lo.c3.cy,
        P.C3_ANT_KEEPOUT,
        lo.c3.w,
    )


def describe(lo: Layout) -> str:
    """Human-readable summary, printed by build.py and pasted into the spec."""
    return "\n".join([
        "Envelope",
        f"  lower shell    {lo.body_l:.1f} x {lo.body_w:.1f} x {lo.body_h:.1f} mm",
        f"  assembled      {lo.body_l:.1f} x {lo.body_w:.1f} x {lo.assembly_h:.1f} mm",
        f"  overall length {lo.total_l:.1f} mm",
        f"  with boss      {lo.total_h:.1f} mm tall",
        f"  cavity         {lo.cav_l:.1f} x {lo.cav_w:.1f} x {lo.cav_h:.1f} mm",
        "",
        "Z stack (from the skin-contact face)",
        f"  {lo.z_floor:6.2f}  inner floor / sensor optical face",
        f"  {lo.z_sensor_top:6.2f}  MAX30102 back",
        f"  {lo.z_c3_pcb:6.2f}  ESP32-C3 PCB",
        f"  {lo.z_c3_top:6.2f}  ESP32-C3 tallest component",
        f"  {'':6s}  ({lo.plenum:.2f} mm wiring plenum)",
        f"  {lo.z_oled:6.2f}  SSD1306 PCB",
        f"  {lo.z_oled_top:6.2f}  SSD1306 glass / lid underside",
        "",
        "Key placements",
        f"  aperture centre  ({lo.aperture.cx:.2f}, {lo.aperture.cy:.2f})",
        f"  aperture window  {lo.aperture.l:.2f} x {lo.aperture.w:.2f} mm",
        f"  screw bosses     +/-{abs(lo.screw_xy[0][0]):.1f}, "
        f"+/-{abs(lo.screw_xy[0][1]):.1f}",
    ])
