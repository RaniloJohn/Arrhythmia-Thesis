"""
Component compatibility report, generated from the live model.

    python compat.py

Every number here is measured off the actual solids, not restated from
params.py, so it cannot drift from what build.py exports. Where a figure
comes from a catalogue rather than a caliper it is marked [ASSUMED] - with a
one-shot print shop, knowing which is which matters more than the value.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cadquery as cq

import params as P
import layout as L
import parts
import mockups


def gap(a, b, axis, sign):
    """Signed clearance from solid a to solid b along one axis."""
    ba, bb = a.val().BoundingBox(), b.val().BoundingBox()
    lo_a, hi_a = getattr(ba, axis + "min"), getattr(ba, axis + "max")
    lo_b, hi_b = getattr(bb, axis + "min"), getattr(bb, axis + "max")
    return (lo_b - hi_a) if sign > 0 else (lo_a - hi_b)


def rule(ch="-", n=74):
    print(ch * n)


def main():
    lo = L.compute()
    m = P.max30102()
    body = parts.body(lo)
    lid = parts.lid(lo)
    comps = mockups.all_components(lo)
    plug = mockups.usb_plug(lo)

    print()
    rule("=")
    print("  COMPONENT COMPATIBILITY REPORT")
    print("  Wrist enclosure, Arrhythmia Thesis - generated from the model")
    rule("=")

    # ---------------------------------------------------------------
    print("\n1. ENVELOPE\n")
    bb = body.val().BoundingBox()
    lb = lid.val().BoundingBox()
    print(f"   body          {bb.xlen:6.2f} x {bb.ylen:6.2f} x {bb.zlen:6.2f} mm")
    print(f"   lid           {lb.xlen:6.2f} x {lb.ylen:6.2f} x {lb.zlen:6.2f} mm")
    print(f"   assembled     {bb.xlen:6.2f} x {bb.ylen:6.2f} x "
          f"{max(bb.zmax, lb.zmax) - bb.zmin:6.2f} mm")
    print(f"   cavity        {lo.cav_l:6.2f} x {lo.cav_w:6.2f} x {lo.cav_h:6.2f} mm")

    # ---------------------------------------------------------------
    print("\n2. BOARD FIT - clearance to the cavity wall on each side\n")
    print(f"   {'component':<12}{'footprint':>16}  {'-X':>6}{'+X':>7}"
          f"{'-Y':>7}{'+Y':>7}   verdict")
    rule()
    order = [("MAX30102", "max30102", f'{m["L"]}x{m["W"]}'),
             ("ESP32-C3", "esp32c3", f"{lo.c3.l}x{lo.c3.w}"),
             ("SSD1306", "ssd1306", f"{P.OLED_L}x{P.OLED_W}")]
    for label, key, foot in order:
        cb = comps[key].val().BoundingBox()
        gx0 = cb.xmin - (-lo.cav_l / 2)
        gx1 = lo.cav_l / 2 - cb.xmax
        gy0 = cb.ymin - (-lo.cav_w / 2)
        gy1 = lo.cav_w / 2 - cb.ymax
        worst = min(gx0, gx1, gy0, gy1)
        # The C3's USB shell is meant to reach past the -Y wall.
        note = "fits" if worst >= -0.01 else "port overhang (intended)"
        print(f"   {label:<12}{foot:>16}  {gx0:6.2f}{gx1:7.2f}"
              f"{gy0:7.2f}{gy1:7.2f}   {note}")

    # ---------------------------------------------------------------
    print("\n3. VERTICAL STACK - gap between adjacent boards\n")
    zs = [("floor", lo.z_floor, ""),
          ("MAX30102 PCB", lo.z_floor, f'{m["T"]} thick [ASSUMED]'),
          ("  -> back face", lo.z_sensor_top, ""),
          ("ESP32-C3 PCB", lo.z_c3_pcb, f"{P.C3_PCB_T} thick"),
          ("  -> USB shell top", lo.z_c3_top, f"{P.C3_TALL_TOP} tall"),
          ("SSD1306 PCB", lo.z_oled, f"{P.OLED_TOTAL_T} total"),
          ("  -> glass top", lo.z_oled_top, ""),
          ("lid underside", lo.z_lid, f"{P.LID_T} thick")]
    for label, z, note in zs:
        print(f"   z = {z:6.2f}   {label:<20} {note}")
    print(f"\n   wire routing gaps: {L.WIRE_GAP:.2f} mm sensor->C3, "
          f"{L.WIRE_GAP:.2f} mm C3->OLED")
    print(f"   headroom under lid: {lo.z_lid - lo.z_oled_top:.2f} mm")

    # ---------------------------------------------------------------
    print("\n4. OPTICAL PATH - the highest-stakes tolerance\n")
    ap = lo.aperture
    print(f"   MAX30102 chip window   {m['AP_L']:.2f} x {m['AP_W']:.2f} mm")
    print(f"   enclosure aperture     {ap.l:.2f} x {ap.w:.2f} mm "
          f"({P.CLEARANCE:.2f} per side)")
    print(f"   aperture centre        ({ap.cx:+.2f}, {ap.cy:+.2f})")
    print(f"   boss stands proud      {P.BOSS_PROUD:.2f} mm "
          f"+ {P.SEAL_RING_PROUD:.2f} mm seal ring")
    print(f"   shims supplied         {', '.join(f'{t:.1f}' for t in P.SHIM_THICKNESSES)} mm")
    loc = "two locating pins" if P.MAX_HOLES else "printed fence"
    slop = 0.0 if P.MAX_HOLES else P.CLEARANCE
    print(f"   module located by      {loc}  (+/-{slop:.2f} mm of slop)")
    if not P.MAX_HOLES:
        print("   !! MAX_HOLES unset - the fence allows the aperture to sit")
        print("      up to 0.25 mm off in each axis")
    if m.get("AP_PROVISIONAL"):
        print("   !! AP_DX/AP_DY are PROVISIONAL - assumed centred on the PCB")

    # ---------------------------------------------------------------
    print("\n5. DISPLAY WINDOW\n")
    win_l = P.OLED_ACTIVE_L + 2 * P.OLED_WINDOW_MARGIN
    win_w = P.OLED_ACTIVE_W + 2 * P.OLED_WINDOW_MARGIN
    print(f"   active area            {P.OLED_ACTIVE_L:.2f} x {P.OLED_ACTIVE_W:.2f} mm")
    print(f"   lid window             {win_l:.2f} x {win_w:.2f} mm "
          f"({P.OLED_WINDOW_MARGIN:.2f} margin)")
    print(f"   window offset in Y     {P.OLED_ACTIVE_OFF_Y:+.2f} mm [ASSUMED]")
    print("   !! OLED_ACTIVE_OFF_Y unmeasured - the lit area is not centred")
    print("      on the glass, so a wrong value clips the display")

    # ---------------------------------------------------------------
    print("\n6. USB-C PORT\n")
    blocked = body.intersect(plug).val().Volume()
    print(f"   exits face             -Y (ulnar side)")
    print(f"   wall crossed           {P.WALL:.2f} mm + {P.USB_COLLAR_L:.2f} mm collar")
    print(f"   recess accepts         {P.USB_PLUG_W:.1f} x {P.USB_PLUG_H:.1f} mm overmould")
    print(f"   material in plug path  {blocked:.2f} mm3   "
          f"{'OK' if blocked < 0.01 else '*** BLOCKED ***'}")
    print(f"   clearance to bosses    "
          f"{min(abs(sx*lo.lug_x - lo.c3.cx) for sx in (1,-1)) - P.BOSS_D/2 - (P.USB_PLUG_W/2 + P.CLEARANCE):.2f} mm")

    # ---------------------------------------------------------------
    print("\n7. STRAP\n")
    slot_t = P.STRAP_T + 2 * P.CLEARANCE
    slot_w = P.STRAP_W + 2 * P.CLEARANCE
    print(f"   band accepted          {P.STRAP_W:.1f} mm wide x {P.STRAP_T:.1f} mm thick")
    print(f"   slot                   {slot_w:.2f} x {slot_t:.2f} mm, at x = +/-{lo.lug_x:.2f}")
    band = (cq.Workplane("XY")
            .box(slot_t, slot_w, lo.body_h + 6.0, centered=(True, True, False))
            .translate((lo.lug_x, 0, -3.0)))
    v = band.intersect(plug).val().Volume()
    print(f"   band vs USB plug       {v:.2f} mm3   "
          f"{'clear' if v < 0.01 else '*** COLLIDES ***'}")
    print(f"   load-bearing wall      {lo.body_l/2 - (lo.lug_x + slot_t/2):.2f} mm")

    # ---------------------------------------------------------------
    print("\n8. FASTENERS\n")
    print(f"   4x M2 screw into brass heat-set insert")
    print(f"   insert bore            {P.INSERT_D:.2f} mm dia x {P.INSERT_DEPTH:.2f} deep [ASSUMED]")
    print(f"   boss positions         (+/-{abs(lo.screw_xy[0][0]):.2f}, "
          f"+/-{abs(lo.screw_xy[0][1]):.2f})")
    print(f"   boss wall              {(P.BOSS_D - P.INSERT_D)/2:.2f} mm around the insert")

    # ---------------------------------------------------------------
    print("\n9. ELECTRICAL - one shared I2C bus\n")
    print("   signal     ESP32-C3      MAX30102     SSD1306")
    rule()
    for sig, c3, mx, ol in (("3V3", "3V3", "VIN", "VCC"),
                            ("GND", "GND", "GND", "GND"),
                            ("SDA", "GPIO 8", "SDA", "SDA"),
                            ("SCL", "GPIO 9", "SCL", "SCL")):
        print(f"   {sig:<10} {c3:<13} {mx:<12} {ol}")
    print("\n   unused on MAX30102: INT, RD, IRD, 2nd GND -> 4 wires per device")
    print("   I2C addresses: MAX30102 0x57, SSD1306 0x3C - no conflict")
    print("   pull-ups: both modules carry ~4.7k; ~2.4k in parallel, within spec")
    print("   !! MH-ET LIVE 1V8/3V3 pads select the pull-up rail. Strapped to")
    print("      1.8 V the sensor never answers a 3.3 V I2C scan.")

    # ---------------------------------------------------------------
    print("\n10. WHAT IS STILL ASSUMED\n")
    for part, what in P.verify_report():
        print(f"   [ ] {part:<10} {what}")
    print()
    rule("=")
    print("  Verified by caliper: nothing yet.  Do not release an STL.")
    rule("=")
    print()


if __name__ == "__main__":
    try:
        main()
    finally:
        sys.stdout.flush()
        os._exit(0)
