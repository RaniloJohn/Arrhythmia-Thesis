"""
Simplified solids for the three real boards, placed where the layout says
they go, so the enclosure can be checked for actual fit.

These are deliberately CRUDE - a PCB slab plus the one or two components tall
enough to matter. They exist to answer "does it collide and does it fit", not
to look like the real hardware. Anything that would change a clearance is
modelled; decorative detail is not.

Everything is positioned in the same world coordinates as the enclosure:
origin at the centre of the skin-contact face, +Z away from the skin.
"""

import cadquery as cq

import params as P
import layout as L


def _slab(l, w, t, cx, cy, z):
    """A PCB, sitting with its underside at z."""
    return (
        cq.Workplane("XY")
        .box(l, w, t, centered=(True, True, False))
        .translate((cx, cy, z))
    )


def max30102(lo: L.Layout):
    """
    MH-ET LIVE MAX30102, mounted FACE DOWN.

    The PCB rests on the inner floor. The MAX30102 chip package itself hangs
    below that, down into the aperture well, so the optical path to the skin
    is through air rather than through the housing.
    """
    m = P.max30102()
    s = lo.sensor

    pcb = _slab(m["L"], m["W"], m["T"], s.cx, s.cy, lo.z_floor)

    # The chip package: 5.6 x 3.3 x 1.55 mm, on the underside, projecting
    # down through the floor aperture.
    ic_h = 1.55
    ic = (
        cq.Workplane("XY")
        .box(m["AP_L"], m["AP_W"], ic_h, centered=(True, True, False))
        .translate((s.cx + m["AP_DX"], s.cy + m["AP_DY"], lo.z_floor - ic_h))
    )
    return pcb.union(ic)


def esp32c3(lo: L.Layout):
    """
    ESP32-C3 Super Mini. PCB plus the USB-C shell, which is both the tallest
    component and the one that has to line up with the wall opening.
    """
    # Rotated 90 degrees: the board's 18 mm edge lies along X and the USB-C
    # faces -Y. See the note in layout.py on why it cannot face +X.
    c = lo.c3
    pcb = _slab(c.l, c.w, P.C3_PCB_T, c.cx, c.cy, lo.z_c3_pcb)

    shell = (
        cq.Workplane("XY")
        .box(P.C3_USB_W, P.C3_USB_PROTRUDE + 3.0, P.C3_USB_H,
             centered=(True, False, True))
        .translate((c.cx, c.y_min - P.C3_USB_PROTRUDE,
                    lo.z_c3_pcb + P.C3_PCB_T + P.C3_USB_H / 2))
    )
    return pcb.union(shell)


def ssd1306(lo: L.Layout):
    """SSD1306 module: PCB plus the glass panel bonded to its top face."""
    o = lo.oled
    pcb = _slab(P.OLED_L, P.OLED_W, P.OLED_PCB_T, o.cx, o.cy, lo.z_oled)

    glass_t = P.OLED_TOTAL_T - P.OLED_PCB_T
    glass = _slab(
        P.OLED_ACTIVE_L + 4.0, P.OLED_ACTIVE_W + 6.0, glass_t,
        o.cx, o.cy + P.OLED_ACTIVE_OFF_Y, lo.z_oled + P.OLED_PCB_T,
    )
    return pcb.union(glass)


def usb_plug(lo: L.Layout, length: float = 14.0):
    """
    A USB-C plug sitting where a mated plug sits, for interference testing.

    Fully mated, the plug's overmould front face ends up essentially AT the
    receptacle's front face - the shell shoulder butts against it. So the
    overmould sweeps the entire depth of the enclosure wall in front of the
    port, and any enclosure material narrower than the overmould anywhere in
    that span physically stops the plug from seating.

    This is the model the interference check needs. Arithmetic on parameters
    cannot see a boolean that did not cut.
    """
    # The receptacle face follows the board, which now stands off the wall by
    # CLEARANCE so it can be dropped in. Deriving this from the cavity wall
    # instead would quietly model the plug in the wrong place.
    mouth = lo.c3.y_min - P.C3_USB_PROTRUDE
    z = lo.z_c3_pcb + P.C3_PCB_T + P.C3_USB_H / 2

    # Overmould: from the receptacle face outward (-Y), past the body.
    over = (
        cq.Workplane("XY")
        .box(P.USB_PLUG_W, length, P.USB_PLUG_H, centered=(True, False, True))
        .translate((lo.c3.cx, mouth - length, z))
    )
    # Metal shell: inserted into the receptacle.
    shell = (
        cq.Workplane("XY")
        .box(8.34, P.USB_PLUG_TONGUE, 2.56, centered=(True, False, True))
        .translate((lo.c3.cx, mouth, z))
    )
    return over.union(shell)


def all_components(lo: L.Layout):
    return {
        "max30102": max30102(lo),
        "esp32c3": esp32c3(lo),
        "ssd1306": ssd1306(lo),
    }
