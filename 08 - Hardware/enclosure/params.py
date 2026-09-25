"""
Wrist Enclosure - Radial-Artery PPG Acquisition Node
Single source of truth for every dimension in the model.

Thesis: An Internet of Things-Based Framework for Cardiac Arrhythmia Detection
        via Photoplethysmography and Machine Learning
Team:   3CPE-2A, University of the East, Caloocan

Nothing else in this project hard-codes a number. Change it here, re-run
build.py, and every part regenerates consistently.

COORDINATE SYSTEM
-----------------
Origin sits at the centre of the skin-contact face.

    +X  along the forearm, toward the ELBOW   (cable exits this way)
    +Y  across the wrist,  toward the THUMB   (radial / lateral side)
    +Z  away from the skin, toward the OLED

The radial artery runs on the thumb side of the volar wrist, so the sensor
boss is deliberately offset to +Y. Centring it would put the aperture over
the flexor tendons instead of the artery.

VERIFICATION STATUS
-------------------
Every value tagged VERIFY is a datasheet or catalogue nominal that has NOT
been confirmed against the physical parts. The enclosure goes to a commercial
print shop with no iteration loop, so each one must be checked with calipers
before an STL is released. See verify_report() at the bottom.
"""

# ---------------------------------------------------------------------
#  Manufacturing
# ---------------------------------------------------------------------

# Per-side clearance on every pocket that receives a component.
# 0.25 mm is the forgiving FDM value. Drop to 0.15 only after the Stage-1
# fit gauge proves the shop actually holds that tolerance.
CLEARANCE = 0.25

WALL = 1.6           # side walls (4 perimeters at a 0.4 mm nozzle)
FLOOR = 1.2          # skin-contact face thickness beneath the sensor pocket
LID_T = 1.6          # lid thickness away from the OLED window
FILLET_OUTER = 2.0   # outer vertical edges - comfort against the wrist
FILLET_TOP = 1.0

# How the lid is fastened. Screws either way - never snap-fits, because a
# snap-fit cannot be tuned after printing and we get exactly one attempt.
#
#   "insert"   M2 screw into a brass heat-set insert. Durable, survives many
#              open/close cycles - which matters, because the shims are meant
#              to be swapped while tuning skin contact pressure. Needs the
#              inserts bought, plus a soldering iron and a steady hand to set
#              them square. A plain conical tip can mangle them.
#
#   "selftap"  M2 self-tapping screw straight into the printed boss. Needs
#              nothing but a screwdriver, which matters when the person doing
#              the assembly is remote. Printed threads wear after roughly
#              5-10 cycles, so tune the shim stack first and assemble last.
#
# Pick on the assembler's toolkit, not on elegance.
FASTENER_MODE = "insert"

SCREW_D = 2.0
_BORE = {
    "insert": 3.2,   # VERIFY against the actual insert (M2 standard: 3.2)
    "selftap": 1.7,  # M2 self-tap pilot in PETG/PLA
}
INSERT_D = _BORE[FASTENER_MODE]
INSERT_DEPTH = 4.0 if FASTENER_MODE == "insert" else 6.0
BOSS_D = 5.0         # 0.9 mm of wall around a 3.2 mm insert
SCREW_HEAD_D = 3.8
BOSS_GAP = 0.3       # clear air between a boss and the cavity wall
BOSS_EDGE = 1.0      # material between a boss and the outer surface

# ---------------------------------------------------------------------
#  ESP32-C3 Super Mini                                        [VERIFY]
# ---------------------------------------------------------------------

C3_L = 22.5          # along X
C3_W = 18.0          # along Y
C3_PCB_T = 1.2
C3_TALL_TOP = 3.4    # tallest top-side component (the USB-C shell)
C3_TALL_BOT = 0.6    # the Super Mini carries very little on its underside

# USB-C receptacle, centred on the 18 mm short edge.
C3_USB_W = 9.0          # shell width  (Y)
C3_USB_H = 3.2          # shell height (Z)
C3_USB_PROTRUDE = 1.2   # how far the shell overhangs the PCB edge

# Ceramic chip antenna occupies the short edge opposite the USB-C.
# No solid material, no screw boss and no metal inside this volume, or the
# 2.4 GHz link degrades badly.
C3_ANT_KEEPOUT = 5.0

# ---------------------------------------------------------------------
#  SSD1306 0.96 inch OLED, 128x64, I2C 4-pin                  [VERIFY]
# ---------------------------------------------------------------------

OLED_L = 27.3        # along X
OLED_W = 27.8        # along Y
OLED_PCB_T = 1.6
OLED_TOTAL_T = 4.1   # PCB + glass + FPC fold

# Active (lit) area. NOTE: it is not centred on the module - it sits toward
# the ribbon edge. OLED_ACTIVE_OFF_Y must be measured, not assumed, or the
# window will clip the display.
OLED_ACTIVE_L = 21.74
OLED_ACTIVE_W = 10.86
OLED_ACTIVE_OFF_Y = 4.5   # [VERIFY] active-area centre offset from module centre

OLED_WINDOW_MARGIN = 0.6  # window oversize around the active area

# ---------------------------------------------------------------------
#  MAX30102 - UNCONFIRMED VARIANT
# ---------------------------------------------------------------------
#
# The team reports an 8-pin module from a Shopee PH listing. At least two
# incompatible 8-pin MAX30102 breakouts are sold under near-identical
# titles, and their optical apertures sit in different places:
#
#   "large"  ~25.4 x 20.3 mm  - regulator + level shifter on board
#   "small"  ~12.7 x 12.7 mm  - bare breakout, castellated
#
# On a radial-pulse design the APERTURE POSITION IS THE DESIGN DATUM: the
# boss is placed from it and the light-seal ring is built around it. A guess
# here silently puts the LEDs over tendon instead of artery, which looks like
# a flat PPG trace and gets misdiagnosed as a firmware fault.
#
# Set MAX_VARIANT once measured. build.py refuses to emit a release STL
# while this is None.

MAX_VARIANT = "mhetlive"

_MAX_PRESETS = {
    # L (X), W (Y), PCB thickness, tallest back-side component,
    # aperture centre offset from PCB centre (dx, dy), aperture size.
    #
    # AP_PROVISIONAL marks an aperture position that has NOT been measured.
    # build.py will refuse to call such a build release-ready.

    # Identified from the MakerLab PH product photo: MH-ET LIVE MAX30102.
    # Two rows of four pads - GND/RD/IRD/INT and VIN/SDA/SCL/GND - plus 1V8
    # and 3V3 pull-up selector pads on the right edge. Four corner mounting
    # holes and two elongated plated slots on the left and right edges.
    "mhetlive": dict(L=21.0, W=16.0, T=1.2, BACK=1.0, AP_DX=0.0, AP_DY=0.0,
                     AP_L=5.6, AP_W=3.3, AP_PROVISIONAL=True),

    "large": dict(L=25.4, W=20.3, T=1.6, BACK=1.6, AP_DX=0.0, AP_DY=0.0,
                  AP_L=5.6, AP_W=3.3, AP_PROVISIONAL=True),
    "small": dict(L=12.7, W=12.7, T=1.0, BACK=0.8, AP_DX=0.0, AP_DY=0.0,
                  AP_L=5.6, AP_W=3.3, AP_PROVISIONAL=True),
}

# ---------------------------------------------------------------------
#  Board mounting holes
# ---------------------------------------------------------------------
#
# Both the SSD1306 module and the MH-ET LIVE MAX30102 carry four corner
# mounting holes (team-confirmed 2026-09-11; the MAX30102's are visible in
# the MakerLab product photo).
#
# These do NOT let the screw bosses move inside the cavity - measured, the
# C3 and the sensor block every fastener column, see the design note. They
# are used instead for what they are actually good at: positive location.
#
# The sensor one matters most. Aperture registration is the highest-stakes
# tolerance in the design - if the module sits a millimetre off, the LEDs
# miss the artery - and two locating pins through real holes hold it far
# better than the printed fence they replace.
#
# Give each as (pitch_x, pitch_y, hole_diameter) in mm, centre-to-centre.
# Leave as None and the feature is simply not generated; the printed fence
# is used for the sensor instead. Do not guess these: an off-pitch pin is
# worse than no pin, because the board then will not seat flat at all.

OLED_HOLES = None    # -> (pitch_x, pitch_y, hole_d), e.g. (23.5, 24.0, 2.2)
MAX_HOLES = None     # -> (pitch_x, pitch_y, hole_d)

PIN_CLEARANCE = 0.15  # per side, pin to hole - pins must not be a press fit

# ---------------------------------------------------------------------
#  Sensor boss and optical interface
# ---------------------------------------------------------------------
#
# The boss lifts the sensor proud of the contact face so the strap presses
# it into the skin over the artery. Contact pressure is the single largest
# determinant of reflectance-PPG quality: too little and the optical path is
# broken by an air gap, too much and the radial artery is occluded and the
# pulsatile component vanishes.
#
# Because the team cannot iterate on hardware, BOSS_PROUD ships as a set of
# stackable shims rather than a fixed height.

# How far the MAX30102's optical window might sit from where we assume it is.
#
# The aperture position has not been measured, and a window that misses the
# chip is a dead sensor. Rather than block the print on a measurement nobody
# can take right now, the aperture is opened up by this much in every
# direction, so the chip is exposed anywhere within the band.
#
# The cost is small: a larger window does not increase LED-to-photodiode
# leakage through the housing - the seal ring still surrounds it - and the
# module's 21 x 16 mm PCB still rests on plenty of floor around the opening.
#
# Set to 0.0 once AP_DX / AP_DY are measured, to get the tight window back.
AP_UNCERTAINTY = 1.5

BOSS_PROUD = 0.8              # nominal boss height above the contact face
SHIM_THICKNESSES = (0.4, 0.8, 1.2)
SHIM_CLEARANCE = 0.15

# Light-seal ring around the aperture. Without it the red/IR LEDs couple
# straight through the housing into the photodiode, producing a large DC
# offset with almost no AC component - the classic flat wrist trace.
SEAL_RING_W = 1.2
SEAL_RING_PROUD = 0.4

# Sensor offset toward the thumb, to sit lateral of the flexor tendons and
# over the radial artery. Makes the body asymmetric about the strap axis.
SENSOR_OFFSET_Y = 6.0         # [VERIFY on a real wrist]

# ---------------------------------------------------------------------
#  Strap - velcro elastic band through integral end lugs
# ---------------------------------------------------------------------

STRAP_W = 16.0       # band width (Y) - deliberately narrow; see layout.py
STRAP_T = 3.0        # band thickness (X) - generous, velcro is bulky
LUG_L = 0.0          # deprecated - lugs are merged into the end blocks
LUG_WALL = 1.8       # material each side of the slot

# ---------------------------------------------------------------------
#  Overall height / wiring plenum
# ---------------------------------------------------------------------
#
# Left to itself the Z stack packs as tightly as the boards allow: floor,
# sensor, 1.2 mm of wire gap, C3, 1.2 mm of wire gap, display, lid. That is
# about 18 mm overall and it is brutal to assemble - eight hand-soldered
# 30 AWG conductors have to live in two 1.2 mm slots, with nowhere for the
# slack to go and no way to get a hand or an iron near a joint once the
# lower board is down.
#
# TARGET_TOTAL_H asks for a specific overall height instead, measured from
# the light-seal ring (the lowest point, against the skin) to the top of the
# lid - i.e. what a caliper across the finished watch reads. Everything the
# request adds over the natural stack goes into ONE place: the wiring plenum
# between the ESP32-C3 and the display. A single tall void is far more useful
# than two mediocre gaps, because wire can be dressed, coiled and strain-
# relieved in it, and it can be reached from above with the lid off.
#
# Set to None to go back to the tightest stack the boards allow.
TARGET_TOTAL_H = 35.0

# Below this much plenum the display simply rests on the C3 as before. Above
# it the board would be floating in mid-air, so parts.py grows a pair of
# support rails up the +/-X cavity walls for it to land on.
PLENUM_SHELF_THRESHOLD = 3.0

# How far each rail underlaps the display's +/-X edges. The rails run the
# full height of the cavity, so the cavity is simply narrower in X below the
# display; every lower board still passes down through the gap between them.
# Only the +/-X walls carry rails: the C3 is pushed hard against the -Y wall
# to reach its USB-C port, so a rail there would block it going in.
OLED_SHELF_W = 1.25

# ---------------------------------------------------------------------
#  Contact face geometry
# ---------------------------------------------------------------------

WRIST_R = 30.0       # transverse curvature of the volar wrist [VERIFY]
EDGE_RELIEF = 0.8    # chamfer at the contact-face rim so it cannot dig in

# ---------------------------------------------------------------------
#  Cable exit - points toward the elbow, runs up the forearm to the Pi
# ---------------------------------------------------------------------

# A USB-C plug's metal shell inserts about 6.5 mm, and its overmould sits
# immediately behind it. So for the plug to seat, the enclosure material in
# front of the receptacle must be either thin enough to ignore, or bored wide
# enough for the overmould to pass. With the screw bosses now buried in the
# end blocks that material is 6.3 mm thick — far too much to ignore.
#
# The bosses are at y = +/-11.9 and the port is at y = -2, so the block only
# needs its full thickness at the bosses. A recess between them brings the
# outer face in to USB_FACE_WALL of the receptacle.
USB_PLUG_TONGUE = 6.5    # plug metal-shell insertion length
USB_PLUG_W = 13.0        # overmould cross-section to accommodate; generous,
USB_PLUG_H = 8.0         # because cable overmoulds vary from ~8 to ~16 mm
# Material left in front of the receptacle. This must be ZERO.
#
# A tempting mistake is to leave a thin face wall with a shell-sized slot, so
# the port looks neat and dust is kept out. It does not work: when a USB-C
# plug is fully mated its overmould front face ends up AT the receptacle face,
# so the overmould has to sweep the whole wall depth. Any face wall at all is
# subtracted directly from insertion depth, and the tongue is only 6.5 mm to
# begin with. A 1.2 mm wall was measured obstructing the plug by 111.6 mm3.
#
# If a neat faceplate is ever wanted, the way to get it is a panel-mount
# USB-C extension pigtail inside the enclosure, not a thinner wall.
USB_FACE_WALL = 0.0

# External strain-relief collar on the -Y side. The side wall is only WALL
# thick, so without this almost nothing grips the plug overmould. A local
# bump on the little-finger side; it does not change the overall width.
USB_COLLAR_L = 3.0


def max30102():
    """Resolve the MAX30102 geometry, or fail loudly if still unconfirmed."""
    if MAX_VARIANT is None:
        raise ValueError(
            "MAX_VARIANT is unset. The MAX30102 aperture position is the design "
            "datum for the sensor boss and light seal, so it cannot be guessed.\n"
            "Measure the module (PCB LxWxT, aperture centre offset, aperture "
            "size, tallest back-side component) and set MAX_VARIANT in params.py."
        )
    if MAX_VARIANT not in _MAX_PRESETS:
        raise ValueError(
            "Unknown MAX_VARIANT %r; expected one of %s"
            % (MAX_VARIANT, sorted(_MAX_PRESETS))
        )
    return _MAX_PRESETS[MAX_VARIANT]


def verify_report():
    """Everything a caliper still has to confirm before an STL is released."""
    return [
        ("MAX30102", "variant, PCB size, aperture offset + size, back-side height"),
        ("ESP32-C3", "PCB 22.5x18.0, USB-C shell 9.0x3.2, protrusion 1.2"),
        ("SSD1306", "PCB 27.3x27.8, total thickness 4.1"),
        ("SSD1306", "OLED_ACTIVE_OFF_Y - the active area is not centred"),
        ("Insert", "brass heat-set insert outer diameter (assumed 3.2)"),
        ("Wrist", "WRIST_R and SENSOR_OFFSET_Y against the actual wearer"),
    ]
