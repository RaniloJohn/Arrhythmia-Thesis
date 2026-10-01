"""
Arrhythmia wrist enclosure - Fusion 360 assembly importer.

HOW TO RUN
----------
1. Run `python build.py --assembly` first, so out/ holds every STEP.
2. In Fusion 360: Utilities -> Scripts and Add-Ins -> Scripts -> the green +
   -> point it at this folder. It then appears in the list as
   "ArrhythmiaEnclosure". Select it and press Run.

Fusion has no headless mode and its API only runs inside the running
application, so this cannot be executed from outside. Everything it does,
though, is deterministic - re-run it after any `build.py` and the assembly
rebuilds from scratch.

WHAT IT BUILDS
--------------
One design containing every printed part plus the reference solids for the
three boards, the mated USB-C plug and the strap band. Each lands as its own
named component, appearance-coded, with the reference bodies grouped so they
can be hidden in one click.

They arrive pre-aligned: CadQuery exports them in a shared origin, with the
centre of the skin-contact face at (0, 0, 0), so no joints or moves are
needed. Section-analyse the result to inspect any clearance directly instead
of taking a written figure on trust.
"""

import os
import traceback

import adsk.core
import adsk.fusion

# name in Fusion, STEP file stem, appearance, is it a reference body
PARTS = [
    ("Body",            "body",           "Plastic - Matte (Grey)",   False),
    ("Lid",             "lid",            "Plastic - Matte (Grey)",   False),
    ("Shim 0.4",        "shim_0p4mm",     "Plastic - Matte (White)",  False),
    ("Shim 0.8",        "shim_0p8mm",     "Plastic - Matte (White)",  False),
    ("Shim 1.2",        "shim_1p2mm",     "Plastic - Matte (White)",  False),
    ("REF MAX30102",    "ref_max30102",   "Plastic - Matte (Red)",    True),
    ("REF ESP32-C3",    "ref_esp32c3",    "Plastic - Matte (Green)",  True),
    ("REF SSD1306",     "ref_ssd1306",    "Plastic - Matte (Blue)",   True),
    ("REF USB-C plug",  "ref_usb_plug",   "Plastic - Matte (Yellow)", True),
    ("REF strap band",  "ref_strap_band", "Plastic - Matte (Black)",  True),
]


def out_dir():
    """out/ sits one level up from this script's folder."""
    here = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(os.path.dirname(here), "out")


def apply_appearance(app, design, occurrence, name):
    """Best-effort appearance. Never let a missing library swatch abort."""
    try:
        lib = app.materialLibraries.itemByName("Fusion Appearance Library")
        if not lib:
            return
        appearance = lib.appearances.itemByName(name)
        if not appearance:
            return
        existing = design.appearances.itemByName(name)
        if not existing:
            existing = design.appearances.addByCopy(appearance, name)
        occurrence.appearance = existing
    except Exception:
        pass


def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface

        folder = out_dir()
        if not os.path.isdir(folder):
            ui.messageBox(
                "No out/ folder found at:\n%s\n\n"
                "Run this first, from the enclosure folder:\n"
                "    python build.py --assembly" % folder)
            return

        missing = [f for _, f, _, _ in PARTS
                   if not os.path.exists(os.path.join(folder, f + ".step"))]
        if missing:
            ui.messageBox(
                "These STEP files are missing from out/:\n\n  %s\n\n"
                "Run:  python build.py --assembly"
                % "\n  ".join(m + ".step" for m in missing))
            return

        doc = app.documents.add(
            adsk.core.DocumentTypes.FusionDesignDocumentType)
        design = adsk.fusion.Design.cast(app.activeProduct)
        design.designType = adsk.fusion.DesignTypes.ParametricDesignType
        root = design.rootComponent

        imp = app.importManager
        placed, refs = [], []

        for label, stem, appearance, is_ref in PARTS:
            path = os.path.join(folder, stem + ".step")
            opts = imp.createSTEPImportOptions(path)
            imp.importToTarget(opts, root)

            occ = root.occurrences.item(root.occurrences.count - 1)
            occ.component.name = label
            apply_appearance(app, design, occ, appearance)
            if is_ref:
                occ.isLightBulbOn = True
                refs.append(occ)
            placed.append(label)

        # Reference solids are not printable parts; make that obvious.
        for occ in refs:
            try:
                occ.component.opacity = 0.45
            except Exception:
                pass

        try:
            app.activeViewport.fit()
            cam = app.activeViewport.camera
            cam.isFitView = True
            app.activeViewport.camera = cam
        except Exception:
            pass

        ui.messageBox(
            "Arrhythmia wrist enclosure assembled.\n\n"
            "Imported %d components:\n  %s\n\n"
            "Everything shares one origin, with (0,0,0) at the centre of the "
            "skin-contact face, so nothing needs joining or moving.\n\n"
            "Components prefixed REF are reference solids for the boards, the "
            "mated USB-C plug and the strap band. They are NOT printed parts "
            "- hide them before exporting for manufacture.\n\n"
            "To inspect a clearance, use Inspect -> Section Analysis rather "
            "than relying on a written figure."
            % (len(placed), "\n  ".join(placed)))

    except Exception:
        if ui:
            ui.messageBox("Script failed:\n%s" % traceback.format_exc())
