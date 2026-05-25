# =============================================================================
# kinetidiff_all.pml
# KinetiDiff — Top-5 Vina-Direct Leads vs ACVR1/ALK2 (PDB: 3MTF)
# =============================================================================
#
# RUN (from repo root):
#   pymol src/visualizations/kinetidiff_all.pml
# OR from inside PyMOL:
#   @src/visualizations/kinetidiff_all.pml
#
# REQUIRES: internet access to fetch PDB:3MTF on first run.
#           On subsequent runs, load the saved session:
#             pymol kinetidiff_all.pse
#
# NOTE: Ligand 3D coordinates are RDKit conformers (from the generation SDF).
#       Each molecule is translated so its centroid falls on the Vina docking
#       site (24.87, -12.54, 38.40); intra-pocket orientation is approximate.
#       H-bond dashes show N/O polar contacts ≤ 3.5 Å.
#
# SCENES  (cycle with PgUp / PgDn, or use the Scene Panel, or F-keys):
#   overview      — all 5 leads + crystal ref overlaid, pocket surface
#   rank01        — Rank 1 alone  │ −11.05 kcal/mol  pKd 8.10  +19.2% vs Xtal
#   rank02        — Rank 2 alone  │ −10.62 kcal/mol  pKd 7.79  2 H-bonds  SO₂
#   rank03        — Rank 3 alone  │ −10.59 kcal/mol  QED 0.812 ★ (best)
#   rank04        — Rank 4 alone  │ −10.55 kcal/mol  SA 2.26   ★ (most synth.)
#   rank05        — Rank 5 alone  │ −10.51 kcal/mol  dual-pyridine  2×F
#   ref_compare   — Rank 1 (gold) vs crystallographic inhibitor (silver)
#   pocket_shape  — Receptor surface view; all leads shown in pocket
# =============================================================================

# ─── 0. GLOBAL RENDER SETTINGS ────────────────────────────────────────────────
bg_color                  white
set antialias,            2
set ray_trace_mode,       1
set ray_shadows,          1
set ray_opaque_background, on
set depth_cue,            1
set fog_start,            0.45
set hash_max,             300
set surface_quality,      1
set cartoon_fancy_helices, 1
set cartoon_fancy_sheets,  1
set cartoon_side_chain_helper, on
set stick_radius,         0.18
set sphere_scale,         0.25
set label_size,           13
set label_color,          black
set label_font_id,        7
set label_bg_color,       white
set label_bg_transparency, 0.25
set dash_width,           3.0
set dash_radius,          0.08
set dash_gap,             0.35
set dash_length,          0.6

# ─── 1. CUSTOM COLOR PALETTE ──────────────────────────────────────────────────
# Five vivid, distinct hues — each tells its own story in the pocket.

# Rank 1: amber gold  (best binder, +19.2% vs crystal)
set_color gold_kd,        [1.00, 0.76, 0.00]
# Rank 2: tomato red  (2 H-bonds, sulfonamide scaffold)
set_color red_kd,         [0.93, 0.23, 0.18]
# Rank 3: emerald green  (best QED = 0.812)
set_color grn_kd,         [0.10, 0.80, 0.38]
# Rank 4: electric cyan  (best SA = 2.26, easiest to make)
set_color cyn_kd,         [0.05, 0.75, 0.95]
# Rank 5: vivid violet  (dual pyridine, 2×F)
set_color vio_kd,         [0.65, 0.12, 0.95]
# Crystallographic ref: cool silver
set_color sil_kd,         [0.75, 0.75, 0.80]
# H-bond dashes: bright canary yellow
set_color hb_yellow,      [1.00, 0.91, 0.08]
# Pocket residue carbons: warm off-white
set_color pkt_c,          [0.96, 0.96, 0.90]
# Hinge residue carbons: warm wheat
set_color hng_c,          [0.97, 0.88, 0.68]

# ─── 2. LOAD RECEPTOR (ACVR1/ALK2 — PDB: 3MTF) ────────────────────────────────
fetch 3MTF, raw_3mtf, async=0

# Strip solvent and small ions
remove raw_3mtf and (resn HOH or resn DOD)
remove raw_3mtf and (elem NA or elem CL or elem MG or elem ZN or elem SO4 or elem PO4)

# Pull out the bound crystallographic inhibitor (all organic HETATM)
select _ref_sel, raw_3mtf and organic
extract ref_ligand, _ref_sel
delete _ref_sel

# Protein backbone only → receptor
select _prot_sel, raw_3mtf and polymer.protein
extract receptor, _prot_sel
delete _prot_sel
delete raw_3mtf

# Add polar H for H-bond proximity detection; hide them immediately
h_add receptor
hide (receptor and elem H)

# ─── 3. RECEPTOR REPRESENTATION ────────────────────────────────────────────────
hide everything, receptor

# Cartoon backbone
show cartoon, receptor
color slate,     receptor
color marine,    receptor and ss H         # α-helices: rich marine blue
color skyblue,   receptor and ss S         # β-strands: sky blue
color lightblue, receptor and ss L+        # loops: light blue

# Ghost surface (whole receptor, very transparent)
show surface, receptor
set transparency, 0.68, receptor
color white, receptor

# ATP hinge region — where kinase inhibitors make backbone H-bonds (~GLY285-GLY288 in ALK2)
# residue numbering from 3MTF; adjust if chain offset differs
select hinge_region, receptor and resi 280-295 and polymer.protein
show sticks,  hinge_region
color hng_c,  hinge_region and elem C
color blue,   hinge_region and elem N
color red,    hinge_region and elem O
hide (hinge_region and elem H)
set stick_radius, 0.14, hinge_region

# Activation loop DFG motif (~resi 354-362)
select dfg_loop, receptor and resi 352-365 and polymer.protein
color salmon, dfg_loop

# P-loop glycine-rich loop (~resi 244-250)
select p_loop, receptor and resi 242-252 and polymer.protein
color paleyellow, p_loop

deselect

# ─── 4. LOAD TOP-5 GENERATED LEADS FROM SDF ───────────────────────────────────
python
import os

# Accept both: run from repo root OR from src/visualizations/
sdf_candidates = [
    "RESULTS/vina_direct/top5_leads_3D.sdf",
    "../../RESULTS/vina_direct/top5_leads_3D.sdf",
]
sdf_path = next((p for p in sdf_candidates if os.path.exists(p)), None)

if sdf_path is None:
    raise FileNotFoundError(
        "Cannot find top5_leads_3D.sdf — run PyMOL from the repo root "
        "(FOP-Code/) or from src/visualizations/."
    )

print("  Loading: %s" % os.path.abspath(sdf_path))
cmd.load(sdf_path, "sdf_raw")
cmd.split_states("sdf_raw")
cmd.delete("sdf_raw")

# split_states names objects as sdf_raw_0001 … 0005
# (or by SDF title if PyMOL uses those: KinetiDiff_Rank01 …)
rank_names = ["Rank01", "Rank02", "Rank03", "Rank04", "Rank05"]
rename_map = {}
for i, rname in enumerate(rank_names, 1):
    rename_map["sdf_raw_%04d" % i]          = rname
    rename_map["KinetiDiff_Rank%02d" % i]   = rname
    rename_map["KinetiDiff_Rank%d"   % i]   = rname

all_objs = cmd.get_names()
for old, new in rename_map.items():
    if old in all_objs:
        cmd.set_name(old, new)
        print("  Renamed %s → %s" % (old, new))

python end

# ─── 5. TRANSLATE LIGANDS TO THE VINA DOCKING CENTROID ────────────────────────
# The SDF holds RDKit 3D conformers (not docked poses). We shift each
# molecule's geometric centroid to the experimentally determined binding site.
python
CENTROID = [24.87, -12.54, 38.40]

for obj in ["Rank01", "Rank02", "Rank03", "Rank04", "Rank05"]:
    stored.coords = []
    cmd.iterate_state(1, "%s and not elem H" % obj,
                      "stored.coords.append((x, y, z))")
    if not stored.coords:
        print("  WARNING: no atoms found in %s — skipping translation" % obj)
        continue
    n  = len(stored.coords)
    cx = sum(p[0] for p in stored.coords) / n
    cy = sum(p[1] for p in stored.coords) / n
    cz = sum(p[2] for p in stored.coords) / n
    cmd.translate([CENTROID[0]-cx, CENTROID[1]-cy, CENTROID[2]-cz], obj)
    print("  Centered %s at pocket centroid" % obj)

python end

# ─── 6. LIGAND REPRESENTATIONS ─────────────────────────────────────────────────
# Ball-and-stick: rank-colored carbons, standard heteroatom element colors.

python
lig_palette = {
    "Rank01": "gold_kd",
    "Rank02": "red_kd",
    "Rank03": "grn_kd",
    "Rank04": "cyn_kd",
    "Rank05": "vio_kd",
}

def style_ligand(obj, carb_color, stick_r=0.21, sph_scale=0.26, stick_trans=0.0):
    cmd.hide("everything", obj)
    cmd.show("sticks",  obj)
    cmd.show("spheres", "%s and not elem H" % obj)
    cmd.hide("sticks",  "%s and elem H" % obj)
    cmd.hide("spheres", "%s and elem H" % obj)
    # Carbon color = rank identity
    cmd.color(carb_color, "%s and elem C" % obj)
    # Heteroatoms = standard element colors (so functional groups pop)
    cmd.color("blue",      "%s and elem N" % obj)
    cmd.color("red",       "%s and elem O" % obj)
    cmd.color("yellow",    "%s and elem S" % obj)
    cmd.color("limegreen", "%s and elem F" % obj)
    cmd.color("green",     "%s and elem Cl" % obj)
    cmd.color("maroon",    "%s and elem Br" % obj)
    cmd.set("stick_radius",      stick_r,     obj)
    cmd.set("sphere_scale",      sph_scale,   obj)
    cmd.set("stick_transparency", stick_trans, obj)

for obj, col in lig_palette.items():
    style_ligand(obj, col)

# Crystallographic reference: muted silver; slightly transparent sticks
style_ligand("ref_ligand", "sil_kd", stick_r=0.14, sph_scale=0.18, stick_trans=0.25)
cmd.color("lightblue",   "ref_ligand and elem N")
cmd.color("lightorange", "ref_ligand and elem O")
cmd.color("lightyellow", "ref_ligand and elem S")

python end

# ─── 7. POCKET ENVIRONMENT ─────────────────────────────────────────────────────
# Residues within 4.5 Å of any lead — shown as thin sticks, off-white carbons.

select all_leads, (Rank01 or Rank02 or Rank03 or Rank04 or Rank05)

select pocket_res, (receptor and polymer.protein within 4.5 of all_leads)
show sticks,  pocket_res
color pkt_c,  pocket_res and elem C
color blue,   pocket_res and elem N
color red,    pocket_res and elem O
color yellow, pocket_res and elem S
hide (pocket_res and elem H)
set stick_radius, 0.11, pocket_res

# Binding-site centroid marker — firebrick sphere
pseudoatom centroid_pt, pos=[24.87, -12.54, 38.40]
show spheres,    centroid_pt
color firebrick, centroid_pt
set sphere_scale,        0.55, centroid_pt
set sphere_transparency, 0.35, centroid_pt

deselect

# ─── 8. H-BOND / POLAR CONTACT DISTANCES ──────────────────────────────────────
# Dashed lines connect receptor N/O atoms to ligand N/O atoms within 3.5 Å.
# All shown in canary yellow; reference shown in pale gray.

python
rank_objs = ["Rank01", "Rank02", "Rank03", "Rank04", "Rank05"]

for obj in rank_objs:
    hb_name = "hb_%s" % obj.lower()
    rec_sel = ("receptor and polymer.protein and "
               "(name N or name O or name OH or name NZ or name NE or name ND1 or name ND2 "
               "or name NE2 or name NH1 or name NH2 or name OG or name OG1 or name OD1 "
               "or name OD2 or name OE1 or name OE2)")
    lig_sel = "(%s and (elem N or elem O))" % obj
    cmd.distance(hb_name, rec_sel, lig_sel, 3.5)
    cmd.color("hb_yellow", hb_name)
    cmd.set("dash_width",  2.8, hb_name)
    cmd.set("dash_radius", 0.08, hb_name)
    cmd.hide("labels", hb_name)

# Reference ligand contacts — gray dashes
cmd.distance("hb_ref",
    ("receptor and polymer.protein and "
     "(name N or name O or name OH or name NZ or name OG or name OG1)"),
    "(ref_ligand and (elem N or elem O))", 3.5)
cmd.color("gray60", "hb_ref")
cmd.set("dash_width",  1.8, "hb_ref")
cmd.set("dash_radius", 0.05, "hb_ref")
cmd.hide("labels", "hb_ref")

python end

# ─── 9. SCORE LABELS (shown in overview scene) ────────────────────────────────
python
# Metadata (from RESULTS/vina_direct/top_leads.csv)
# ★ marks best-in-class per property
meta = [
    ("Rank01", "gold_kd",
     "#1 | -11.05 kcal/mol | pKd 8.10 | +19.2% vs crystal"),
    ("Rank02", "red_kd",
     "#2 | -10.62 kcal/mol | pKd 7.79 | 2 H-bonds | SO2"),
    ("Rank03", "grn_kd",
     "#3 | -10.59 kcal/mol | QED 0.812 [BEST] | MW 311"),
    ("Rank04", "cyn_kd",
     "#4 | -10.55 kcal/mol | SA 2.26 [BEST] | MW 399"),
    ("Rank05", "vio_kd",
     "#5 | -10.51 kcal/mol | pKd 7.71 | dual-pyridine"),
]

for obj, col, text in meta:
    stored.pos = []
    cmd.iterate_state(1, "%s and not elem H" % obj,
                      "stored.pos.append((x, y, z))")
    if not stored.pos:
        continue
    n  = len(stored.pos)
    cx = sum(p[0] for p in stored.pos) / n
    # Place label above the molecule's highest atom + 2.8 Å offset
    cy = max(p[1] for p in stored.pos) + 2.8
    cz = sum(p[2] for p in stored.pos) / n
    lbl = "score_%s" % obj.lower()
    cmd.pseudoatom(lbl, pos=[cx, cy, cz])
    cmd.label(lbl, '"%s"' % text)
    cmd.color(col, lbl)
    cmd.set("sphere_scale", 0.001, lbl)   # effectively invisible anchor
    # Labels are visible when object is enabled; will be toggled per scene

python end

# Reference label
python
stored.ref_pos = []
cmd.iterate_state(1, "ref_ligand and not elem H",
                  "stored.ref_pos.append((x, y, z))")
if stored.ref_pos:
    n   = len(stored.ref_pos)
    cx  = sum(p[0] for p in stored.ref_pos) / n
    cy  = min(p[1] for p in stored.ref_pos) - 2.5   # below ref ligand
    cz  = sum(p[2] for p in stored.ref_pos) / n
    cmd.pseudoatom("score_ref", pos=[cx, cy, cz])
    cmd.label("score_ref", '"Xtal ref (A3F) | -9.27 kcal/mol | pKd 6.80"')
    cmd.color("sil_kd", "score_ref")
    cmd.set("sphere_scale", 0.001, "score_ref")
python end

# ─── 10. ORGANIZE GROUPS ───────────────────────────────────────────────────────
group Leads,      Rank01 Rank02 Rank03 Rank04 Rank05
group HBonds,     hb_rank01 hb_rank02 hb_rank03 hb_rank04 hb_rank05
group HB_Ref,     hb_ref
group Pocket,     pocket_res centroid_pt
group Reference,  ref_ligand
group Labels,     score_rank01 score_rank02 score_rank03 score_rank04 score_rank05 score_ref
group Annot,      hinge_region dfg_loop p_loop

# ─── 11. SCENES ────────────────────────────────────────────────────────────────
# Each scene sets visibility then stores camera + state.

# ── overview: all 5 leads overlaid ────────────────────────────────────────────
disable all
enable receptor
enable Leads
enable Pocket
enable HBonds
enable Reference
enable HB_Ref
enable centroid_pt
enable Labels
zoom all_leads, 20
orient all_leads
scene overview, store

# ── rank01: best binder, amber gold ───────────────────────────────────────────
disable all
enable receptor
enable Rank01
enable pocket_res
enable centroid_pt
enable hb_rank01
enable hinge_region
zoom Rank01, 12
orient Rank01
scene rank01, store

# ── rank02: SO₂-sulfonamide, 2 H-bonds ────────────────────────────────────────
disable all
enable receptor
enable Rank02
enable pocket_res
enable centroid_pt
enable hb_rank02
enable hinge_region
zoom Rank02, 12
orient Rank02
scene rank02, store

# ── rank03: highest QED, emerald green ────────────────────────────────────────
disable all
enable receptor
enable Rank03
enable pocket_res
enable centroid_pt
enable hb_rank03
enable hinge_region
zoom Rank03, 12
orient Rank03
scene rank03, store

# ── rank04: best SA score, electric cyan ──────────────────────────────────────
disable all
enable receptor
enable Rank04
enable pocket_res
enable centroid_pt
enable hb_rank04
enable hinge_region
zoom Rank04, 12
orient Rank04
scene rank04, store

# ── rank05: dual pyridine, vivid violet ───────────────────────────────────────
disable all
enable receptor
enable Rank05
enable pocket_res
enable centroid_pt
enable hb_rank05
enable hinge_region
zoom Rank05, 12
orient Rank05
scene rank05, store

# ── ref_compare: Rank 1 (gold) vs crystallographic reference (silver) ─────────
disable all
enable receptor
enable Rank01
enable ref_ligand
enable pocket_res
enable centroid_pt
enable hb_rank01
enable hb_ref
enable score_rank01
enable score_ref
enable hinge_region
zoom (Rank01 or ref_ligand), 14
orient (Rank01 or ref_ligand)
scene ref_compare, store

# ── pocket_shape: receptor surface-first, emphasize pocket geometry ────────────
disable all
enable receptor
set transparency, 0.35, receptor    # more opaque to show pocket shape
enable Leads
enable centroid_pt
enable Pocket
zoom all_leads, 24
scene pocket_shape, store
# Restore transparency for other scenes
set transparency, 0.68, receptor

# ── Land on overview for startup ──────────────────────────────────────────────
scene overview, recall

# ─── 12. LEGEND ────────────────────────────────────────────────────────────────
python
print("")
print("╔═══════════════════════════════════════════════════════════════════════╗")
print("║   KinetiDiff  ·  ACVR1/ALK2 (3MTF)  ·  Top-5 Vina-Direct Leads    ║")
print("╠═══════════════════════════════════════════════════════════════════════╣")
print("║  COLOR          OBJECT    VINA(kcal/mol)  pKd    QED    SA    MW    ║")
print("║  ─────────────────────────────────────────────────────────────────  ║")
print("║  AMBER GOLD     Rank01        -11.05      8.10  0.618  3.79  404   ║")
print("║  TOMATO RED     Rank02        -10.62      7.79  0.389  2.61  476   ║")
print("║  EMERALD GRN    Rank03        -10.59      7.77  0.812* 3.12  311   ║")
print("║  ELECTRIC CYN   Rank04        -10.55      7.74  0.524  2.26* 399   ║")
print("║  VIVID VIOLET   Rank05        -10.51      7.71  0.759  3.75  383   ║")
print("║  SILVER         Xtal ref       -9.27      6.80  0.680  3.34  371   ║")
print("║  ─────────────────────────────────────────────────────────────────  ║")
print("║  * = best in class.  All leads pass Lipinski Ro5 (except Rank02).  ║")
print("║  Red sphere = Vina centroid (24.87, -12.54, 38.40).                ║")
print("║  Yellow dashes = polar contacts <= 3.5 A (N/O proximity).          ║")
print("╠═══════════════════════════════════════════════════════════════════════╣")
print("║  SCENES  (PgUp / PgDn or Scene Panel):                              ║")
print("║    overview  rank01  rank02  rank03  rank04  rank05                 ║")
print("║    ref_compare  pocket_shape                                         ║")
print("╠═══════════════════════════════════════════════════════════════════════╣")
print("║  To render panel PNGs:                                               ║")
print("║    @src/visualizations/kinetidiff_render.pml                         ║")
print("╚═══════════════════════════════════════════════════════════════════════╝")
python end

# Save session (can be reloaded without re-fetching 3MTF)
save kinetidiff_all.pse
python
print("  Session saved → kinetidiff_all.pse")
python end
