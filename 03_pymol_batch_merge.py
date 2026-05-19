"""
03_pymol_batch_merge.py
=======================
Uses PyMOL in command-line (headless) mode to combine each protein .pdb file
with its corresponding docked ligand .sdf file into a single complex PDB.

NOTE: In this project, DiffDock outputs a separate complex.pdb (receptor) and
rank1.sdf (ligand) for each run. The PRODIGY and distance scripts below read
these two files directly without merging. This script is therefore provided for
reference and visualisation purposes only; it was NOT used in the main pipeline.

Usage (requires PyMOL with Python API installed):
    pymol -c 03_pymol_batch_merge.py

Output:
    /mnt/Coco/merged_structures/<complex_name>_merged.pdb
    One file per successfully merged complex.
"""

import os
import glob

try:
    from pymol import cmd
except ImportError:
    raise ImportError(
        "This script must be run via PyMOL: pymol -c 03_pymol_batch_merge.py\n"
        "Alternatively, activate a conda env that includes PyMOL."
    )

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
RESULTS_DIR  = "/mnt/Coco/diffdock_results"   # DiffDock output folder
PROTEIN_DIR  = "/mnt/Coco/A15-B2"             # original protein .pdb files
OUTPUT_DIR   = "/mnt/Coco/merged_structures"  # where merged files are saved
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

complex_dirs = sorted([
    d for d in glob.glob(os.path.join(RESULTS_DIR, "*"))
    if os.path.isdir(d)
])

print(f"Found {len(complex_dirs)} complex directories.")
success, skip, fail = 0, 0, 0

for complex_dir in complex_dirs:
    complex_name = os.path.basename(complex_dir)

    # Locate rank1 SDF
    sdf_files = glob.glob(os.path.join(complex_dir, "rank1*.sdf"))
    if not sdf_files:
        print(f"  SKIP {complex_name}: no rank1 sdf found")
        skip += 1
        continue
    sdf_path = sdf_files[0]

    # Locate matching protein PDB (name prefix before first underscore)
    protein_stem = complex_name.split("_")[0]          # e.g. "A15"
    protein_pdb  = os.path.join(PROTEIN_DIR, f"{protein_stem}.pdb")
    if not os.path.exists(protein_pdb):
        print(f"  SKIP {complex_name}: protein {protein_pdb} not found")
        skip += 1
        continue

    out_pdb = os.path.join(OUTPUT_DIR, f"{complex_name}_merged.pdb")

    try:
        cmd.reinitialize()
        cmd.load(protein_pdb, "protein")   # load receptor
        cmd.load(sdf_path,    "ligand")    # load docked ligand

        # Rename ligand residue to LIG for compatibility with PRODIGY
        cmd.alter("ligand", "resn='LIG'")
        cmd.alter("ligand", "chain='B'")
        cmd.alter("protein", "chain='A'")

        # Save combined structure
        cmd.save(out_pdb, "protein or ligand")
        print(f"  OK  {complex_name} -> {os.path.basename(out_pdb)}")
        success += 1
    except Exception as e:
        print(f"  FAIL {complex_name}: {e}")
        fail += 1

print(f"\nDone. Success: {success} | Skipped: {skip} | Failed: {fail}")
print(f"Merged files saved to: {OUTPUT_DIR}")
