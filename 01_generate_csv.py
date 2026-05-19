"""
01_generate_csv.py
==================
Automatically scans protein (.pdb) and ligand (.sdf) directories and writes a
tasks.csv file in the format required by DiffDock's batch inference pipeline.

Directory layout expected:
    /mnt/Coco/A15-B2/          <- one .pdb file per ALDH isoenzyme
        A15.pdb
        A16.pdb
        ...
        B2.pdb
    /mnt/Coco/substrate/       <- one .sdf file per aldehyde substrate
        formaldehyde.sdf
        acetaldehyde.sdf
        ...

Output:
    /mnt/Coco/tasks.csv        <- DiffDock-compatible batch CSV

Usage (on the GPU server):
    conda activate diffdock
    python 01_generate_csv.py
"""

import os
import csv

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
PROTEIN_DIR = "/mnt/Coco/A15-B2"      # folder containing .pdb files
LIGAND_DIR  = "/mnt/Coco/substrate"   # folder containing .sdf files
OUTPUT_CSV  = "/mnt/Coco/tasks.csv"   # output path
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # Collect and sort file paths
    proteins = sorted([
        os.path.join(PROTEIN_DIR, f)
        for f in os.listdir(PROTEIN_DIR)
        if f.endswith(".pdb")
    ])
    ligands = sorted([
        os.path.join(LIGAND_DIR, f)
        for f in os.listdir(LIGAND_DIR)
        if f.endswith(".sdf")
    ])

    if not proteins:
        raise FileNotFoundError(f"No .pdb files found in {PROTEIN_DIR}")
    if not ligands:
        raise FileNotFoundError(f"No .sdf files found in {LIGAND_DIR}")

    print(f"Found {len(proteins)} protein(s): {[os.path.basename(p) for p in proteins]}")
    print(f"Found {len(ligands)} ligand(s)")
    print(f"Total pairs: {len(proteins)} × {len(ligands)} = {len(proteins)*len(ligands)}")

    # Write CSV with the 4 columns required by DiffDock inference.py
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["complex_name", "protein_path", "ligand_description", "protein_sequence"])
        for prot in proteins:
            prot_stem = os.path.splitext(os.path.basename(prot))[0]   # e.g. "A15"
            for lig in ligands:
                lig_stem  = os.path.splitext(os.path.basename(lig))[0] # e.g. "formaldehyde"
                complex_name = f"{prot_stem}_{lig_stem}"               # e.g. "A15_formaldehyde"
                writer.writerow([complex_name, prot, lig, ""])         # protein_sequence left blank

    print(f"\nCSV written to: {OUTPUT_CSV}")
    print("Columns: complex_name | protein_path | ligand_description | protein_sequence")
    print("Ready for DiffDock batch inference.")

if __name__ == "__main__":
    main()
