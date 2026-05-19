"""
04_batch_prodigy.py
===================
For every DiffDock output complex, merges the receptor (complex.pdb) and
docked ligand (rank1*.sdf) into a temporary combined PDB, then calls
prodigy_lig to compute the binding free energy (ΔG, kcal/mol).

Results are written to prodigy_results.csv.

Directory layout expected:
    /mnt/Coco/diffdock_results/
        A15_formaldehyde/
            complex.pdb          <- receptor (written by DiffDock)
            rank1_confidence*.sdf  <- best docked pose
        A15_acetaldehyde/
            ...

Prerequisites:
    pip install pandas numpy biopython
    pip install prodigy-lig           # or: conda install -c conda-forge prodigy-lig

Usage:
    conda activate diffdock
    python 04_batch_prodigy.py
"""

import os
import re
import glob
import subprocess
import pandas as pd
from typing import Optional

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
RESULTS_DIR   = "/mnt/Coco/diffdock_results"
PROTEIN_DIR   = "/mnt/Coco/A15-B2"
OUTPUT_CSV    = "/mnt/Coco/prodigy_results.csv"
PRODIGY_CMD   = "prodigy_lig"   # must be on PATH; adjust if installed elsewhere
PRODIGY_CHAIN = "A"             # receptor chain label in merged PDB
LIGAND_CHAIN  = "B"             # ligand chain label
LIGAND_RESN   = "LIG"           # ligand residue name used when merging
# ──────────────────────────────────────────────────────────────────────────────


def sdf_to_hetatm_lines(sdf_path: str, chain: str = "B", resname: str = "LIG") -> list:
    """
    Minimal SDF → HETATM PDB-line converter.
    Reads only the atom block (lines between counts line and first 'M  END').
    Returns a list of HETATM record strings.
    """
    hetatm_lines = []
    with open(sdf_path) as fh:
        lines = fh.readlines()

    # The 4th line (index 3) is the counts line: aaabbblllfffcccsssxxxrrrpppiiimmmvvvvvv
    try:
        counts_line = lines[3]
        n_atoms = int(counts_line[:3].strip())
    except (IndexError, ValueError):
        return []

    atom_start = 4
    for i in range(n_atoms):
        line = lines[atom_start + i]
        try:
            x       = float(line[0:10])
            y       = float(line[10:20])
            z       = float(line[20:30])
            element = line[31:34].strip()
        except (ValueError, IndexError):
            continue

        atom_serial = i + 1
        atom_name   = f"{element}{atom_serial}" if len(element) == 1 else element
        hetatm_lines.append(
            f"HETATM{atom_serial:5d}  {atom_name:<4s}{resname:>3s} {chain}{atom_serial:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}  \n"
        )
    return hetatm_lines


def merge_complex(protein_pdb: str, sdf_path: str, out_pdb: str) -> bool:
    """Merge receptor PDB and ligand SDF into a single PDB for PRODIGY."""
    try:
        with open(protein_pdb) as fh:
            protein_lines = [l for l in fh if l.startswith(("ATOM", "HETATM", "TER"))]
        lig_lines = sdf_to_hetatm_lines(sdf_path)
        if not lig_lines:
            return False
        with open(out_pdb, "w") as fh:
            fh.writelines(protein_lines)
            fh.write("TER\n")
            fh.writelines(lig_lines)
            fh.write("TER\nEND\n")
        return True
    except Exception as e:
        print(f"    merge error: {e}")
        return False


def run_prodigy(pdb_path: str) -> Optional[float]:
    """
    Run prodigy_lig and parse ΔG from its output.
    Returns ΔG as float (kcal/mol), or None on failure.
    """
    cmd = [PRODIGY_CMD, "-c", PRODIGY_CHAIN, f"{LIGAND_CHAIN}:{LIGAND_RESN}", "-i", pdb_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        output = result.stdout + result.stderr
        # PRODIGY output line looks like:  "A15_formaldehyde  -6.34"
        # Parse the first float in range (-30, 5)
        for line in output.splitlines():
            for token in line.split():
                try:
                    val = float(token)
                    if -30 < val < 5:
                        return val
                except ValueError:
                    continue
    except subprocess.TimeoutExpired:
        print("    PRODIGY timed out")
    except FileNotFoundError:
        raise RuntimeError(
            f"'{PRODIGY_CMD}' not found. Install with: pip install prodigy-lig"
        )
    return None


def main():
    complex_dirs = sorted([
        d for d in glob.glob(os.path.join(RESULTS_DIR, "*"))
        if os.path.isdir(d)
    ])
    print(f"Found {len(complex_dirs)} complex directories.\n")

    rows = []
    for idx, complex_dir in enumerate(complex_dirs):
        complex_name = os.path.basename(complex_dir)

        # Find rank1 SDF
        sdf_files = glob.glob(os.path.join(complex_dir, "rank1*.sdf"))
        if not sdf_files:
            print(f"[{idx+1}/{len(complex_dirs)}] SKIP {complex_name}: no rank1 sdf")
            rows.append({"complex_name": complex_name, "dG_kcal_mol": None, "status": "no_sdf"})
            continue

        sdf_path = sdf_files[0]

        # Resolve protein PDB from prefix (e.g. "A15_formaldehyde" → "A15.pdb")
        protein_stem = complex_name.split("_")[0]
        protein_pdb  = os.path.join(PROTEIN_DIR, f"{protein_stem}.pdb")
        if not os.path.exists(protein_pdb):
            print(f"[{idx+1}/{len(complex_dirs)}] SKIP {complex_name}: protein not found")
            rows.append({"complex_name": complex_name, "dG_kcal_mol": None, "status": "no_protein"})
            continue

        # Merge into temporary complex PDB
        tmp_pdb = os.path.join(complex_dir, "complex_merged.pdb")
        if not merge_complex(protein_pdb, sdf_path, tmp_pdb):
            print(f"[{idx+1}/{len(complex_dirs)}] FAIL {complex_name}: merge failed")
            rows.append({"complex_name": complex_name, "dG_kcal_mol": None, "status": "merge_failed"})
            continue

        # Run PRODIGY
        dg = run_prodigy(tmp_pdb)
        status = "OK" if dg is not None else "prodigy_failed"
        print(f"[{idx+1}/{len(complex_dirs)}] {complex_name}: ΔG = {dg}")
        rows.append({"complex_name": complex_name, "dG_kcal_mol": dg, "status": status})

    # Save results
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    ok = (df["status"] == "OK").sum()
    print(f"\nFinished. {ok}/{len(rows)} successful.")
    print(f"Results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
