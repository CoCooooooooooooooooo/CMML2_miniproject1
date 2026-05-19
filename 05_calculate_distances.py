"""
05_calculate_distances.py
=========================
Calculates the 3D Euclidean distance between:
  - the catalytic Cysteine Sγ (SG) atom of each ALDH isoenzyme, and
  - the aldehyde carbon (C=O) of each docked small molecule.

Catalytic Cys residue numbers are read from Catalytic_Cys_positions.xlsx
(provided by the instructor). Proteins with no annotated Cys (marked "/")
are skipped — binding assessment for those relies on ΔG only.

The aldehyde carbon is identified from the rank1.sdf file as the carbon atom
closest to a terminal oxygen (C=O motif). For straight-chain aliphatic
aldehydes this is unambiguous (C1 of the chain).

Output: distance_results.csv
Columns: complex_name, protein, ligand, cys_residue, distance_A, within_5A

Usage:
    conda activate diffdock
    python 05_calculate_distances.py

Prerequisites:
    pip install pandas openpyxl numpy biopython
"""

import os
import re
import glob
import numpy as np
import pandas as pd

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
RESULTS_DIR   = "/mnt/Coco/diffdock_results"
PROTEIN_DIR   = "/mnt/Coco/A15-B2"
CYS_TABLE     = "Catalytic_Cys_positions.xlsx"   # instructor-provided spreadsheet
OUTPUT_CSV    = "/mnt/Coco/distance_results.csv"
DISTANCE_THRESHOLD = 5.0                          # Å; threshold for "productive binding"
# ──────────────────────────────────────────────────────────────────────────────


# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────

def load_cys_table(path: str) -> dict:
    """
    Read catalytic Cys positions from Excel.
    Returns {protein_label: residue_number}, e.g. {"A15": 330, "A16": None, ...}
    Proteins with "/" in the Cys column are stored as None (no distance analysis).
    """
    df = pd.read_excel(path, header=0)
    # Expected columns: protein label (col 0), Cys annotation (col 1)
    cys_map = {}
    for _, row in df.iterrows():
        vals = [str(v).strip() for v in row.values if str(v).strip() not in ("nan", "")]
        if len(vals) < 2:
            continue
        label = vals[0]   # e.g. "A15"
        annot = vals[1]   # e.g. "Cys330" or "/"
        if annot == "/":
            cys_map[label] = None
        else:
            m = re.search(r"(\d+)", annot)
            cys_map[label] = int(m.group(1)) if m else None
    return cys_map


def parse_pdb_atoms(pdb_path: str) -> list:
    """
    Parse ATOM records from a PDB file.
    Returns list of dicts: {name, resname, resseq, x, y, z}
    """
    atoms = []
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            try:
                atoms.append({
                    "name":   line[12:16].strip(),
                    "resname": line[17:20].strip(),
                    "resseq": int(line[22:26]),
                    "x": float(line[30:38]),
                    "y": float(line[38:46]),
                    "z": float(line[46:54]),
                })
            except (ValueError, IndexError):
                continue
    return atoms


def find_aldehyde_carbon_sdf(sdf_path: str) -> np.ndarray:
    """
    Locate the aldehyde carbon (C=O) in an SDF file.

    Strategy:
      1. Read all heavy atoms and build a simple distance-based bond graph.
      2. A carbon is the aldehyde carbon if it has exactly one oxygen neighbour
         within 1.6 Å and that oxygen has no other heavy-atom neighbours.
      3. Tie-break: if multiple candidates, pick the one with the most oxygens.

    Returns the (x, y, z) numpy array of the aldehyde carbon, or None.
    """
    with open(sdf_path) as fh:
        lines = fh.readlines()

    try:
        n_atoms = int(lines[3][:3].strip())
    except (IndexError, ValueError):
        return None

    atoms = []
    for i in range(n_atoms):
        line = lines[4 + i]
        try:
            x = float(line[0:10])
            y = float(line[10:20])
            z = float(line[20:30])
            elem = line[31:34].strip().upper()
            atoms.append({"elem": elem, "xyz": np.array([x, y, z])})
        except (ValueError, IndexError):
            continue

    if not atoms:
        return None

    coords = np.array([a["xyz"] for a in atoms])

    # Build distance matrix
    diff = coords[:, None, :] - coords[None, :, :]
    dmat = np.sqrt((diff ** 2).sum(axis=2))

    BOND = 1.6  # covalent bond cutoff (Å)

    # Find aldehyde carbon candidates
    candidates = []
    for i, atom in enumerate(atoms):
        if atom["elem"] != "C":
            continue
        # oxygen neighbours of this carbon
        o_neighbours = [
            j for j, other in enumerate(atoms)
            if other["elem"] == "O" and 0 < dmat[i, j] < BOND
        ]
        if len(o_neighbours) != 1:
            continue
        o_idx = o_neighbours[0]
        # that oxygen must have no other heavy-atom neighbours (terminal =O)
        o_heavy_neighbours = [
            j for j, other in enumerate(atoms)
            if other["elem"] not in ("H",) and 0 < dmat[o_idx, j] < BOND and j != i
        ]
        if len(o_heavy_neighbours) == 0:
            candidates.append(i)

    if not candidates:
        # Fallback: return the carbon closest to any oxygen
        carbon_idx = [i for i, a in enumerate(atoms) if a["elem"] == "C"]
        oxygen_idx = [i for i, a in enumerate(atoms) if a["elem"] == "O"]
        if not carbon_idx or not oxygen_idx:
            return None
        min_d = np.inf
        best  = carbon_idx[0]
        for ci in carbon_idx:
            for oi in oxygen_idx:
                if dmat[ci, oi] < min_d:
                    min_d = dmat[ci, oi]
                    best  = ci
        return atoms[best]["xyz"]

    return atoms[candidates[0]]["xyz"]


def get_cys_sg(protein_atoms: list, resseq: int) -> np.ndarray:
    """Return the SG (sulphur) coordinates of Cysteine at residue number resseq."""
    for atom in protein_atoms:
        if atom["resseq"] == resseq and atom["name"] == "SG":
            return np.array([atom["x"], atom["y"], atom["z"]])
    return None


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    cys_map = load_cys_table(CYS_TABLE)
    print(f"Loaded Cys table: {cys_map}\n")

    complex_dirs = sorted([
        d for d in glob.glob(os.path.join(RESULTS_DIR, "*"))
        if os.path.isdir(d)
    ])
    print(f"Found {len(complex_dirs)} complex directories.\n")

    rows = []
    for idx, complex_dir in enumerate(complex_dirs):
        complex_name = os.path.basename(complex_dir)
        parts = complex_name.split("_", 1)
        protein_label = parts[0]              # e.g. "A15"
        ligand_name   = parts[1] if len(parts) > 1 else "unknown"

        # Check if this protein has a catalogued Cys
        if protein_label not in cys_map:
            print(f"[{idx+1}] SKIP {complex_name}: protein not in Cys table")
            rows.append({
                "complex_name": complex_name,
                "protein": protein_label,
                "ligand": ligand_name,
                "cys_residue": "N/A",
                "distance_A": None,
                "within_5A": None,
                "status": "not_in_table",
            })
            continue

        cys_resnum = cys_map[protein_label]
        if cys_resnum is None:
            # No conserved catalytic Cys (e.g. ADH1A, ADH1B, ALDH16A1)
            print(f"[{idx+1}] SKIP {complex_name}: no catalytic Cys (ΔG-only protein)")
            rows.append({
                "complex_name": complex_name,
                "protein": protein_label,
                "ligand": ligand_name,
                "cys_residue": "/",
                "distance_A": None,
                "within_5A": None,
                "status": "no_cys",
            })
            continue

        # Load protein atoms
        protein_pdb = os.path.join(PROTEIN_DIR, f"{protein_label}.pdb")
        if not os.path.exists(protein_pdb):
            print(f"[{idx+1}] SKIP {complex_name}: protein PDB not found")
            rows.append({
                "complex_name": complex_name, "protein": protein_label,
                "ligand": ligand_name, "cys_residue": cys_resnum,
                "distance_A": None, "within_5A": None, "status": "no_protein_pdb",
            })
            continue

        protein_atoms = parse_pdb_atoms(protein_pdb)
        cys_sg = get_cys_sg(protein_atoms, cys_resnum)
        if cys_sg is None:
            print(f"[{idx+1}] WARN {complex_name}: Cys{cys_resnum} SG atom not found in PDB")
            rows.append({
                "complex_name": complex_name, "protein": protein_label,
                "ligand": ligand_name, "cys_residue": cys_resnum,
                "distance_A": None, "within_5A": None, "status": "cys_sg_missing",
            })
            continue

        # Load ligand and find aldehyde carbon
        sdf_files = glob.glob(os.path.join(complex_dir, "rank1*.sdf"))
        if not sdf_files:
            print(f"[{idx+1}] SKIP {complex_name}: no rank1 sdf")
            rows.append({
                "complex_name": complex_name, "protein": protein_label,
                "ligand": ligand_name, "cys_residue": cys_resnum,
                "distance_A": None, "within_5A": None, "status": "no_sdf",
            })
            continue

        ald_c = find_aldehyde_carbon_sdf(sdf_files[0])
        if ald_c is None:
            print(f"[{idx+1}] WARN {complex_name}: aldehyde carbon not identified")
            rows.append({
                "complex_name": complex_name, "protein": protein_label,
                "ligand": ligand_name, "cys_residue": cys_resnum,
                "distance_A": None, "within_5A": None, "status": "no_aldehyde_c",
            })
            continue

        # Compute Euclidean distance
        dist = float(np.linalg.norm(cys_sg - ald_c))
        within = dist <= DISTANCE_THRESHOLD
        print(f"[{idx+1}] {complex_name}: Cys{cys_resnum} SG — aldehyde C = {dist:.2f} Å  {'✓' if within else '✗'}")
        rows.append({
            "complex_name": complex_name,
            "protein": protein_label,
            "ligand": ligand_name,
            "cys_residue": f"Cys{cys_resnum}",
            "distance_A": round(dist, 3),
            "within_5A": within,
            "status": "OK",
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    ok = (df["status"] == "OK").sum()
    within = df["within_5A"].sum() if "within_5A" in df.columns else 0
    print(f"\nFinished. {ok} distances computed; {within} within {DISTANCE_THRESHOLD} Å.")
    print(f"Results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
