"""
06_extension_analysis.py
========================
Extension analysis: for straight-chain aliphatic aldehydes, integrates three
quantitative descriptors to dissect ALDH substrate selectivity:

  1. Carbon chain length  (C1 = formaldehyde, C2 = acetaldehyde, …, C16 = hexadecanal)
  2. Binding free energy  (ΔG, kcal/mol) from PRODIGY
  3. Atomic contact number (heavy-atom contacts ≤ 4.5 Å between ligand and protein)

Outputs:
  - extension_results.csv    : per-complex table of the three descriptors
  - extension_correlation.csv: Pearson and Spearman correlations (ΔG & contacts vs chain length)
  - Supplementary figures    : scatter plots saved as PDF

Usage:
    conda activate diffdock
    python 06_extension_analysis.py

Prerequisites:
    pip install pandas openpyxl numpy scipy matplotlib biopython
"""

import os
import re
import glob
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")   # headless rendering for server
import matplotlib.pyplot as plt

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
RESULTS_DIR     = "/mnt/Coco/diffdock_results"
PROTEIN_DIR     = "/mnt/Coco/A15-B2"
PRODIGY_CSV     = "/mnt/Coco/prodigy_results.csv"
OUT_TABLE       = "/mnt/Coco/extension_results.csv"
OUT_CORR        = "/mnt/Coco/extension_correlation.csv"
OUT_FIGURE_DIR  = "/mnt/Coco/figures"
CONTACT_CUTOFF  = 4.5   # Å; heavy-atom contact threshold
# ──────────────────────────────────────────────────────────────────────────────

# Straight-chain aliphatic aldehydes and their carbon chain lengths
# Key = ligand filename stem (must match your .sdf naming convention)
ALIPHATIC_CHAIN_MAP = {
    "formaldehyde":      1,
    "acetaldehyde":      2,
    "propanal":          3,
    "butyraldehyde":     4,
    "pentanal":          5,
    "hexanal":           6,
    "heptanal":          7,
    "octanal":           8,
    "nonanal":           9,
    "decanal":           10,
    "undecanal":         11,
    "dodecanal":         12,
    "tridecanal":        13,
    "tetradecanal":      14,
    "pentadecanal":      15,
    "hexadecanal":       16,
}

os.makedirs(OUT_FIGURE_DIR, exist_ok=True)


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def parse_pdb_protein_atoms(pdb_path: str) -> np.ndarray:
    """Return Nx3 array of heavy-atom coordinates from ATOM records."""
    coords = []
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            elem = line[76:78].strip() if len(line) > 76 else ""
            if elem == "H":
                continue
            try:
                coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            except ValueError:
                continue
    return np.array(coords) if coords else np.empty((0, 3))


def parse_sdf_ligand_atoms(sdf_path: str) -> np.ndarray:
    """Return Nx3 array of heavy-atom coordinates from SDF file."""
    with open(sdf_path) as fh:
        lines = fh.readlines()
    try:
        n_atoms = int(lines[3][:3].strip())
    except (IndexError, ValueError):
        return np.empty((0, 3))
    coords = []
    for i in range(n_atoms):
        line = lines[4 + i]
        try:
            elem = line[31:34].strip().upper()
            if elem == "H":
                continue
            coords.append([float(line[0:10]), float(line[10:20]), float(line[20:30])])
        except (ValueError, IndexError):
            continue
    return np.array(coords) if coords else np.empty((0, 3))


def count_contacts(protein_coords: np.ndarray, ligand_coords: np.ndarray,
                   cutoff: float = 4.5) -> int:
    """Count protein–ligand heavy-atom pairs within cutoff Å."""
    if protein_coords.size == 0 or ligand_coords.size == 0:
        return 0
    diff = protein_coords[:, None, :] - ligand_coords[None, :, :]
    dmat = np.sqrt((diff ** 2).sum(axis=2))
    return int((dmat < cutoff).sum())


def normalise_ligand_name(raw: str) -> str:
    """Strip numeric prefix and suffix added by generate_csv (e.g. '1_formaldehyde' → 'formaldehyde')."""
    # complex_name format: A15_1_formaldehyde  or  A15_formaldehyde
    # After splitting on '_' and dropping protein prefix, join remaining parts
    tokens = raw.split("_")
    # Drop leading numeric token if present
    cleaned = "_".join(t for t in tokens if not t.isdigit())
    return cleaned.lower()


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    # Load PRODIGY ΔG values
    dg_df = pd.read_csv(PRODIGY_CSV)
    dg_lookup = {row["complex_name"]: row["dG_kcal_mol"] for _, row in dg_df.iterrows()}

    complex_dirs = sorted([
        d for d in glob.glob(os.path.join(RESULTS_DIR, "*"))
        if os.path.isdir(d)
    ])

    rows = []
    for complex_dir in complex_dirs:
        complex_name  = os.path.basename(complex_dir)
        parts         = complex_name.split("_", 1)
        protein_label = parts[0]
        raw_ligand    = parts[1] if len(parts) > 1 else ""
        ligand_name   = normalise_ligand_name(raw_ligand)

        # Only process straight-chain aliphatic aldehydes
        if ligand_name not in ALIPHATIC_CHAIN_MAP:
            continue

        chain_length = ALIPHATIC_CHAIN_MAP[ligand_name]
        dg           = dg_lookup.get(complex_name)

        # Compute contact number
        protein_pdb = os.path.join(PROTEIN_DIR, f"{protein_label}.pdb")
        sdf_files   = glob.glob(os.path.join(complex_dir, "rank1*.sdf"))

        contacts = None
        if os.path.exists(protein_pdb) and sdf_files:
            prot_coords = parse_pdb_protein_atoms(protein_pdb)
            lig_coords  = parse_sdf_ligand_atoms(sdf_files[0])
            contacts    = count_contacts(prot_coords, lig_coords, CONTACT_CUTOFF)

        rows.append({
            "complex_name":  complex_name,
            "protein":       protein_label,
            "ligand":        ligand_name,
            "chain_length":  chain_length,
            "dG_kcal_mol":   dg,
            "contact_number": contacts,
        })
        print(f"{complex_name}: chain={chain_length}, ΔG={dg}, contacts={contacts}")

    df = pd.DataFrame(rows).dropna(subset=["dG_kcal_mol", "contact_number"])
    df.to_csv(OUT_TABLE, index=False)
    print(f"\nExtension table saved to: {OUT_TABLE}  ({len(df)} rows)")

    # ─── Correlation analysis ───────────────────────────────────────────────
    corr_rows = []
    for protein, grp in df.groupby("protein"):
        if len(grp) < 3:
            continue
        x  = grp["chain_length"].values
        y1 = grp["dG_kcal_mol"].values
        y2 = grp["contact_number"].values

        r_dg, p_dg       = stats.pearsonr(x, y1)
        rho_dg, pp_dg    = stats.spearmanr(x, y1)
        r_ct, p_ct       = stats.pearsonr(x, y2)
        rho_ct, pp_ct    = stats.spearmanr(x, y2)

        corr_rows.append({
            "protein": protein,
            "n": len(grp),
            "Pearson_dG_vs_chainlen":    round(r_dg,  3),  "p_pearson_dG":  round(p_dg,  4),
            "Spearman_dG_vs_chainlen":   round(rho_dg,3),  "p_spearman_dG": round(pp_dg, 4),
            "Pearson_contacts_vs_chainlen":  round(r_ct,  3), "p_pearson_ct":  round(p_ct,  4),
            "Spearman_contacts_vs_chainlen": round(rho_ct, 3), "p_spearman_ct": round(pp_ct, 4),
        })

    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(OUT_CORR, index=False)
    print(f"Correlation table saved to: {OUT_CORR}")
    print(corr_df.to_string(index=False))

    # ─── Figures ───────────────────────────────────────────────────────────
    proteins = df["protein"].unique()
    colors   = plt.cm.tab10(np.linspace(0, 1, len(proteins)))

    # Figure A: ΔG vs chain length
    fig, ax = plt.subplots(figsize=(7, 5))
    for prot, col in zip(proteins, colors):
        sub = df[df["protein"] == prot].sort_values("chain_length")
        ax.plot(sub["chain_length"], sub["dG_kcal_mol"], "o-", color=col, label=prot, alpha=0.8)
    ax.set_xlabel("Carbon chain length", fontsize=11)
    ax.set_ylabel("ΔG (kcal/mol)", fontsize=11)
    ax.set_title("Binding free energy vs aldehyde chain length", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.invert_yaxis()   # more negative = stronger binding, conventionally at top
    plt.tight_layout()
    out_a = os.path.join(OUT_FIGURE_DIR, "FigS_dG_vs_chainlength.pdf")
    fig.savefig(out_a, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out_a}")

    # Figure B: Contact number vs chain length
    fig, ax = plt.subplots(figsize=(7, 5))
    for prot, col in zip(proteins, colors):
        sub = df[df["protein"] == prot].sort_values("chain_length")
        ax.plot(sub["chain_length"], sub["contact_number"], "s-", color=col, label=prot, alpha=0.8)
    ax.set_xlabel("Carbon chain length", fontsize=11)
    ax.set_ylabel(f"Atomic contact number (≤{CONTACT_CUTOFF} Å)", fontsize=11)
    ax.set_title("Atomic contact number vs aldehyde chain length", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    out_b = os.path.join(OUT_FIGURE_DIR, "FigS_contacts_vs_chainlength.pdf")
    fig.savefig(out_b, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out_b}")

    print("\nExtension analysis complete.")


if __name__ == "__main__":
    main()
