"""
contact_analysis.py
====================
Calculates the number of atom contacts within 5 Å for each protein–ligand complex.
This serves as a third interaction strength metric alongside ΔG and the Cys–aldehyde distance.

Input files (same directory):
    prodigy_results.csv       — contains complex_name, dG_kcal_mol
    cys_aldehyde_distances.csv — contains complex, protein, distance_A

DiffDock output directory structure (server):
    /mnt/Coco/diffdock_results/
        A15-new-ALDH7A1_1_formaldehyde/
            complex.pdb      ← protein structure
            rank1.sdf        ← top-ranked docked ligand

Usage:
    python contact_analysis.py

Dependencies:
    pip install biopython pandas numpy

Outputs:
    contact_results.csv       — contact statistics per complex
    Fig_contact_vs_chain.png  — contact count vs. carbon chain length
    Fig_three_metrics.png     — correlation scatter matrix for the three metrics
    Fig_contact_type_stack.png — stacked bar chart by contact type
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ─── Configuration (edit as needed) ───────────────────────────────────────────────────────────

RESULTS_DIR  = "D:/CMML_ICA2/Data/diffdock_results"   # root DiffDock output directory
PRODIGY_CSV  = "prodigy_results.csv"           # PRODIGY results file
DISTANCE_CSV = "cys_aldehyde_distances.csv"    # aldehyde distance results file
OUTPUT_CSV   = "contact_results.csv"           # script output file
CUTOFF       = 5.0                             # contact distance cutoff (Å)

# Only process these 7 proteins
TARGET_PROTEINS = {"A15", "A16", "A17", "A18", "A19", "B1", "B2"}

# Folder name prefix → protein identifier
PREFIX_MAP = {
    "A15-new-ALDH7A1_": "A15",
    "A15_":              "A15",
    "A16_":              "A16",
    "A17-new-ALDH9A1_": "A17",
    "A17_":              "A17",
    "A18_":              "A18",
    "A19_":              "A19",
    "B1_":               "B1",
    "B2_":               "B2",
}

# Straight-chain aliphatic aldehydes: name → carbon count
ALIPHATIC_MAP = {
    "formaldehyde":   1,
    "acetaldehyde":   2,
    "propylaldehyde": 3,
    "butyraldehyde":  4,
    "amylaldehyde":   5,
    "hexanal":        6,
    "heptanal":       7,
    "octanaldehyde":  8,
    "nonanal":        9,
    "decanal":        10,
    "dodecanal":      12,
    "Tetradecanal":   14,
    "hexadecanal":    16,
}

# ─── Utility functions ─────────────────────────────────────────────────────────────────

def get_protein_id(folder_name):
    """Extract the protein identifier from a folder name, e.g. 'A15-new-ALDH7A1_6_hexanal' → 'A15'"""
    for prefix, prot in PREFIX_MAP.items():
        if folder_name.startswith(prefix):
            return prot
    # fallback: use the first alphanumeric segment
    m = re.match(r"^(A1[5-9]|B[12])", folder_name)
    return m.group(1) if m else None


def get_substrate(folder_name):
    """Extract the substrate name from a folder name, e.g. 'A15-new-ALDH7A1_6_hexanal' → 'hexanal'"""
    m = re.search(r"_\d+_(.+)$", folder_name)
    return m.group(1) if m else None


def parse_pdb_atoms(pdb_path):
    """
    Parse ATOM records from a PDB file (protein) and return a list of atoms.
    Each atom contains {"name", "resname", "resseq", "element", "coord": np.array([x,y,z])}
    """
    atoms = []
    with open(pdb_path, "r") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            try:
                name    = line[12:16].strip()
                resname = line[17:20].strip()
                resseq  = int(line[22:26].strip())
                x       = float(line[30:38])
                y       = float(line[38:46])
                z       = float(line[46:54])
                # Element: prefer columns 76-78, otherwise use the first atom name letter
                element = line[76:78].strip() if len(line) > 76 else name[0]
                element = re.sub(r"[^A-Za-z]", "", element).upper()
                if not element:
                    element = name[0].upper()
                atoms.append({
                    "name":    name,
                    "resname": resname,
                    "resseq":  resseq,
                    "element": element,
                    "coord":   np.array([x, y, z])
                })
            except (ValueError, IndexError):
                continue
    return atoms


def parse_sdf_atoms(sdf_path):
    """
    Parse ligand atoms from an SDF file and return a list of atoms.
    Each atom contains {"element", "coord": np.array([x,y,z])}
    """
    atoms = []
    with open(sdf_path, "r") as f:
        lines = f.readlines()

    if len(lines) < 4:
        return atoms

    # The 4th line is the counts line: aaabbblllfffcccsssxxxrrrpppiiimmmvvvvvv
    try:
        n_atoms = int(lines[3][0:3].strip())
    except ValueError:
        return atoms

    for i in range(4, 4 + n_atoms):
        if i >= len(lines):
            break
        parts = lines[i].split()
        if len(parts) < 4:
            continue
        try:
            element = re.sub(r"[^A-Za-z]", "", parts[3]).upper()
            if not element:
                continue
            atoms.append({
                "element": element,
                "coord":   np.array([float(parts[0]),
                                     float(parts[1]),
                                     float(parts[2])])
            })
        except ValueError:
            continue
    return atoms


def count_contacts(prot_atoms, lig_atoms, cutoff=5.0):
    """
    Count protein-ligand atom contacts within cutoff Å.

    Returns a dictionary:
        total       — total contacts
        CC          — carbon-carbon (hydrophobic) contacts
        CO          — carbon-oxygen contacts
        CN          — carbon-nitrogen contacts
        SC          — sulfur-carbon (catalytic) contacts
        polar       — polar contacts (N/O/S with N/O/S)
        hydrophobic — hydrophobic contacts (C with C)
    """
    if not prot_atoms or not lig_atoms:
        return None

    # Extract protein coordinate matrix to speed up batch distance calculations
    prot_coords = np.array([a["coord"] for a in prot_atoms])  # (N, 3)
    prot_elems  = [a["element"] for a in prot_atoms]

    total = CC = CO = CN = SC = polar = hydrophobic = 0

    for lig_atom in lig_atoms:
        le = lig_atom["element"]
        if le == "H":
            continue  # skip hydrogen atoms

        lc = lig_atom["coord"]
        # vectorized distance calculation
        diffs = prot_coords - lc          # (N, 3)
        dists = np.sqrt((diffs**2).sum(axis=1))  # (N,)
        nearby_idx = np.where(dists <= cutoff)[0]

        for idx in nearby_idx:
            pe = prot_elems[idx]
            if pe == "H":
                continue

            total += 1

            # classify contact types
            pair = tuple(sorted([pe, le]))
            if pair == ("C", "C"):
                CC += 1
                hydrophobic += 1
            elif pair in [("C", "O"), ("O", "C")]:
                CO += 1
            elif pair in [("C", "N"), ("N", "C")]:
                CN += 1
            elif "S" in pair and "C" in pair:
                SC += 1
            # polar: N/O/S with N/O/S
            if pe in ("N", "O", "S") and le in ("N", "O", "S"):
                polar += 1

    return {
        "total":       total,
        "CC":          CC,
        "CO":          CO,
        "CN":          CN,
        "SC":          SC,
        "polar":       polar,
        "hydrophobic": hydrophobic,
    }


# ─── Main program ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Contact Analysis: protein–ligand 5Å atom contact counts")
    print("=" * 60)

    # 1. collect all complex folders
    all_folders = [
        d for d in os.listdir(RESULTS_DIR)
        if os.path.isdir(os.path.join(RESULTS_DIR, d))
    ]
    print(f"\nFound {len(all_folders)} complex folders")

    # 2. read existing data (ΔG and distance) for merging
    df_dg = pd.read_csv(PRODIGY_CSV)
    df_dg = df_dg.rename(columns={"complex_name": "complex"})

    try:
        df_dist = pd.read_csv(DISTANCE_CSV)
        has_dist = True
        print("✅ Distance data loaded")
    except FileNotFoundError:
        has_dist = False
        print("⚠️  Distance CSV not found; skipping distance merge")

    # 3. iterate over each complex and compute contacts
    rows = []
    success = 0
    failed  = 0

    for folder in sorted(all_folders):
        prot = get_protein_id(folder)
        if prot not in TARGET_PROTEINS:
            continue

        substrate = get_substrate(folder)
        folder_path = os.path.join(RESULTS_DIR, folder)

        pdb_path = os.path.join(folder_path, "complex.pdb")
        sdf_path = os.path.join(folder_path, "rank1.sdf")

        # check file existence
        if not os.path.exists(pdb_path):
            print(f"  [SKIP] missing complex.pdb: {folder}")
            failed += 1
            continue
        if not os.path.exists(sdf_path):
            print(f"  [SKIP] missing rank1.sdf: {folder}")
            failed += 1
            continue

        # parse atoms
        prot_atoms = parse_pdb_atoms(pdb_path)
        lig_atoms  = parse_sdf_atoms(sdf_path)

        if not prot_atoms:
            print(f"  [WARN] no ATOM records in protein: {folder}")
            failed += 1
            continue
        if not lig_atoms:
            print(f"  [WARN] no ligand atoms: {folder}")
            failed += 1
            continue

        # compute contacts
        contacts = count_contacts(prot_atoms, lig_atoms, cutoff=CUTOFF)
        if contacts is None:
            failed += 1
            continue

        rows.append({
            "complex":   folder,
            "protein":   prot,
            "substrate": substrate,
            **contacts
        })
        success += 1

        if success % 50 == 0:
            print(f"  Processed {success} complexes...")

    print(f"\n✅ Success: {success}  ❌ Failed: {failed}")

    # 4. assemble results
    df = pd.DataFrame(rows)
    if df.empty:
        print("❌ No valid results found; please check file paths")
        return

    # 5. merge ΔG
    df = df.merge(
        df_dg[["complex", "dG_kcal_mol"]],
        on="complex", how="left"
    )

    # 6. merge distance
    if has_dist:
        df = df.merge(
            df_dist[["complex", "distance_A"]],
            on="complex", how="left"
        )

    # 7. add carbon chain length
    df["carbon_num"] = df["substrate"].map(ALIPHATIC_MAP)

    # 8. save results
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n📄 Results saved → {OUTPUT_CSV}")
    print(f"   columns: {list(df.columns)}")
    print(f"\nPreview first 5 rows:")
    print(df.head().to_string(index=False))

    # 9. generate charts
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy import stats

        proteins   = ["A15","A16","A17","A18","A19","B1","B2"]
        colors     = ["#2196F3","#4CAF50","#FF9800","#9C27B0","#F44336","#00BCD4","#795548"]

        df_ali = df[df["carbon_num"].notna()].copy()

        # ── Figure 1: contact count vs. carbon chain length ────────────────────────────────────────
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        for i, (prot, col) in enumerate(zip(proteins, colors)):
            ax = axes[i]
            sub = df_ali[df_ali["protein"] == prot].sort_values("carbon_num")
            x, y = sub["carbon_num"].values, sub["total"].values

            ax.plot(x, y, "o-", color=col, linewidth=2, markersize=7,
                    markerfacecolor="white", markeredgewidth=2)

            if len(x) >= 3:
                slope, intercept, r, p, _ = stats.linregress(x, y)
                x_fit = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_fit, slope*x_fit+intercept, "--",
                        color=col, alpha=0.5, linewidth=1.2)
                ax.set_title(f"{prot}  (R²={r**2:.3f}, p={p:.3f})",
                             fontsize=10, fontweight="bold", color=col)
            else:
                ax.set_title(prot, fontsize=10, fontweight="bold", color=col)

            ax.set_xlabel("Carbon chain length (C)", fontsize=9)
            ax.set_ylabel("Contact count (5Å)", fontsize=9)
            ax.set_xticks([1,2,3,4,5,6,7,8,9,10,12,14,16])
            ax.grid(True, alpha=0.3)

        # overlay all proteins
        ax_all = axes[7]
        for prot, col in zip(proteins, colors):
            sub = df_ali[df_ali["protein"] == prot].sort_values("carbon_num")
            ax_all.plot(sub["carbon_num"], sub["total"],
                        color=col, linewidth=1.5, markersize=5,
                        marker="o", label=prot,
                        markerfacecolor="white", markeredgewidth=1.5)
        ax_all.set_title("All proteins (overlay)", fontsize=10, fontweight="bold")
        ax_all.set_xlabel("Carbon chain length (C)", fontsize=9)
        ax_all.set_ylabel("Contact count (5Å)", fontsize=9)
        ax_all.legend(fontsize=7, ncol=2)
        ax_all.grid(True, alpha=0.3)

        fig.suptitle("Protein–ligand contact count (5Å) vs. carbon chain length\n"
                     "across 7 ALDH enzymes", fontsize=12, fontweight="bold", y=1.01)
        plt.tight_layout()
        plt.savefig("Fig_contact_vs_chain.png", dpi=180, bbox_inches="tight")
        plt.close()
        print("\n📊 Fig_contact_vs_chain.png saved")

        # ── Figure 2: correlation scatter matrix for the three metrics ──────────────────────────────────────
        # include protein column for color grouping by protein
        df_corr = df_ali[["protein","total","dG_kcal_mol","distance_A"]].dropna(
            subset=["total","dG_kcal_mol","distance_A"]
        ).reset_index(drop=True)

        if len(df_corr) >= 10:
            labels = ["Contact count", "ΔG (kcal/mol)", "Cys–aldehyde dist (Å)"]
            cols   = ["total", "dG_kcal_mol", "distance_A"]

            fig2, axes2 = plt.subplots(3, 3, figsize=(10, 9))
            for r, (col_r, lab_r) in enumerate(zip(cols, labels)):
                for c, (col_c, lab_c) in enumerate(zip(cols, labels)):
                    ax = axes2[r][c]
                    if r == c:
                        # diagonal: histogram
                        ax.hist(df_corr[col_r].values, bins=20,
                                color="#2196F3", alpha=0.7, edgecolor="white")
                        ax.set_title(lab_r, fontsize=9, fontweight="bold")
                    else:
                        # scatter points colored by protein
                        for prot, col in zip(proteins, colors):
                            sub_p = df_corr[df_corr["protein"] == prot]
                            if len(sub_p) == 0:
                                continue
                            ax.scatter(sub_p[col_c].values,
                                       sub_p[col_r].values,
                                       alpha=0.6, s=20, color=col, label=prot)

                        # overall trend line
                        x_data = df_corr[col_c].values
                        y_data = df_corr[col_r].values
                        if len(x_data) >= 3:
                            r_val, p_val = stats.pearsonr(x_data, y_data)
                            slope, intercept, *_ = stats.linregress(x_data, y_data)
                            x_fit = np.linspace(x_data.min(), x_data.max(), 100)
                            ax.plot(x_fit, slope*x_fit+intercept,
                                    "k--", linewidth=1.2, alpha=0.7)
                            ax.set_title(f"r={r_val:.3f}, p={p_val:.3f}",
                                         fontsize=8, color="darkred")

                        # add legend only in row 1, column 2
                        if r == 0 and c == 1:
                            ax.legend(fontsize=6, ncol=2,
                                      loc="upper right", markerscale=1.5)

                    if r == 2:
                        ax.set_xlabel(lab_c, fontsize=8)
                    if c == 0:
                        ax.set_ylabel(lab_r, fontsize=8)
                    ax.tick_params(labelsize=7)
                    ax.grid(True, alpha=0.2)

            fig2.suptitle("Correlation matrix: Contact count, ΔG, and Cys–aldehyde distance",
                          fontsize=11, fontweight="bold")
            plt.tight_layout()
            plt.savefig("Fig_three_metrics.png", dpi=180, bbox_inches="tight")
            plt.close()
            print("📊 Fig_three_metrics.png saved")

        # ── Figure 3: stacked bar chart by contact type (averaged across proteins) ─────────
        df_stack = df_ali.groupby("carbon_num")[
            ["CC","CO","CN","SC","polar"]
        ].mean().reset_index()

        fig3, ax3 = plt.subplots(figsize=(10, 5))
        x_pos  = np.arange(len(df_stack))
        bottom = np.zeros(len(df_stack))
        type_colors = ["#1565C0","#2E7D32","#F57F17","#6A1B9A","#C62828"]
        type_labels = ["C–C (hydrophobic)","C–O","C–N","S–C (catalytic)","polar (N/O/S)"]

        for tcol, tcol_color, tlab in zip(
            ["CC","CO","CN","SC","polar"], type_colors, type_labels
        ):
            vals = df_stack[tcol].values
            ax3.bar(x_pos, vals, bottom=bottom, color=tcol_color,
                    label=tlab, alpha=0.85, width=0.6)
            bottom += vals

        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(
            [f"C{int(c)}" for c in df_stack["carbon_num"]], fontsize=9
        )
        ax3.set_xlabel("Carbon chain length", fontsize=11)
        ax3.set_ylabel("Mean contact count (5Å)", fontsize=11)
        ax3.set_title("Contact type composition vs. carbon chain length\n"
                      "(averaged across 7 ALDH enzymes)", fontsize=11, fontweight="bold")
        ax3.legend(fontsize=9, loc="upper left")
        ax3.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig("Fig_contact_type_stack.png", dpi=180, bbox_inches="tight")
        plt.close()
        print("📊 Fig_contact_type_stack.png saved")

    except ImportError as e:
        print(f"  plotting skipped (missing library): {e}")

    print(" Done!")
    print(f"   output file: {OUTPUT_CSV}")
    print("   figures: Fig_contact_vs_chain.png")
    print("            Fig_three_metrics.png")
    print("            Fig_contact_type_stack.png")


if __name__ == "__main__":
    main()
