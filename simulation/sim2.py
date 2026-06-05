import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap


# =========================================================
# basic utils
# =========================================================
def softmax(x):
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / ex.sum()


def gaussian_bump(t, peak, sigma):
    return np.exp(-((t - peak) ** 2) / (2 * sigma ** 2))


def cyclic_signal(t, phase, amp=1.0, period=1.0):
    return amp * (1.0 + np.sin(2 * np.pi * t / period + phase)) / 2.0


# =========================================================
# main simulator
# =========================================================
def simulate_timeseries_topology(
    topology="linear",              # "linear", "bifurcation", "cyclic"
    n_timepoints=8,
    reps_per_time=3,
    cells_per_sample=300,
    cells_per_sample_sd=30,
    seed=1,
    model="negbin",                 # "poisson" or "negbin"
    dispersion_theta=20.0,
    dropout_rate=0.0,
    library_scale=1.0,
    branch_mode="mixed",            # bifurcation only: "mixed" or "split_samples"
    bifurcation_time=0.4,

    # realism knobs
    dirichlet_conc=12.0,            # lower => more sample-to-sample composition variability
    sample_time_sd=0.02,            # sample-level temporal jitter
    cell_time_sd=0.06,              # cell-level temporal jitter
    marker_strength=0.16,           # weaker discrete identity
    dynamic_strength=1.25,          # stronger trajectory program
    shared_strength=1.35,           # stronger shared temporal program
    phase_label_mode="soft",        # cyclic: "soft" or "hard"
    gene_dispersion=True,           # gene-specific NB theta
    dropout_mid=1.2,                # expression-dependent dropout midpoint on log1p(mu)
    dropout_shape=1.1,              # expression-dependent dropout slope
):
    """
    Topology-aware time-series simulator with more realistic heterogeneity.

    Returns
    -------
    counts_df : gene x cell count matrix
    cell_meta : per-cell metadata
    sample_meta : per-sample metadata
    gene_meta : per-gene metadata
    pseudobulk_df : gene x sample pseudobulk counts
    composition_df : sample x composition table
    """

    rng = np.random.default_rng(seed)

    # ---------------------------------------------------------
    # 1) topology-specific cell types and gene programs
    # ---------------------------------------------------------
    if topology == "linear":
        cell_types = ["CT0", "CT1", "CT2"]
        genes_per_program = {
            "shared_early": 40,
            "shared_mid": 40,
            "shared_late": 40,
            "ct0_marker": 60,
            "ct1_marker": 60,
            "ct2_marker": 60,
            "dynamic_ct0": 40,
            "dynamic_ct1": 40,
            "dynamic_ct2": 40,
        }

    elif topology == "bifurcation":
        cell_types = ["Prog", "FateA", "FateB"]
        genes_per_program = {
            "shared_pre": 50,
            "shared_transition": 40,
            "prog_marker": 60,
            "fateA_marker": 70,
            "fateB_marker": 70,
            "dynamic_prog": 40,
            "dynamic_fateA": 50,
            "dynamic_fateB": 50,
        }

    elif topology == "cyclic":
        cell_types = ["Phase0", "Phase1", "Phase2"]
        genes_per_program = {
            "cyclic_shared": 80,
            "phase0_marker": 60,
            "phase1_marker": 60,
            "phase2_marker": 60,
            "dynamic_phase0": 40,
            "dynamic_phase1": 40,
            "dynamic_phase2": 40,
        }

    else:
        raise ValueError("topology must be one of: linear, bifurcation, cyclic")

    # ---------------------------------------------------------
    # 2) sample metadata
    # ---------------------------------------------------------
    times = np.linspace(0, 1, n_timepoints)

    sample_rows = []
    for t_idx, t in enumerate(times):
        for rep in range(reps_per_time):
            t_obs = float(np.clip(t + rng.normal(0, sample_time_sd), 0.0, 1.0))

            row = {
                "sample_id": f"S{t_idx}_R{rep}",
                "time_idx": t_idx,
                "time": t,
                "time_observed": t_obs,
                "replicate": rep,
                "topology": topology
            }

            if topology == "bifurcation" and branch_mode == "split_samples":
                if t < bifurcation_time:
                    row["sample_branch"] = "pre"
                else:
                    row["sample_branch"] = rng.choice(["A", "B"])
            else:
                row["sample_branch"] = "mixed"

            sample_rows.append(row)

    sample_meta = pd.DataFrame(sample_rows)

    # ---------------------------------------------------------
    # 3) composition functions
    # ---------------------------------------------------------
    def composition_linear(t):
        z0 = 2.0 - 3.3 * t
        z1 = 1.45 - 5.8 * (t - 0.5) ** 2
        z2 = -0.1 + 3.0 * t
        return softmax([z0, z1, z2])

    def composition_bifurcation_mixed(t):
        if t < bifurcation_time:
            z_prog = 3.2 - 5.2 * t
            z_a = -1.8 + 0.6 * t
            z_b = -1.8 + 0.6 * t
        else:
            dt = t - bifurcation_time
            z_prog = 0.8 - 3.0 * dt
            z_a = -0.2 + 2.2 * dt
            z_b = -0.2 + 2.2 * dt
        return softmax([z_prog, z_a, z_b])

    def composition_bifurcation_split(t, sample_branch):
        if t < bifurcation_time:
            z_prog = 2.8 - 4.5 * t
            z_a = -2.0 + 1.0 * t
            z_b = -2.0 + 1.0 * t
        else:
            z_prog = 1.0 - 4.0 * t
            if sample_branch == "A":
                z_a = 2.5 + 4.0 * (t - bifurcation_time)
                z_b = -1.5
            else:
                z_a = -1.5
                z_b = 2.5 + 4.0 * (t - bifurcation_time)
        return softmax([z_prog, z_a, z_b])

    def composition_cyclic(t):
        p0 = 1.35 + 0.55 * np.sin(2 * np.pi * t + 0.0)
        p1 = 1.35 + 0.55 * np.sin(2 * np.pi * t + 2 * np.pi / 3)
        p2 = 1.35 + 0.55 * np.sin(2 * np.pi * t + 4 * np.pi / 3)
        p = np.array([p0, p1, p2])
        p = np.clip(p, 1e-4, None)
        return p / p.sum()

    composition_rows = []
    for _, srow in sample_meta.iterrows():
        sid = srow["sample_id"]
        t = srow["time_observed"]

        if topology == "linear":
            base_p = composition_linear(t)

        elif topology == "bifurcation":
            if branch_mode == "split_samples":
                base_p = composition_bifurcation_split(t, srow["sample_branch"])
            else:
                base_p = composition_bifurcation_mixed(t)

        elif topology == "cyclic":
            base_p = composition_cyclic(t)

        noisy = rng.dirichlet(np.clip(dirichlet_conc * base_p, 1e-3, None))

        comp_row = {"sample_id": sid, "time": srow["time"], "time_observed": t}
        for i, ct in enumerate(cell_types):
            comp_row[ct] = noisy[i]
        composition_rows.append(comp_row)

    composition_df = pd.DataFrame(composition_rows)

    # ---------------------------------------------------------
    # 4) gene programs
    # ---------------------------------------------------------
    gene_rows = []
    gene_names = []

    def add_genes(program_name, n, preferred_ct=None):
        start = len(gene_names)
        for i in range(n):
            gid = f"G{start+i}"
            gene_names.append(gid)
            gene_rows.append({
                "gene_id": gid,
                "program": program_name,
                "preferred_cell_type": preferred_ct
            })

    if topology == "linear":
        add_genes("shared_early", genes_per_program["shared_early"])
        add_genes("shared_mid", genes_per_program["shared_mid"])
        add_genes("shared_late", genes_per_program["shared_late"])
        add_genes("ct0_marker", genes_per_program["ct0_marker"], "CT0")
        add_genes("ct1_marker", genes_per_program["ct1_marker"], "CT1")
        add_genes("ct2_marker", genes_per_program["ct2_marker"], "CT2")
        add_genes("dynamic_ct0", genes_per_program["dynamic_ct0"], "CT0")
        add_genes("dynamic_ct1", genes_per_program["dynamic_ct1"], "CT1")
        add_genes("dynamic_ct2", genes_per_program["dynamic_ct2"], "CT2")

    elif topology == "bifurcation":
        add_genes("shared_pre", genes_per_program["shared_pre"])
        add_genes("shared_transition", genes_per_program["shared_transition"])
        add_genes("prog_marker", genes_per_program["prog_marker"], "Prog")
        add_genes("fateA_marker", genes_per_program["fateA_marker"], "FateA")
        add_genes("fateB_marker", genes_per_program["fateB_marker"], "FateB")
        add_genes("dynamic_prog", genes_per_program["dynamic_prog"], "Prog")
        add_genes("dynamic_fateA", genes_per_program["dynamic_fateA"], "FateA")
        add_genes("dynamic_fateB", genes_per_program["dynamic_fateB"], "FateB")

    elif topology == "cyclic":
        add_genes("cyclic_shared", genes_per_program["cyclic_shared"])
        add_genes("phase0_marker", genes_per_program["phase0_marker"], "Phase0")
        add_genes("phase1_marker", genes_per_program["phase1_marker"], "Phase1")
        add_genes("phase2_marker", genes_per_program["phase2_marker"], "Phase2")
        add_genes("dynamic_phase0", genes_per_program["dynamic_phase0"], "Phase0")
        add_genes("dynamic_phase1", genes_per_program["dynamic_phase1"], "Phase1")
        add_genes("dynamic_phase2", genes_per_program["dynamic_phase2"], "Phase2")

    gene_meta = pd.DataFrame(gene_rows)
    n_genes = len(gene_meta)

    # gene-level parameters
    gene_meta["baseline"] = rng.lognormal(mean=0.9, sigma=0.55, size=n_genes)
    gene_meta["marker_effect"] = rng.lognormal(mean=0.45, sigma=0.40, size=n_genes)
    gene_meta["dynamic_amp"] = rng.lognormal(mean=1.05, sigma=0.45, size=n_genes)
    gene_meta["sigma"] = rng.uniform(0.08, 0.30, size=n_genes)
    gene_meta["phase"] = rng.uniform(0, 2 * np.pi, size=n_genes)

    if gene_dispersion:
        gene_meta["theta"] = rng.lognormal(
            mean=np.log(dispersion_theta), sigma=0.45, size=n_genes
        )
    else:
        gene_meta["theta"] = dispersion_theta

    # topology-specific peak times
    if topology == "linear":
        peak_map = {
            "shared_early": 0.15,
            "shared_mid": 0.50,
            "shared_late": 0.85,
            "dynamic_ct0": 0.20,
            "dynamic_ct1": 0.50,
            "dynamic_ct2": 0.80,
        }
    elif topology == "bifurcation":
        peak_map = {
            "shared_pre": 0.20,
            "shared_transition": bifurcation_time,
            "dynamic_prog": 0.22,
            "dynamic_fateA": min(0.78, bifurcation_time + 0.25),
            "dynamic_fateB": min(0.82, bifurcation_time + 0.30),
        }
    else:
        peak_map = {}

    gene_meta["peak_time"] = gene_meta["program"].map(peak_map).fillna(np.nan)
    mask = gene_meta["peak_time"].notna()
    gene_meta.loc[mask, "peak_time"] = np.clip(
        gene_meta.loc[mask, "peak_time"] + rng.normal(0, 0.09, mask.sum()),
        0.0, 1.0
    )

    # ---------------------------------------------------------
    # 5) generate cells
    # ---------------------------------------------------------
    cell_rows = []
    for _, srow in sample_meta.iterrows():
        sid = srow["sample_id"]
        t = srow["time"]

        n_cells_sample = max(20, int(rng.normal(cells_per_sample, cells_per_sample_sd)))

        comp_vals = composition_df.loc[
            composition_df["sample_id"] == sid, cell_types
        ].values[0]

        n_ct = rng.multinomial(n_cells_sample, comp_vals)

        for ct_i, ct in enumerate(cell_types):
            for k in range(n_ct[ct_i]):
                cid = f"{sid}_{ct}_{k}"
                cell_rows.append({
                    "cell_id": cid,
                    "sample_id": sid,
                    "time": t,
                    "time_idx": srow["time_idx"],
                    "replicate": srow["replicate"],
                    "cell_type": ct,
                    "topology": topology,
                    "sample_branch": srow["sample_branch"]
                })

    cell_meta = pd.DataFrame(cell_rows)
    n_cells = len(cell_meta)

    # sample_id -> observed time map
    sample_time_obs_map = sample_meta.set_index("sample_id")["time_observed"].to_dict()

    # ---------------------------------------------------------
    # 5.5) latent per-cell trajectory variables
    # ---------------------------------------------------------
    tau_list = []
    branch_progress_list = []
    wA_list = []
    wB_list = []
    soft_label_list = []

    for _, crow in cell_meta.iterrows():
        sid = crow["sample_id"]
        t_obs = sample_time_obs_map[sid]

        tau = float(np.clip(rng.normal(t_obs, cell_time_sd), 0.0, 1.0))

        if topology == "linear":
            tau_list.append(tau)
            branch_progress_list.append(np.nan)
            wA_list.append(np.nan)
            wB_list.append(np.nan)
            soft_label_list.append(crow["cell_type"])
            continue

        if topology == "cyclic":
            tau_cyc = float((tau + rng.normal(0, 0.04)) % 1.0)

            s0 = np.sin(2 * np.pi * tau_cyc + 0.0)
            s1 = np.sin(2 * np.pi * tau_cyc + 2 * np.pi / 3)
            s2 = np.sin(2 * np.pi * tau_cyc + 4 * np.pi / 3)

            if phase_label_mode == "hard":
                soft_lab = ["Phase0", "Phase1", "Phase2"][np.argmax([s0, s1, s2])]
            else:
                soft_lab = "cyclic_continuum"

            tau_list.append(tau_cyc)
            branch_progress_list.append(np.nan)
            wA_list.append(np.nan)
            wB_list.append(np.nan)
            soft_label_list.append(soft_lab)
            continue

        # bifurcation
        if tau <= bifurcation_time:
            bp = 0.0
        else:
            bp = (tau - bifurcation_time) / (1.0 - bifurcation_time)
            bp = float(np.clip(bp, 0.0, 1.0))

        alpha = max(0.45, 3.2 - 2.0 * bp)
        a_bias = rng.beta(alpha, alpha)
        b_bias = 1.0 - a_bias

        wA = bp * a_bias
        wB = bp * b_bias

        if bp < 0.18:
            soft_lab = "Prog"
        else:
            soft_lab = "FateA" if wA >= wB else "FateB"

        tau_list.append(tau)
        branch_progress_list.append(bp)
        wA_list.append(wA)
        wB_list.append(wB)
        soft_label_list.append(soft_lab)

    cell_meta["tau"] = tau_list
    cell_meta["branch_progress"] = branch_progress_list
    cell_meta["wA"] = wA_list
    cell_meta["wB"] = wB_list
    cell_meta["cell_type_soft"] = soft_label_list

    # ---------------------------------------------------------
    # 6) mean expression
    # ---------------------------------------------------------
    mu = np.zeros((n_genes, n_cells), dtype=float)

    sample_sf = {
        sid: rng.lognormal(mean=0.0, sigma=0.15)
        for sid in sample_meta["sample_id"]
    }

    for j, crow in cell_meta.iterrows():
        ct = crow["cell_type"]
        sid = crow["sample_id"]

        tau = crow["tau"]
        branch_progress = 0.0 if pd.isna(crow["branch_progress"]) else crow["branch_progress"]
        wA = 0.0 if pd.isna(crow["wA"]) else crow["wA"]
        wB = 0.0 if pd.isna(crow["wB"]) else crow["wB"]

        cell_sf = rng.lognormal(mean=0.0, sigma=0.20) * library_scale * sample_sf[sid]

        for i, grow in gene_meta.iterrows():
            val = grow["baseline"]
            prog = grow["program"]
            pref = grow["preferred_cell_type"]

            # -----------------------------
            # linear
            # -----------------------------
            if topology == "linear":
                tau_lin = tau

                w0 = gaussian_bump(tau_lin, 0.18, 0.24)
                w1 = gaussian_bump(tau_lin, 0.50, 0.24)
                w2 = gaussian_bump(tau_lin, 0.82, 0.24)
                wsum = w0 + w1 + w2 + 1e-8
                w0, w1, w2 = w0 / wsum, w1 / wsum, w2 / wsum

                ct_weight_map = {"CT0": w0, "CT1": w1, "CT2": w2}
                pref_w = ct_weight_map.get(pref, 0.0)
                same_ct = float(pref == ct)

                if prog in ("shared_early", "shared_mid", "shared_late"):
                    dyn = gaussian_bump(tau_lin, grow["peak_time"], max(grow["sigma"], 0.10))
                    val += shared_strength * grow["dynamic_amp"] * dyn

                elif prog.startswith("ct") and prog.endswith("marker"):
                    val += marker_strength * grow["marker_effect"] * (
                        0.15 + 0.65 * pref_w + 0.20 * same_ct
                    )

                elif prog.startswith("dynamic_ct"):
                    dyn = gaussian_bump(tau_lin, grow["peak_time"], max(grow["sigma"], 0.09))
                    val += dynamic_strength * grow["dynamic_amp"] * dyn * (
                        0.20 + 0.60 * pref_w + 0.15 * same_ct
                    )

            # -----------------------------
            # bifurcation
            # -----------------------------
            elif topology == "bifurcation":
                prog_w = max(0.0, 1.0 - branch_progress)

                if prog == "prog_marker":
                    val += marker_strength * grow["marker_effect"] * (0.70 * prog_w + 0.20)

                elif prog == "fateA_marker":
                    val += marker_strength * grow["marker_effect"] * (0.15 + 0.75 * wA)

                elif prog == "fateB_marker":
                    val += marker_strength * grow["marker_effect"] * (0.15 + 0.75 * wB)

                elif prog == "dynamic_prog":
                    pre_bump = gaussian_bump(tau, grow["peak_time"], max(grow["sigma"], 0.09))
                    val += dynamic_strength * grow["dynamic_amp"] * pre_bump * (
                        0.75 * prog_w + 0.15
                    )

                elif prog == "dynamic_fateA":
                    post_bump = gaussian_bump(tau, grow["peak_time"], max(grow["sigma"], 0.09))
                    val += dynamic_strength * grow["dynamic_amp"] * post_bump * (
                        0.15 + 0.80 * wA
                    )

                elif prog == "dynamic_fateB":
                    post_bump = gaussian_bump(tau, grow["peak_time"], max(grow["sigma"], 0.09))
                    val += dynamic_strength * grow["dynamic_amp"] * post_bump * (
                        0.15 + 0.80 * wB
                    )

                elif prog == "shared_pre":
                    val += shared_strength * grow["dynamic_amp"] * gaussian_bump(
                        tau, grow["peak_time"], max(grow["sigma"], 0.09)
                    ) * (0.80 * prog_w + 0.20)

                elif prog == "shared_transition":
                    val += 1.25 * shared_strength * grow["dynamic_amp"] * gaussian_bump(
                        tau, bifurcation_time, max(grow["sigma"], 0.12)
                    )

            # -----------------------------
            # cyclic
            # -----------------------------
            elif topology == "cyclic":
                tau_cyc = tau

                w0 = 1.0 + np.sin(2 * np.pi * tau_cyc + 0.0)
                w1 = 1.0 + np.sin(2 * np.pi * tau_cyc + 2 * np.pi / 3)
                w2 = 1.0 + np.sin(2 * np.pi * tau_cyc + 4 * np.pi / 3)
                w = np.array([w0, w1, w2], dtype=float)
                w = np.clip(w, 1e-6, None)
                w = w / w.sum()

                phase_weight_map = {"Phase0": w[0], "Phase1": w[1], "Phase2": w[2]}
                pref_w = phase_weight_map.get(pref, 0.0)
                same_ct = float(pref == ct)

                if prog == "cyclic_shared":
                    val += shared_strength * cyclic_signal(
                        tau_cyc, grow["phase"], amp=grow["dynamic_amp"], period=1.0
                    )

                elif prog in ("phase0_marker", "phase1_marker", "phase2_marker"):
                    val += marker_strength * grow["marker_effect"] * (
                        0.12 + 0.70 * pref_w + 0.10 * same_ct
                    )

                elif prog.startswith("dynamic_phase"):
                    dyn = cyclic_signal(
                        tau_cyc, grow["phase"], amp=grow["dynamic_amp"], period=1.0
                    )
                    val += dynamic_strength * dyn * (
                        0.20 + 0.65 * pref_w + 0.10 * same_ct
                    )

            mu[i, j] = max(val * cell_sf, 1e-5)

    # ---------------------------------------------------------
    # 7) sample counts
    # ---------------------------------------------------------
    if model == "poisson":
        counts = rng.poisson(mu)

    elif model == "negbin":
        if gene_dispersion:
            theta = gene_meta["theta"].values[:, None]
        else:
            theta = float(dispersion_theta)

        lam = rng.gamma(shape=theta, scale=mu / theta)
        counts = rng.poisson(lam)

    else:
        raise ValueError("model must be 'poisson' or 'negbin'")

    # expression-dependent dropout
    if dropout_rate > 0:
        log_mu = np.log1p(mu)
        p_drop = dropout_rate / (1.0 + np.exp((log_mu - dropout_mid) / dropout_shape))
        drop_mask = rng.uniform(size=counts.shape) < p_drop
        counts[drop_mask] = 0

    counts_df = pd.DataFrame(
        counts,
        index=gene_meta["gene_id"],
        columns=cell_meta["cell_id"]
    )

    # ---------------------------------------------------------
    # 8) pseudobulk
    # ---------------------------------------------------------
    sample_ids = sample_meta["sample_id"].tolist()
    pseudobulk_cols = []

    for sid in sample_ids:
        cols = cell_meta.loc[cell_meta["sample_id"] == sid, "cell_id"].tolist()
        pseudobulk_cols.append(counts_df[cols].sum(axis=1).values)

    pseudobulk_df = pd.DataFrame(
        np.column_stack(pseudobulk_cols),
        index=gene_meta["gene_id"],
        columns=sample_ids
    )

    return counts_df, cell_meta, sample_meta, gene_meta, pseudobulk_df, composition_df


# =========================================================
# heatmaps
# =========================================================
def plot_pseudobulk_heatmap(
    pseudobulk_df,
    sample_meta,
    gene_meta=None,
    n_top_genes=120,
    scale_rows=True,
    figsize=(12, 7)
):
    sample_order = sample_meta.sort_values(["time", "replicate"])["sample_id"].tolist()
    mat = pseudobulk_df[sample_order].copy()

    gene_var = mat.var(axis=1).sort_values(ascending=False)
    top_genes = gene_var.head(n_top_genes).index
    mat = mat.loc[top_genes]

    mat = np.log1p(mat)

    if scale_rows:
        row_mean = mat.mean(axis=1)
        row_std = mat.std(axis=1).replace(0, 1)
        mat = mat.sub(row_mean, axis=0).div(row_std, axis=0)

    plt.figure(figsize=figsize)
    sns.heatmap(
        mat,
        cmap="viridis",
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "scaled log-expression" if scale_rows else "log-expression"}
    )
    plt.title("Pseudobulk heatmap")
    plt.xlabel("Samples (ordered by time)")
    plt.ylabel("Top variable genes")
    plt.tight_layout()
    plt.show()


def plot_program_heatmap(
    pseudobulk_df,
    sample_meta,
    gene_meta,
    scale_rows=True,
    figsize=(10, 5)
):
    sample_order = sample_meta.sort_values(["time", "replicate"])["sample_id"].tolist()
    mat = np.log1p(pseudobulk_df[sample_order])

    program_means = []
    program_names = []

    for program, sub in gene_meta.groupby("program"):
        genes = sub["gene_id"].tolist()
        genes = [g for g in genes if g in mat.index]
        if len(genes) == 0:
            continue
        program_means.append(mat.loc[genes].mean(axis=0).values)
        program_names.append(program)

    prog_df = pd.DataFrame(program_means, index=program_names, columns=sample_order)

    if scale_rows:
        row_mean = prog_df.mean(axis=1)
        row_std = prog_df.std(axis=1).replace(0, 1)
        prog_df = prog_df.sub(row_mean, axis=0).div(row_std, axis=0)

    plt.figure(figsize=figsize)
    sns.heatmap(
        prog_df,
        cmap="viridis",
        xticklabels=False,
        yticklabels=True,
        cbar_kws={"label": "scaled program activity" if scale_rows else "program activity"}
    )
    plt.title("Gene program heatmap")
    plt.xlabel("Samples (ordered by time)")
    plt.ylabel("Programs")
    plt.tight_layout()
    plt.show()


def plot_singlecell_heatmap(
    counts_df,
    cell_meta,
    gene_meta=None,
    n_cells=300,
    n_genes=80,
    scale_rows=True,
    random_state=0,
    figsize=(12, 6)
):
    rng = np.random.default_rng(random_state)

    if counts_df.shape[1] > n_cells:
        chosen_cells = rng.choice(counts_df.columns, size=n_cells, replace=False)
    else:
        chosen_cells = counts_df.columns.to_numpy()

    sub_meta = cell_meta.set_index("cell_id").loc[chosen_cells].reset_index()
    sort_cols = ["time", "tau", "cell_type"] if "tau" in sub_meta.columns else ["time", "cell_type"]
    sub_meta = sub_meta.sort_values(sort_cols)

    mat = counts_df[sub_meta["cell_id"]].copy()

    gene_var = mat.var(axis=1).sort_values(ascending=False)
    top_genes = gene_var.head(n_genes).index
    mat = mat.loc[top_genes]

    mat = np.log1p(mat)

    if scale_rows:
        row_mean = mat.mean(axis=1)
        row_std = mat.std(axis=1).replace(0, 1)
        mat = mat.sub(row_mean, axis=0).div(row_std, axis=0)

    plt.figure(figsize=figsize)
    sns.heatmap(
        mat,
        cmap="viridis",
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "scaled log-expression" if scale_rows else "log-expression"}
    )
    plt.title("Single-cell heatmap")
    plt.xlabel("Cells")
    plt.ylabel("Top variable genes")
    plt.tight_layout()
    plt.show()


def plot_cyclic_heatmap(
    counts_df,
    cell_meta,
    gene_meta,
    cell_id_col="cell_id",
    gene_id_col="gene_id",
    cell_phase_col="tau",
    gene_phase_col="phase",
    program_col="program",
    figsize=(12, 7),
    use_log=True,
    program_filter="cyclic_shared",
    n_genes=None
):
    cm = cell_meta.copy()
    gm = gene_meta.copy()
    counts = counts_df.copy()

    counts.columns = counts.columns.astype(str)
    counts.index = counts.index.astype(str)
    cm[cell_id_col] = cm[cell_id_col].astype(str)
    gm[gene_id_col] = gm[gene_id_col].astype(str)

    ordered_cells = (
        cm.sort_values(cell_phase_col)[cell_id_col].astype(str).tolist()
    )
    ordered_cells = [c for c in ordered_cells if c in counts.columns]

    if program_filter is not None and program_col in gm.columns:
        gm = gm[gm[program_col] == program_filter].copy()

    if gene_phase_col in gm.columns:
        gm = gm.sort_values(gene_phase_col)
    else:
        gm = gm.copy()

    ordered_genes = gm[gene_id_col].astype(str).tolist()
    ordered_genes = [g for g in ordered_genes if g in counts.index]

    if n_genes is not None:
        ordered_genes = ordered_genes[:n_genes]

    mat = counts.loc[ordered_genes, ordered_cells]

    if use_log:
        mat = np.log1p(mat)

    row_mean = mat.mean(axis=1)
    row_std = mat.std(axis=1).replace(0, 1)
    mat = mat.sub(row_mean, axis=0).div(row_std, axis=0)

    plt.figure(figsize=figsize)
    sns.heatmap(
        mat,
        cmap="twilight",
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "scaled log-expression"}
    )
    plt.title(f"Cyclic heatmap ({program_filter})")
    plt.xlabel("Cells ordered by cyclic phase")
    plt.ylabel("Genes ordered by phase")
    plt.tight_layout()
    plt.show()

    return mat, gm, cm


# =========================================================
# UMAP helpers
# =========================================================
def prepare_umap_matrix(
    counts_df,
    gene_meta=None,
    n_top_genes=1000,
    exclude_marker_genes=False
):
    mat = np.log1p(counts_df)

    if exclude_marker_genes and gene_meta is not None:
        gm = gene_meta.set_index("gene_id").loc[mat.index]
        keep = ~gm["program"].str.contains("marker", case=False, na=False)
        mat = mat.loc[gm.index[keep]]

    gene_var = mat.var(axis=1).sort_values(ascending=False)
    top_genes = gene_var.head(min(n_top_genes, mat.shape[0])).index

    X = mat.loc[top_genes].T.values
    X = StandardScaler().fit_transform(X)

    return X, top_genes.tolist()


def _fit_umap(
    counts_df,
    gene_meta=None,
    n_top_genes=1000,
    random_state=42,
    exclude_marker_genes=False,
    n_neighbors=20,
    min_dist=0.35
):
    X, used_genes = prepare_umap_matrix(
        counts_df,
        gene_meta=gene_meta,
        n_top_genes=n_top_genes,
        exclude_marker_genes=exclude_marker_genes
    )

    reducer = umap.UMAP(
        n_components=2,
        random_state=random_state,
        n_neighbors=n_neighbors,
        min_dist=min_dist
    )
    emb = reducer.fit_transform(X)
    return emb, used_genes


def plot_umap_by_time(
    counts_df,
    cell_meta,
    gene_meta=None,
    n_top_genes=1000,
    random_state=42,
    figsize=(6, 5),
    exclude_marker_genes=False,
    n_neighbors=20,
    min_dist=0.35
):
    emb, _ = _fit_umap(
        counts_df,
        gene_meta=gene_meta,
        n_top_genes=n_top_genes,
        random_state=random_state,
        exclude_marker_genes=exclude_marker_genes,
        n_neighbors=n_neighbors,
        min_dist=min_dist
    )

    plot_df = cell_meta.copy().reset_index(drop=True)
    plot_df["UMAP1"] = emb[:, 0]
    plot_df["UMAP2"] = emb[:, 1]

    color_col = "tau" if "tau" in plot_df.columns else "time"

    plt.figure(figsize=figsize)
    sc = plt.scatter(
        plot_df["UMAP1"],
        plot_df["UMAP2"],
        c=plot_df[color_col],
        s=8,
        alpha=0.8
    )
    plt.colorbar(sc, label=color_col)
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title(f"UMAP colored by {color_col}")
    plt.tight_layout()
    plt.show()


def plot_umap_by_branch(
    counts_df,
    cell_meta,
    gene_meta=None,
    n_top_genes=1000,
    random_state=42,
    figsize=(6, 5),
    exclude_marker_genes=False,
    n_neighbors=20,
    min_dist=0.35
):
    emb, _ = _fit_umap(
        counts_df,
        gene_meta=gene_meta,
        n_top_genes=n_top_genes,
        random_state=random_state,
        exclude_marker_genes=exclude_marker_genes,
        n_neighbors=n_neighbors,
        min_dist=min_dist
    )

    plot_df = cell_meta.copy().reset_index(drop=True)
    plot_df["UMAP1"] = emb[:, 0]
    plot_df["UMAP2"] = emb[:, 1]

    plt.figure(figsize=figsize)
    for br in plot_df["sample_branch"].unique():
        sub = plot_df[plot_df["sample_branch"] == br]
        plt.scatter(
            sub["UMAP1"],
            sub["UMAP2"],
            s=8,
            alpha=0.8,
            label=br
        )

    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title("UMAP colored by sample branch")
    plt.legend(markerscale=2)
    plt.tight_layout()
    plt.show()


def plot_umap_by_celltype(
    counts_df,
    cell_meta,
    gene_meta=None,
    n_top_genes=1000,
    random_state=42,
    figsize=(6, 5),
    exclude_marker_genes=False,
    n_neighbors=20,
    min_dist=0.35,
    use_soft_label=False
):
    emb, _ = _fit_umap(
        counts_df,
        gene_meta=gene_meta,
        n_top_genes=n_top_genes,
        random_state=random_state,
        exclude_marker_genes=exclude_marker_genes,
        n_neighbors=n_neighbors,
        min_dist=min_dist
    )

    plot_df = cell_meta.copy().reset_index(drop=True)
    plot_df["UMAP1"] = emb[:, 0]
    plot_df["UMAP2"] = emb[:, 1]

    label_col = "cell_type_soft" if use_soft_label and "cell_type_soft" in plot_df.columns else "cell_type"

    plt.figure(figsize=figsize)
    for ct in plot_df[label_col].unique():
        sub = plot_df[plot_df[label_col] == ct]
        plt.scatter(
            sub["UMAP1"],
            sub["UMAP2"],
            s=8,
            alpha=0.8,
            label=ct
        )

    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title(f"UMAP colored by {label_col}")
    plt.legend(markerscale=2, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


# =========================================================
# example runs
# =========================================================
if __name__ == "__main__":
    # -------------------------
    # linear example
    # -------------------------
    counts_df, cell_meta, sample_meta, gene_meta, pseudobulk_df, composition_df = simulate_timeseries_topology(
        topology="linear",
        n_timepoints=10,
        reps_per_time=4,
        cells_per_sample=250,
        model="negbin",
        dispersion_theta=12,
        dropout_rate=0.18,
        dirichlet_conc=10,
        marker_strength=0.14,
        dynamic_strength=1.35,
        shared_strength=1.40,
        sample_time_sd=0.025,
        cell_time_sd=0.07,
        seed=1
    )

    plot_program_heatmap(pseudobulk_df, sample_meta, gene_meta)
    plot_umap_by_time(
        counts_df, cell_meta, gene_meta=gene_meta,
        exclude_marker_genes=True, n_top_genes=800
    )
    plot_umap_by_celltype(
        counts_df, cell_meta, gene_meta=gene_meta,
        exclude_marker_genes=False, n_top_genes=800
    )

    # -------------------------
    # bifurcation example
    # -------------------------
    counts_df_b, cell_meta_b, sample_meta_b, gene_meta_b, pseudobulk_df_b, composition_df_b = simulate_timeseries_topology(
        topology="bifurcation",
        n_timepoints=10,
        reps_per_time=4,
        cells_per_sample=250,
        model="negbin",
        dispersion_theta=10,
        dropout_rate=0.20,
        dirichlet_conc=10,
        marker_strength=0.13,
        dynamic_strength=1.30,
        shared_strength=1.30,
        bifurcation_time=0.45,
        sample_time_sd=0.025,
        cell_time_sd=0.07,
        seed=2
    )

    plot_program_heatmap(pseudobulk_df_b, sample_meta_b, gene_meta_b)
    plot_umap_by_time(
        counts_df_b, cell_meta_b, gene_meta=gene_meta_b,
        exclude_marker_genes=True, n_top_genes=800
    )
    plot_umap_by_celltype(
        counts_df_b, cell_meta_b, gene_meta=gene_meta_b,
        exclude_marker_genes=False, n_top_genes=800, use_soft_label=True
    )
    plot_umap_by_branch(
        counts_df_b, cell_meta_b, gene_meta=gene_meta_b,
        exclude_marker_genes=True, n_top_genes=800
    )

    # -------------------------
    # cyclic example
    # -------------------------
    counts_df_c, cell_meta_c, sample_meta_c, gene_meta_c, pseudobulk_df_c, composition_df_c = simulate_timeseries_topology(
        topology="cyclic",
        n_timepoints=12,
        reps_per_time=4,
        cells_per_sample=220,
        model="negbin",
        dispersion_theta=10,
        dropout_rate=0.16,
        dirichlet_conc=9,
        marker_strength=0.10,
        dynamic_strength=1.45,
        shared_strength=1.50,
        phase_label_mode="soft",
        sample_time_sd=0.02,
        cell_time_sd=0.06,
        seed=3
    )

    plot_program_heatmap(pseudobulk_df_c, sample_meta_c, gene_meta_c)
    plot_umap_by_time(
        counts_df_c, cell_meta_c, gene_meta=gene_meta_c,
        exclude_marker_genes=True, n_top_genes=800,
        n_neighbors=25, min_dist=0.45
    )
    plot_umap_by_celltype(
        counts_df_c, cell_meta_c, gene_meta=gene_meta_c,
        exclude_marker_genes=False, n_top_genes=800, use_soft_label=True
    )
    plot_cyclic_heatmap(
        counts_df_c, cell_meta_c, gene_meta_c,
        cell_phase_col="tau",
        gene_phase_col="phase",
        program_filter="cyclic_shared",
        n_genes=120
    )