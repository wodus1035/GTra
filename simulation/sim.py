import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap

def softmax(x):
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / ex.sum()


def gaussian_bump(t, peak, sigma):
    return np.exp(-((t - peak) ** 2) / (2 * sigma ** 2))


def cyclic_signal(t, phase, amp=1.0, period=1.0):
    # shifted to be nonnegative
    return amp * (1.0 + np.sin(2 * np.pi * t / period + phase)) / 2.0


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
):
    """
    Topology-aware time-series simulator with cell types.

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
            row = {
                "sample_id": f"S{t_idx}_R{rep}",
                "time_idx": t_idx,
                "time": t,
                "replicate": rep,
                "topology": topology
            }

            if topology == "bifurcation" and branch_mode == "split_samples":
                # sample-level branch identity after branch point
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
        # overlap을 크게 해서 연속 trajectory처럼 보이게
        z0 = 2.0 - 3.3 * t
        z1 = 1.45 - 5.8 * (t - 0.5) ** 2
        z2 = -0.1 + 3.0 * t
        return softmax([z0, z1, z2])

    def composition_bifurcation_mixed(t):
        # Prog dominates early, around branch point transition cells are abundant,
        # later FateA/FateB emerge together.
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
        # After branch point, each sample strongly favors one fate
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
        # rotating proportions
        p0 = 1.35 + 0.55 * np.sin(2 * np.pi * t + 0.0)
        p1 = 1.35 + 0.55 * np.sin(2 * np.pi * t + 2 * np.pi / 3)
        p2 = 1.35 + 0.55 * np.sin(2 * np.pi * t + 4 * np.pi / 3)
        p = np.array([p0, p1, p2])
        p = np.clip(p, 1e-4, None)
        return p / p.sum()

    composition_rows = []
    for _, srow in sample_meta.iterrows():
        sid = srow["sample_id"]
        t = srow["time"]

        if topology == "linear":
            base_p = composition_linear(t)

        elif topology == "bifurcation":
            if branch_mode == "split_samples":
                base_p = composition_bifurcation_split(t, srow["sample_branch"])
            else:
                base_p = composition_bifurcation_mixed(t)

        elif topology == "cyclic":
            base_p = composition_cyclic(t)

        noisy = rng.dirichlet(50 * base_p)

        comp_row = {"sample_id": sid, "time": t}
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
    gene_meta["baseline"] = rng.lognormal(mean=1.0, sigma=0.30, size=n_genes)
    gene_meta["marker_effect"] = rng.lognormal(mean=0.85, sigma=0.22, size=n_genes)
    gene_meta["dynamic_amp"] = rng.lognormal(mean=0.95, sigma=0.28, size=n_genes)
    gene_meta["sigma"] = rng.uniform(0.14, 0.26, size=n_genes)
    gene_meta["phase"] = rng.uniform(0, 2 * np.pi, size=n_genes)

    # topology-specific peak times
    peak_map = {}
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
        
    # gene_meta["peak_time"] = gene_meta["program"].map(peak_map).fillna(np.nan)
    gene_meta["peak_time"] = gene_meta["program"].map(peak_map).fillna(np.nan)
    mask = gene_meta["peak_time"].notna()
    gene_meta.loc[mask, "peak_time"] = np.clip(
        gene_meta.loc[mask, "peak_time"] + rng.normal(0, 0.05, mask.sum()),
        0.0, 1.0
    )

    # ---------------------------------------------------------
    # 5) generate cells
    # ---------------------------------------------------------
    cell_rows = []
    cell_ids = []

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
                cell_ids.append(cid)
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
    
    # ---------------------------------------------------------
    # 5.5) latent per-cell trajectory variables
    # ---------------------------------------------------------
    tau_list = []
    branch_progress_list = []
    wA_list = []
    wB_list = []
    soft_label_list = []

    for _, crow in cell_meta.iterrows():
        t = crow["time"]

        # sample-level time 주변으로 cell-level pseudotime 분산
        # tau = float(np.clip(rng.normal(t, 0.08), 0.0, 1.0))
        sample_time_shift = rng.normal(0, 0.003)
        tau = float(np.clip(rng.normal(t + sample_time_shift, 0.09), 0.0, 1.0))

        if topology == "linear":
            tau_list.append(tau)
            branch_progress_list.append(np.nan)
            wA_list.append(np.nan)
            wB_list.append(np.nan)
            soft_label_list.append(crow["cell_type"])
            continue

        if topology == "cyclic":
            # cyclic은 phase가 원형이라서 wrap-around 유지
            tau_cyc = float((tau + rng.normal(0, 0.035)) % 1.0)

            s0 = np.sin(2 * np.pi * tau_cyc + 0.0)
            s1 = np.sin(2 * np.pi * tau_cyc + 2 * np.pi / 3)
            s2 = np.sin(2 * np.pi * tau_cyc + 4 * np.pi / 3)

            soft_lab = ["Phase0", "Phase1", "Phase2"][np.argmax([s0, s1, s2])]

            tau_list.append(tau_cyc)
            branch_progress_list.append(np.nan)
            wA_list.append(np.nan)
            wB_list.append(np.nan)
            soft_label_list.append(soft_lab)
            continue

        # branch progress
        if tau <= bifurcation_time:
            bp = 0.0
        else:
            bp = (tau - bifurcation_time) / (1.0 - bifurcation_time)
            bp = float(np.clip(bp, 0.0, 1.0))

        # 분기 직후엔 FateA/FateB가 섞이고,
        # 후기로 갈수록 점점 한쪽으로 더 치우치게
        alpha = max(0.25, 4.0 - 3.2 * bp)
        a_bias = rng.beta(alpha, alpha)
        b_bias = 1.0 - a_bias

        wA = bp * a_bias
        wB = bp * b_bias

        if bp < 0.15:
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
        t = crow["time"]
        sid = crow["sample_id"]
        sbranch = crow["sample_branch"]

        # latent trajectory variables
        tau = crow["tau"] if "tau" in cell_meta.columns else t
        branch_progress = (
            crow["branch_progress"]
            if ("branch_progress" in cell_meta.columns and not pd.isna(crow["branch_progress"]))
            else 0.0
        )
        wA = (
            crow["wA"]
            if ("wA" in cell_meta.columns and not pd.isna(crow["wA"]))
            else 0.0
        )
        wB = (
            crow["wB"]
            if ("wB" in cell_meta.columns and not pd.isna(crow["wB"]))
            else 0.0
        )

        cell_sf = rng.lognormal(mean=0.0, sigma=0.20) * library_scale * sample_sf[sid]

        for i, grow in gene_meta.iterrows():
            val = grow["baseline"]
            prog = grow["program"]
            pref = grow["preferred_cell_type"]

            if topology == "linear":
    # linear에서는 sample time보다 per-cell latent time이 더 중요
                tau_lin = tau

                # 연속적인 soft progression weight
                w0 = gaussian_bump(tau_lin, 0.18, 0.22)
                w1 = gaussian_bump(tau_lin, 0.50, 0.22)
                w2 = gaussian_bump(tau_lin, 0.82, 0.22)
                wsum = w0 + w1 + w2 + 1e-8
                w0, w1, w2 = w0 / wsum, w1 / wsum, w2 / wsum

                ct_weight_map = {"CT0": w0, "CT1": w1, "CT2": w2}
                pref_w = ct_weight_map.get(pref, 0.0)
                own_w = ct_weight_map.get(ct, 0.0)

                # 실제 cell identity effect를 추가
                same_ct = float(pref == ct)

                if prog in ("shared_early", "shared_mid", "shared_late"):
                    # time trajectory는 유지하되 기존보다 덜 지배적
                    dyn = gaussian_bump(tau_lin, grow["peak_time"], max(grow["sigma"], 0.13))
                    val += 1.10 * grow["dynamic_amp"] * dyn

                elif prog.startswith("ct") and prog.endswith("marker"):
                    # 여기서 핵심: time 위치 + 실제 cell type identity 둘 다 반영
                    # pref_w만 보지 말고 same_ct를 강하게 넣기
                    val += 0.32 * grow["marker_effect"] * (0.25 + 0.55 * pref_w + 0.55 * same_ct)

                elif prog.startswith("dynamic_ct"):
                    dyn = gaussian_bump(tau_lin, grow["peak_time"], max(grow["sigma"], 0.12))
                    val += 0.42 * grow["dynamic_amp"] * dyn * (0.20 + 0.45 * pref_w + 0.45 * same_ct)
            
            elif topology == "bifurcation":
                prog_w = max(0.0, 1.0 - branch_progress)
                leak = 0.35
                
                if prog == "prog_marker":
                    val += 0.45 * grow["marker_effect"] * (0.65 * prog_w + leak)
                
                elif prog == "fateA_marker":
                    if branch_progress < 0.35:
                        val += 0.05
                    else:
                        val += grow["marker_effect"] * (1.4 * wA)

                elif prog == "fateB_marker":
                    if branch_progress < 0.35:
                        val += 0.05
                    else:
                        val += grow["marker_effect"] * (1.4 * wB)

                elif prog == "dynamic_prog":
                    pre_bump = gaussian_bump(tau, 0.25, max(grow["sigma"], 0.10))
                    val += 0.70 * grow["dynamic_amp"] * pre_bump * (0.75 * prog_w + 0.20)

                elif prog == "dynamic_fateA":
                    post_bump = gaussian_bump(tau, grow["peak_time"], max(grow["sigma"], 0.10))
                    val += 0.70 * grow["dynamic_amp"] * post_bump * (0.75 * wA + 0.15)

                elif prog == "dynamic_fateB":
                    post_bump = gaussian_bump(tau, grow["peak_time"], max(grow["sigma"], 0.10))
                    val += 0.70 * grow["dynamic_amp"] * post_bump * (0.75 * wB + 0.15)

                elif prog == "shared_pre":
                    val += 1.00 * grow["dynamic_amp"] * gaussian_bump(
                        tau, grow["peak_time"], max(grow["sigma"], 0.10)
                    ) * (0.80 * prog_w + 0.25)

                elif prog == "shared_transition":
                    val += 1.80 * grow["dynamic_amp"] * gaussian_bump(
                        tau, bifurcation_time, max(grow["sigma"], 0.14)
                    )
                

            elif topology == "cyclic":
                tau_cyc = tau  # 위에서 cyclic은 already wrapped latent phase

                # phase별 soft weight
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
                    # 전체 원형 trajectory를 만드는 주축
                    val += 1.10 * cyclic_signal(
                        tau_cyc, grow["phase"], amp=grow["dynamic_amp"], period=1.0
                    )

                elif prog in ("phase0_marker", "phase1_marker", "phase2_marker"):
                    # 자기 phase에서 조금 더 높지만 완전 분리되지는 않게
                    val += 0.30 * grow["marker_effect"] * (0.30 + 0.55 * pref_w + 0.45 * same_ct)

                elif prog.startswith("dynamic_phase"):
                    dyn = cyclic_signal(
                        tau_cyc, grow["phase"], amp=grow["dynamic_amp"], period=1.0
                    )
                    # dynamic program도 trajectory를 따르되 약한 phase identity만 부여
                    val += 0.42 * dyn * (0.25 + 0.50 * pref_w + 0.35 * same_ct)

            mu[i, j] = max(val * cell_sf, 1e-5)

    # ---------------------------------------------------------
    # 7) sample counts
    # ---------------------------------------------------------
    if model == "poisson":
        counts = rng.poisson(mu)
    elif model == "negbin":
        lam = rng.gamma(shape=dispersion_theta, scale=mu / dispersion_theta)
        counts = rng.poisson(lam)
    else:
        raise ValueError("model must be 'poisson' or 'negbin'")

    if dropout_rate > 0:
        drop_mask = rng.uniform(size=counts.shape) < dropout_rate
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


def plot_pseudobulk_heatmap(
    pseudobulk_df,
    sample_meta,
    gene_meta=None,
    n_top_genes=120,
    scale_rows=True,
    figsize=(12, 7)
):
    """
    pseudobulk_df: gene x sample
    sample_meta: must contain sample_id, time
    gene_meta: optional
    """

    # sample order by time
    sample_order = sample_meta.sort_values(["time", "replicate"])["sample_id"].tolist()
    mat = pseudobulk_df[sample_order].copy()

    # select variable genes
    gene_var = mat.var(axis=1).sort_values(ascending=False)
    top_genes = gene_var.head(n_top_genes).index
    mat = mat.loc[top_genes]

    # log transform
    mat = np.log1p(mat)

    # row scaling
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
    """
    gene_meta must contain gene_id, program
    """
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

    # sample cells
    if counts_df.shape[1] > n_cells:
        chosen_cells = rng.choice(counts_df.columns, size=n_cells, replace=False)
    else:
        chosen_cells = counts_df.columns.to_numpy()

    sub_meta = cell_meta.set_index("cell_id").loc[chosen_cells].reset_index()
    sub_meta = sub_meta.sort_values(["time", "cell_type"])

    mat = counts_df[sub_meta["cell_id"]].copy()

    # choose top variable genes
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
    plt.xlabel("Cells (ordered by time, cell type)")
    plt.ylabel("Top variable genes")
    plt.tight_layout()
    plt.show()

def prepare_umap_matrix(
    counts_df,
    n_top_genes=1000
):
    """
    returns X (cell x gene), selected_gene_ids
    """
    mat = np.log1p(counts_df)

    gene_var = mat.var(axis=1).sort_values(ascending=False)
    top_genes = gene_var.head(min(n_top_genes, mat.shape[0])).index

    X = mat.loc[top_genes].T.values
    X = StandardScaler().fit_transform(X)

    return X, top_genes.tolist()

def plot_umap_by_time(
    counts_df,
    cell_meta,
    n_top_genes=1000,
    random_state=42,
    figsize=(6, 5)
):
    X, _ = prepare_umap_matrix(counts_df, n_top_genes=n_top_genes)

    reducer = umap.UMAP(
        n_components=2,
        random_state=random_state
    )
    emb = reducer.fit_transform(X)

    plot_df = cell_meta.copy().reset_index(drop=True)
    plot_df["UMAP1"] = emb[:, 0]
    plot_df["UMAP2"] = emb[:, 1]

    plt.figure(figsize=figsize)
    sc = plt.scatter(
        plot_df["UMAP1"],
        plot_df["UMAP2"],
        c=plot_df["time"],
        s=8,
        alpha=0.8
    )
    plt.colorbar(sc, label="time")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title("UMAP colored by time")
    plt.tight_layout()
    plt.show()

def plot_umap_by_branch(
    counts_df,
    cell_meta,
    n_top_genes=1000,
    random_state=42,
    figsize=(6, 5)
):
    X, _ = prepare_umap_matrix(counts_df, n_top_genes=n_top_genes)

    reducer = umap.UMAP(
        n_components=2,
        random_state=random_state
    )
    emb = reducer.fit_transform(X)

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
    n_top_genes=1000,
    random_state=42,
    figsize=(6, 5)
):
    X, _ = prepare_umap_matrix(counts_df, n_top_genes=n_top_genes)

    reducer = umap.UMAP(
        n_components=2,
        random_state=random_state
    )
    emb = reducer.fit_transform(X)

    plot_df = cell_meta.copy().reset_index(drop=True)
    plot_df["UMAP1"] = emb[:, 0]
    plot_df["UMAP2"] = emb[:, 1]

    plt.figure(figsize=figsize)

    for ct in plot_df["cell_type"].unique():
        sub = plot_df[plot_df["cell_type"] == ct]
        plt.scatter(
            sub["UMAP1"],
            sub["UMAP2"],
            s=8,
            alpha=0.8,
            label=ct
        )

    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title("UMAP colored by cell type")
    plt.legend(markerscale=2, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()