"""COVID patient-level concordance + group-slope analysis (revision metric).

Two metrics on the DP+/RP- module (genes UP over time in DP and DOWN over time
in RP CD8+T/NK cells = the manuscript's opposite-trajectory intersection):

  MAIN  rho_p  : per-patient Spearman(module score, clinical severity) over the
                 patient's sampling timepoints; one-sample Wilcoxon rho>0.
  GROUP beta_p : per-patient temporal slope of module score; DP (expect >0) vs
                 RP (expect <0) by Mann-Whitney U  (the core opposite-trajectory
                 claim, severity-free).

Severity from /data2/.../severity_score/<pid>.csv (daily WHO & NEWS), sampled at
clinical.csv sampling days. T/NK cells + day labels from covid_obj/*.dill.
"""
import glob, warnings
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp, dill
from scipy.stats import spearmanr, wilcoxon, mannwhitneyu, linregress
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, seaborn as sns

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
SEV = "/data2/NIH_COVID_19/clinical/severity_score"
OUT = HERE / "CONCORDANCE_figs"; OUT.mkdir(exist_ok=True)
TNK = {"CD8+T", "NK"}


def lognorm(ad):
    X = ad.X.toarray() if sp.issparse(ad.X) else np.asarray(ad.X)
    return np.log1p(X / (X.sum(1, keepdims=True) + 1e-12) * 1e4)


def patient_tnk_pseudobulk(obj):
    """genes x timepoints pseudobulk of CD8T/NK cells."""
    cols = []
    for tp in range(obj.tp_data_num):
        ad = obj.tp_data_dict[tp]
        m = ad.obs["mye_sub"].astype(str).isin(TNK).values
        Xln = lognorm(ad)
        cols.append(Xln[m].mean(0) if m.sum() else np.full(ad.n_vars, np.nan))
    return np.array(cols).T, list(map(str, obj.tp_data_dict[0].var_names))  # genes x tp


def main():
    cl = pd.read_csv(HERE / "clinical.csv", index_col=0)
    pname2pid = dict(cl[["pname", "pid"]].values)
    pheno = dict(cl[["pname", "pheno"]].values)
    dates = {r["pname"]: [int(d.replace("Day", "")) for d in r["dates"].split("|")]
             for _, r in cl.iterrows()}

    # ---- per-patient T/NK pseudobulk over sampling timepoints ----
    pb, common = {}, None
    for p in sorted(glob.glob("../../../../covid_obj/*.dill")):
        pn = Path(p).stem.split("_")[0]
        o = dill.load(open(p, "rb"))
        M, g = patient_tnk_pseudobulk(o)
        pb[pn] = pd.DataFrame(M, index=g)
        common = set(g) if common is None else (common & set(g))
        del o
    common = sorted(common)

    # ---- DP+/RP- module: DP-up INTERSECT RP-down (mean per-gene slope across patients) ----
    def gene_slopes(group):
        S = []
        for pn in group:
            Mtp = pb[pn].loc[common].values  # genes x tp
            x = np.arange(Mtp.shape[1])
            sl = np.array([linregress(x, row).slope if np.std(row) > 0 else 0.0 for row in Mtp])
            S.append(sl)
        return np.nanmean(S, 0)
    DP = [pn for pn in pb if pheno[pn] == "DP"]; RP = [pn for pn in pb if pheno[pn] == "RP"]
    dp_slope = gene_slopes(DP); rp_slope = gene_slopes(RP)
    cg = np.array(common)
    DPplus = set(cg[dp_slope > np.percentile(dp_slope, 80)])
    RPminus = set(cg[rp_slope < np.percentile(rp_slope, 20)])
    module = sorted(DPplus & RPminus)
    print(f"DP+ genes={len(DPplus)} RP- genes={len(RPminus)}  DP+ ∩ RP- module = {len(module)} genes", flush=True)

    # ---- score + severity per patient ----
    rows = []
    for pn in pb:
        pid = pname2pid[pn]
        try:
            sv = pd.read_csv(f"{SEV}/{pid}.csv", index_col=0)
        except Exception:
            sv = None
        Mtp = pb[pn].loc[module].values  # module genes x tp
        score = np.nanmean(Mtp, 0)       # module score per timepoint
        days = dates[pn]
        who, news = [], []
        for d in days:
            if sv is not None and d - 1 < len(sv):
                who.append(sv.iloc[d - 1]["WHO"]); news.append(sv.iloc[d - 1]["NEWS"])
            else:
                who.append(np.nan); news.append(np.nan)
        x = np.arange(len(score))
        beta = linregress(x, score).slope
        # concordance over timepoints with severity
        def rho(sev):
            sev = np.array(sev, float); ok = ~np.isnan(sev)
            return spearmanr(score[ok], sev[ok])[0] if ok.sum() >= 3 else np.nan
        rows.append({"patient": pn, "pheno": pheno[pn], "n_tp": len(score),
                     "beta": beta, "rho_WHO": rho(who), "rho_NEWS": rho(news),
                     "n_sev": int((~np.isnan(who)).sum())})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "concordance.csv", index=False)
    print(df.round(3).to_string(index=False), flush=True)

    # ---- tests ----
    rW = df["rho_WHO"].dropna()
    if len(rW) >= 3:
        print(f"\n[MAIN] per-patient Spearman(score,WHO) >0?  median rho={rW.median():.3f}, "
              f"n={len(rW)}, Wilcoxon p={wilcoxon(rW, alternative='greater')[1]:.3f}")
    dpb = df[df.pheno == "DP"]["beta"]; rpb = df[df.pheno == "RP"]["beta"]
    u, pmw = mannwhitneyu(dpb, rpb, alternative="greater")
    print(f"[GROUP] slope DP(median {dpb.median():.3f}) vs RP(median {rpb.median():.3f})  "
          f"Mann-Whitney p={pmw:.3f}")

    # ---- figure ----
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    sns.boxplot(data=df, x="pheno", y="beta", order=["DP", "RP"], ax=ax[0])
    sns.stripplot(data=df, x="pheno", y="beta", order=["DP", "RP"], color="0.2", size=7, ax=ax[0])
    ax[0].axhline(0, ls="--", c="grey"); ax[0].set_title(f"Module-score slope (DP vs RP)\nMann-Whitney p={pmw:.3f}")
    m = df.melt(id_vars=["patient", "pheno"], value_vars=["rho_WHO", "rho_NEWS"], var_name="severity", value_name="rho").dropna()
    if len(m):
        sns.stripplot(data=m, x="severity", y="rho", hue="pheno", size=8, ax=ax[1])
        ax[1].axhline(0, ls="--", c="grey"); ax[1].set_ylim(-1, 1)
    ax[1].set_title("Per-patient score–severity concordance (rho)")
    fig.tight_layout(); fig.savefig(OUT / "CONC1_concordance_slope.pdf", dpi=200)
    print("\nsaved CONCORDANCE_figs/CONC1_concordance_slope.pdf\nDONE")


if __name__ == "__main__":
    main()
