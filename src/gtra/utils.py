import numpy as np
import scipy.stats as ss
import networkx as nx

from scipy.sparse import csr_matrix
from soyclustering import SphericalKMeans
from scipy.spatial import distance
from scipy.interpolate import CubicSpline

from collections import Counter

import math
import os
import re

from .preproc import *


def magnitude(v):
    return math.sqrt(sum(v[i]*v[i] for i in range(len(v))))

def normalize(v):
    vmag=magnitude(v)
    return [v[i]/vmag for i in range(len(v))]

def vect_mu(v_list):
    v = np.array(v_list)
    vsum=np.ndarray.sum(v,axis=0)
    return normalize(vsum)


## Calculate jaccard similarity
def Jaccard(l1, l2):
    a = set(l1).union(set(l2))
    b = set(l1).intersection(set(l2))

    return len(b) / len(a), b

## Jaccard similarity distribution
def JS_distribution(obj, tp1, tp2):
    t1 = obj.gene_label_info[tp1]  # list: cellcluster -> list of gene-modules
    t2 = obj.gene_label_info[tp2]

    js_list = []

    for t1_s, mods1 in enumerate(t1):
        if not mods1:  # [] or None
            continue
        for t1_g, gset1 in enumerate(mods1):
            if not gset1:
                continue

            for t2_s, mods2 in enumerate(t2):
                if not mods2:
                    continue
                for t2_g, gset2 in enumerate(mods2):
                    if not gset2:
                        continue

                    sim, inter_genes = Jaccard(gset1, gset2)
                    if len(inter_genes) == 0:
                        continue

                    js_list.append([t1_s, t1_g, t2_s, t2_g, sim])

    return js_list


def JS_threshold_test(obj, tp1, tp2):
    """
    Compute the optimal Jaccard threshold between two timepoints.

    Steps:
      1) Compute JS list
      2) Sweep threshold from 0 to 1 (step 0.01)
      3) Compute connection stats
      4) Min-max scaling
      5) Select threshold with largest |ctc - gtg|
    """
    
    # === 1) JS distribution === #
    js_list = JS_distribution(obj, tp1, tp2)
    if len(js_list) == 0:
        return 0.0

    # Extract JS values only
    js_values = np.array([x[4] for x in js_list])

    # === 2) Pre-sort high → low === #
    sorted_idx = np.argsort(-js_values)
    js_sorted = js_values[sorted_idx]
    list_sorted = np.array(js_list, dtype=object)[sorted_idx]

    # Threshold sweep
    thresholds = np.arange(0.0, 1.01, 0.01)

    ctc_list = []
    gtg_list = []

    for th in thresholds:
        # mask JS >= th
        mask = js_sorted >= th
        selected = list_sorted[mask]

        if len(selected) == 0:
            ctc_list.append(0)
            gtg_list.append(0)
            break

        # === 3) connection statistics === #
        pairs = [(x[0], x[2]) for x in selected]   # (cell1, cell2)
        unique_pairs = set(pairs)

        ctc = len(unique_pairs)
        gtg = len(pairs)

        ctc_list.append(ctc)
        gtg_list.append(gtg)

        if ctc == 0 or gtg == 0:
            break

    # convert to arrays
    ctc_arr = np.array(ctc_list)
    gtg_arr = np.array(gtg_list)
    th_arr  = thresholds[:len(ctc_arr)]

    # === 4) min-max scaling === #
    def scale(x):
        if x.max() == x.min():
            return np.zeros_like(x)
        return (x - x.min()) / (x.max() - x.min())

    ctc_scaled = scale(ctc_arr)
    gtg_scaled = scale(gtg_arr)

    # save for debugging
    obj.ctc_list[tp1] = ctc_scaled
    obj.gtg_list[tp1] = gtg_scaled

    # === 5) gap curve: |ctc - gtg| === #
    gap = np.abs(ctc_scaled - gtg_scaled)

    top2_idx = np.argsort(-gap)[:2]
    best_idx = min(top2_idx) if len(top2_idx) > 1 else top2_idx[0]

    optimal_th = th_arr[best_idx]
    return optimal_th


# Create normalized pseudo-bulk data
def cal_cos_dist(obj,tp1,tp2,t1_s,t2_s,t1_df,t2_df,inter_genes):
    # Load cells
    t1_cells = obj.cell_label_index[tp1][t1_s]
    t2_cells = obj.cell_label_index[tp2][t2_s]
    
    valid_genes = list(
        set(inter_genes) 
        & set(t1_df.index)
        & set(t2_df.index)
    )
    if len(valid_genes) < 2:
        return -1
    
    valid_t1_cells = list(set(t1_cells) & set(t1_df.columns))
    valid_t2_cells = list(set(t2_cells) & set(t2_df.columns))

    if len(valid_t1_cells) == 0 or len(valid_t2_cells) == 0:
        return -1

    # Calculate average of data
    t1_pseudo = t1_df.loc[valid_genes,valid_t1_cells].apply(pd.to_numeric).mean(axis=1)
    t2_pseudo = t2_df.loc[valid_genes,valid_t2_cells].apply(pd.to_numeric).mean(axis=1)
    
    # remove genes with both zeros (optional but consistent with your intention)
    mask = (t1_pseudo + t2_pseudo) != 0
    t1_pseudo = t1_pseudo[mask]
    t2_pseudo = t2_pseudo[mask]
    if t1_pseudo.size < 2:
        return None

    # if either vector becomes all-zero, define max distance
    if np.allclose(t1_pseudo, 0) or np.allclose(t2_pseudo, 0):
        return 1.0

    return float(distance.cosine(t1_pseudo, t2_pseudo))

def get_edge_info(obj, tp1):
    COS_PARAM = 5
    sort_edge_info = sorted(obj.edge_info[tp1], 
                            key=lambda x:x[COS_PARAM], reverse=False)
    
    # Save edge values for tp1's cell type
    edge_info_dict = {}
    source_genes_info = obj.gene_label_info[tp1]
    for c in range(len(source_genes_info)):
        for edge in sort_edge_info:
            if edge[0] == c:
                if c not in edge_info_dict:
                    edge_info_dict[c] = [edge[1:]]
                else:
                    edge_info_dict[c].append(edge[1:])
    return edge_info_dict


def cal_rank(edge_weight, sw, dw):
    edge_weight = np.array(edge_weight)
    candidate_n = len(edge_weight) + 1
    
    SIM_PARAM, COS_PARAM = 3, 4
    sim_rank = candidate_n - ss.rankdata(edge_weight[:,SIM_PARAM], method="min")
    cos_rank = ss.rankdata(edge_weight[:,COS_PARAM], method="min")
    rank_sum = ((sw*sim_rank)+(dw*cos_rank))/2
    
    # save rank test results
    conversion = []
    for i in range(len(edge_weight)):
        origin = list(edge_weight[i])
        origin.extend([round(list(rank_sum)[i],3)])
        conversion.append(origin)
    
    return conversion

# Check standard path
def check_standard_path(obj, tp1, t1_s, top_rank, einfo, einfo_results):
    # Paths that match the source cell type
    source_ct = get_unique_celltype(obj, tp1, t1_s)
    
    # Load standard path info
    fname = obj.params.answer_path_dir
    
    
    path_pvals = obj.pval_df.copy()
    sig_paths = path_pvals[(path_pvals["Interval"]==tp1)]
    sig_paths = sig_paths[(sig_paths["adj_p-value"]<=obj.pval_th)]
    answers = sig_paths[sig_paths["source"]==source_ct]['target'].values.tolist()

    if os.path.isfile(fname):
        obj.answer_path = pd.read_csv(fname, sep=",")
        paths = obj.answer_path.copy()
        answer_input = sum(paths.loc[paths['source']==source_ct,
                                ['target']].values.tolist(),[])
        answers = list(set(answer_input).intersection(set(answers))).copy()
        
    
    # Filter incorrected paths among paths
    for r in range(top_rank):
        t2_s = int(einfo[r,:3][1])
        target_ct = get_unique_celltype(obj, tp1+1, t2_s)
        
        # If target cell type exist not in answer path then continue
        if target_ct not in answers: continue
        
        label_info = list(map(int,einfo[r,:3]))
        score_info = list(einfo[r,3:])
        einfo_results.append([t1_s]+label_info+score_info)
        
    return einfo_results

# Convert label name
def _conv_label(label):
    t, c, g = map(int, re.findall(r'\d+', label))
    return t, c, g

def _get_inter_genes(obj, path):
    t, c, g = _conv_label(path[0])
    inter_genes = set(obj.gene_label_info[t][c][g])
    
    for node in path[1:]:
        t, c, g = _conv_label(node)
        inter_genes &= set(obj.gene_label_info[t][c][g])
        
        if not inter_genes:
            break
    
    return inter_genes


# Create candidate paths
def get_networks(obj, sub_graphs, inter_gene_th=10):
    networks = []
    
    for sub in sub_graphs:
        nodes = sorted(sub.nodes(), key=lambda x: float(re.findall(r'\d+', x)[0]))
        
        first_time = nodes[0].split('_')[0]
        last_time = nodes[-1].split('_')[0]
        
        sources = [n for n in nodes if n.startswith(first_time)]
        targets = [n for n in nodes if n.startswith(last_time)]
        
        for s in sources:
            for t in targets:
                for path in nx.all_simple_paths(sub, source=s, target=t):
                    inter_genes = _get_inter_genes(obj, path)
                    
                    if len(inter_genes) > inter_gene_th:
                        networks.append(path)
                        obj.path_gene_sets[tuple(path)] = inter_genes
    
    return networks

## -------------- Helper function for step 3 -------------- ##
# Constructing cell type-specific trajectory
def group_cell_type_trajectory(net_info):
    merge_path_dict = {}
    for path in net_info:
        cell_type_path = [node[:node.rfind('_')] for node in path]
        key = tuple(cell_type_path)

        if merge_path_dict.get(key) is None:
            merge_path_dict[key] = [path]
        else:
            merge_path_dict[key].append(path)

    return merge_path_dict

# Concat expression matrix
def merge_expr(obj, path):
    """
    Merge time-series expression matrix along a given answer path.

    For each node in the path, gene expression is averaged across
    cells belonging to the corresponding cell type at that time point.
    Genes with zero expression across all time points are removed.
    """

    # --- genes involved in this path ---
    path_genes = list(obj.path_gene_sets[tuple(path)])

    expr_list = []
    times = []

    for node in path:
        # --- parse node label (fast & safe) ---
        # expected format: t{time}_{celltype}_{cluster}
        tname, cname, _ = map(int, re.findall(r'\d+', node))

        # --- cell indices ---
        cnames = obj.cell_label_index[tname][cname]

        # --- expression matrix (genes × cells) ---
        expr_mat = obj.tp_data_dict[tname].to_df().T

        # --- subset & average ---
        expr_mean = expr_mat.loc[path_genes, cnames].mean(axis=1)

        expr_list.append(expr_mean)
        times.append(f"t{tname}")

    # --- concatenate once ---
    expr_df = pd.concat(expr_list, axis=1)
    expr_df.columns = times
    expr_df.index.name = "GeneID"

    # --- remove genes with all-zero expression ---
    expr_df = expr_df.loc[(expr_df != 0).any(axis=1)]

    return expr_df



# L2 normalization
def l2norm(dat):
    norm = np.sqrt(np.sum(np.square(dat), axis=1))
    norm = np.array(norm).reshape((-1, 1))
    norm = dat / norm
    return norm


# Calculate interval confidence
def cal_ic(df):
    n = len(df) # freedom
    std_err = np.std(df.to_numpy()) / n**0.5 # std error
    ic = ss.t.interval(0.95, n, list(df.mean().values.real), scale=std_err)
    return ic


def elbow_method(dat, k_min=2, k_max=10, random_state=25):
    """
    Estimate the optimal number of clusters using the elbow method
    with Spherical K-Means.

    Parameters
    ----------
    dat : array-like (n_samples × n_features)
        Input data matrix.
    k_min : int
        Minimum number of clusters to test.
    k_max : int
        Maximum number of clusters to test.
    random_state : int
        Random seed for reproducibility.
    """

    # --- normalize once ---
    X = csr_matrix(l2norm(dat))

    ks = np.arange(k_min, k_max + 1)
    inertia = np.empty(len(ks))

    # --- fit models ---
    for i, k in enumerate(ks):
        spk = SphericalKMeans(
            n_clusters=k,
            random_state=random_state,
            max_iter=100
        )
        spk.fit(X)
        inertia[i] = spk.inertia_

    # --- trivial / degenerate case ---
    if len(inertia) <= 2 or np.allclose(inertia, inertia[0]):
        return k_min

    # --- elbow detection: largest relative drop ---
    deltas = np.diff(inertia)
    rel_drop = -deltas / inertia[:-1]   # 상대 감소율

    optimal_idx = np.argmax(rel_drop)
    optimal_k = ks[optimal_idx + 1]

    return int(optimal_k)


def pattern_filtering(obj, deviation_th=0.2):
    """
    Filter time-series gene expression patterns based on
    centroid consistency across time points.

    Patterns with large deviation from the centroid profile
    are removed.
    """

    filtered_patterns = {}

    for key, pattern in obj.merge_pattern_dict.items():

        # --- cheap check first ---
        if pattern.shape[1] != obj.tp_data_num:
            continue

        # --- normalize ---
        pt_df = l2norm(pattern)

        # --- centroid (mean profile across genes) ---
        centroid = pt_df.mean(axis=0).to_numpy()

        # --- intra-cluster deviation ---
        # cal_ic returns (something, ic_profile)
        _, ic_profile = cal_ic(pt_df)

        max_dev = np.max(np.abs(ic_profile - centroid))

        if max_dev <= deviation_th:
            filtered_patterns[key] = pattern

    return filtered_patterns


from collections import defaultdict
# Calculate pearson correlation
def cal_corr(mp_dict, ptc, pcut=0.05):
    candidate_pair = set()
    # Pattern comparison
    for i,s in enumerate(ptc):
        for j,t in enumerate(ptc):
            if i>=j: continue
            source = l2norm(mp_dict[s]).mean().values
            target = l2norm(mp_dict[t]).mean().values
            if ((np.isnan(source)) | (np.isnan(target))).any(): continue
            stat, pvals = ss.pearsonr(source, target)
            if (pvals>pcut) or (stat<=0): continue
            # Save candidate pairs
            pairs = tuple(sorted((s,t),key=lambda x:int(x.replace('_',''))))
            candidate_pair.add(pairs)
    
    return candidate_pair

def get_candidate_keys(keys):
    groups = defaultdict(list)
    for k in keys:
        group = k.split('_', 1)[0]
        groups[group].append(k)
    return dict(groups)

def select_patterns(ptc, mp_dict, sub_nets, key, th=0.2):
    """
    Merge correlated pattern groups and keep unmerged patterns.
    """
    pt_dict = {}

    ptc_set = set(ptc)
    merged_keys = set()

    for i, nodes in enumerate(sub_nets):
        nodes = list(nodes)

        # --- collect patterns once ---
        mats = [mp_dict[n] for n in nodes]
        m = pd.concat(mats, axis=0)

        # --- normalize once ---
        m_norm = l2norm(m)
        centroid = m_norm.mean(axis=0).to_numpy()
        _, ic_profile = cal_ic(m_norm)

        max_dev = np.max(np.abs(ic_profile - centroid))
        if max_dev > th:
            continue

        merged_keys.update(nodes)
        pt_dict[f"{key}_M{i}"] = m.drop_duplicates()

    # --- keep unmerged patterns ---
    for k in ptc_set - merged_keys:
        pt_dict[k] = mp_dict[k].copy()

    return pt_dict


def merge_sim_patterns(obj):
    mp_dict = obj.merge_pattern_dict
    candidate_keys = get_candidate_keys(mp_dict.keys())

    new_mp_dict = {}

    for key, ptc in candidate_keys.items():
        pairs = cal_corr(mp_dict, ptc)

        if not pairs:
            for k in ptc:
                new_mp_dict[k] = mp_dict[k]
            continue

        pair_net = pd.DataFrame(pairs, columns=['s', 't'])
        g = nx.from_pandas_edgelist(pair_net, 's', 't', create_using=nx.Graph())

        sub_nets = [
            list(c) for c in nx.connected_components(g)
        ]

        merged = select_patterns(ptc, mp_dict, sub_nets, key)
        new_mp_dict.update(merged)

    return new_mp_dict


# Renaming pattern name
def renaming_pattern_id(mp_dict):
    unique_path = np.unique([i[:i.find('_')] for i in mp_dict.keys()])
    for path in unique_path:
        candidates = [k for k in mp_dict.keys() if k[:k.find('_')] == path]
        for i, key in enumerate(candidates):
            mp_dict[f'{path}_{i}'] = mp_dict.pop(key)
    # Sorting pattenr id
    mp_dict = dict(sorted(mp_dict.items(), key=lambda x:int(x[0][:x[0].find('_')])))
    return mp_dict


# Save pattern centroid
def save_pattern_centroid(obj):
    pt_csv_dict = {}
    for key, df in obj.merge_pattern_dict.items():
        mean_val = l2norm(df).mean().values
        pt_csv_dict[key] = mean_val
    
    pt_csv_df = pd.DataFrame(pt_csv_dict).T
    pt_csv_df.columns = ['T'+str(i+1) for i in range(len(pt_csv_df.columns))]
    pt_csv_df.index.name = 'Key'

    output_name = f'{obj.params.output_dir}/{obj.params.output_name}_pattern_centroid.csv'
    os.makedirs(f'{obj.params.output_dir}', exist_ok=True)
    pt_csv_df.to_csv(output_name)


## Make gene set data frame for cell trajectory
def make_gene_set_frame(idx, gene_set, pt_df, key, convert_name):
    if idx == 0:
        gene_set = pd.DataFrame(list(pt_df.index), columns=[convert_name + '[' + key + ']'])
    else:
        tmp = pd.DataFrame(list(pt_df.index), columns=[convert_name + '[' + key + ']'])
        gene_set = pd.concat([gene_set, tmp], axis=1)
    return gene_set


## Renaming cell-state trajectories
def convert_path_name(obj, key):
    key_label = key[:key.find('_')]
    merge_path_keys = list(obj.merge_path_dict.keys())

    # Convert path name to cell type name
    convert_name = ''
    for _, path in enumerate(merge_path_keys[int(key_label)]):
        cluster_label = int(path[path.find('_') + 1:])
        time_label = int(path[path.find('t') + 1: path.find('_')])
        cells = obj.cell_label_index[time_label][cluster_label]
        cname = obj.params.cell_type_label
        
        unique_cell_type = dict(Counter(obj.cell_type_info.loc[cells,cname].values.tolist()))
        cell_type = sorted(unique_cell_type.items(), key=lambda x: x[1], reverse=True)[0][0]
        convert_name += cell_type + '->'

    return convert_name[:convert_name.rfind('->')]


# Smoothing the pattern
def spline_func(x, y):
    f = CubicSpline(x, y, bc_type='natural')
    x_new = np.linspace(0, len(x)-1, 100)
    y_new = f(x_new)
    return x_new, y_new

# Pattern interpolation based on a confidence interval
def interval_spline(x, y):
    x_new = np.linspace(0, len(x)-1, 100)

    pos_f = CubicSpline(x, y[0], bc_type='natural')
    neg_f = CubicSpline(x, y[1], bc_type='natural')
    
    pos_y = pos_f(x_new)
    neg_y = neg_f(x_new)
    return pos_y, neg_y

# Plotting gene expression patterns
def plotting_patterns(pt_df, key, start_cells, ax, pos):
    # Cubic spline function version
    import seaborn as sns
    colors = sns.color_palette('colorblind', n_colors=9)

    cubic_x = [i for i in range(len(pt_df.columns))]
    cubic_y = list(pt_df.mean().values)
    x_new, y_new = spline_func(cubic_x, cubic_y)

    # Calculation interval confidence    
    ic = cal_ic(pt_df)
    pos_y, neg_y = interval_spline(cubic_x, ic)

    row = pos // 3
    col = pos % 3

    # ax[row][col].plot(x_new, y_new, color='r', linewidth=0.8, linestyle='dashed')
    ax[row][col].plot(x_new, y_new, color=colors[row], linewidth=2)
    # ax[row][col].plot(cubic_x, cubic_y, 'ro')
    ax[row][col].scatter(cubic_x, cubic_y, color=colors[row], edgecolor='black', s=70, zorder=3)

    # ax[row][col].plot(x_new, pos_y, linestyle='dashed', color='gray', linewidth=0.8)
    # ax[row][col].plot(x_new, neg_y, linestyle='dashed', color='gray', linewidth=0.8)
    ax[row][col].fill_between(x_new, pos_y, neg_y, color=colors[row], alpha=0.15)

    ax[row][col].grid(False)


    ax[row][col].set_xlabel(f'Time points', size=10)
    ax[row][col].set_ylabel('Normalized gene expression', size=10)
    ax[row][col].set_xticks(cubic_x, labels=pt_df.columns, rotation=45)
    ax[row][col].set_title(f'Start:{start_cells}, CL: {key}\n (N={len(pt_df)})')
    

def extract_time_and_celltype(label):
    t, cell = label.split(": ")
    return int(t[1:]), cell


def rgb01_to_rgbstr(c):
    r, g, b = c
    return f"rgb({int(round(r*255))},{int(round(g*255))},{int(round(b*255))})"


def time_ctconvert_majority(obj, time):
    ct_col = obj.params.cell_type_label
    cl_col="cluster_label"
    df = obj.tp_data_dict[time].obs[[cl_col, ct_col]].copy()
    out = {}
    for cl, sub in df.groupby(cl_col):
        out[int(cl)] = Counter(sub[ct_col]).most_common(1)[0][0]
    return out


def build_path_gn_table(obj):
    path_gene_dict = {}
    for _, paths in obj.merge_path_dict.items():
        for path in paths:
            for i in range(len(path) - 1):
                src = path[i]
                tar = path[i + 1]

                st, sc, sg = src.split('_')
                tt, tc, tg = tar.split('_')

                stime = int(st[1:])
                ttime = int(tt[1:])

                sc, sg = int(sc), int(sg)
                tc, tg = int(tc), int(tg)

                sgenes = obj.gene_label_info[stime][sc][sg]
                tgenes = obj.gene_label_info[ttime][tc][tg]

                sr = src[:src.rfind('_')]  # tX_cellcluster
                ta = tar[:tar.rfind('_')]
                key = f"{sr}|{ta}"

                inter = set(sgenes).intersection(set(tgenes))
                if not inter:
                    continue

                if key not in path_gene_dict:
                    path_gene_dict[key] = []
                path_gene_dict[key].extend(list(inter))
    rows = []
    for key, genes in path_gene_dict.items():
        src, tar = key.split('|')
        stok = src.split('_')  # ["t0","3"]
        ttok = tar.split('_')  # ["t1","5"]

        st, sc = int(stok[0][1:]), int(stok[1])
        tt, tc = int(ttok[0][1:]), int(ttok[1])

        scc = time_ctconvert_majority(obj, st)
        tcc = time_ctconvert_majority(obj, tt)

        src_ct = scc.get(sc, str(sc))
        tar_ct = tcc.get(tc, str(tc))

        rows.append([st, src_ct, tar_ct, len(set(genes))])

    path_gn = pd.DataFrame(rows, columns=["Interval", "source", "target", "GN"])
    return path_gn

def build_sankey_df_from_pvals(obj, min_gn=1, pval_col="adj_p-value", pval_th=None):
    path_gn = build_path_gn_table(obj)

    pval_df = obj.pval_df.copy()
    if pval_th is not None:
        pval_df = pval_df[pval_df[pval_col] <= pval_th].copy()

    df = pval_df.merge(path_gn, on=["Interval","source","target"], how="left")
    df["GN"] = df["GN"].fillna(0).astype(int)
    df = df[df["GN"] >= min_gn].copy()
    return df

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import pearsonr
from scipy.stats import friedmanchisquare

def detect_expression_trend(expression_vector):
    """
    expression_vector: list or array of average expression across timepoints
    """
    x = np.array(expression_vector)
    t = np.arange(len(x))
    
    # Normalize for stability
    x_norm = (x - np.mean(x)) / (np.std(x) + 1e-6)

    # Check increasing/decreasing using correlation with time
    corr, _ = pearsonr(x_norm, t)
    
    if corr > 0.8:
        return "increasing"
    elif corr < -0.8:
        return "decreasing"
    elif np.std(x) < 0.05:
        return "flat"
    else:
        return "transient"
    
def get_sig_patterns(obj, pval_th=1e-2):
    # Prefer the in-memory {key -> trajectory} map populated by plot_patterns();
    # fall back to the *_pattern_genes.csv it writes (e.g. when loading a saved
    # object in a fresh session).
    key_map = getattr(obj, "pattern_key_map", None)
    if key_map:
        pt_key_dict = dict(key_map)
    else:
        csv_path = f'{obj.params.output_dir}/{obj.params.output_name}_pattern_genes.csv'
        if not os.path.isfile(csv_path):
            raise RuntimeError(
                "No pattern set available: call plot_patterns() before "
                "module_evaluation() (or ensure the *_pattern_genes.csv exists)."
            )
        pt = pd.read_csv(csv_path, index_col=0)
        pt_key_dict = {i[i.find('[')+1:i.find(']')]: i[:i.find('[')] for i in pt.columns}
    pattern_data = obj.merge_pattern_dict.copy()
    sig_pt_dat = {i: pattern_data[i] for i in pt_key_dict.keys()}
    
    pattern_stats = []

    for pattern_id, df in sig_pt_dat.items():
        df = l2norm(df)

        # timepoint 컬럼 자동 추출: t0, t1, t2 ... (필요하면 규칙만 바꾸면 됨)
        tp_cols = [c for c in df.columns if str(c).startswith("t")]
        # 숫자 정렬 (t10이 t2 앞에 오는 문제 방지)
        def _tp_key(c):
            s = str(c)
            num = "".join(ch for ch in s[1:] if ch.isdigit())
            return int(num) if num else 10**9
        tp_cols = sorted(tp_cols, key=_tp_key)

        # timepoint가 3개 이상이어야 Friedman 의미 있음 (2개면 test 불가)
        if len(tp_cols) < 3:
            continue

        # trend 계산도 timepoint 수에 맞춰 자동
        avg_expr = df[tp_cols].values.mean(axis=0)
        trend = detect_expression_trend(avg_expr)

        # Friedman test: 컬럼들을 리스트로 모아서 *args로 전달
        arrays = [df[c].to_numpy() for c in tp_cols]
        f_stat, p_value = friedmanchisquare(*arrays)

        log_p = -np.log10(p_value) if (p_value is not None and p_value > 0) else np.inf
        trj = pt_key_dict.get(pattern_id, None)

        genes = ";".join(map(str, df.index))

        pattern_stats.append({
            "Pattern_ID": pattern_id,
            "Trend": trend,
            "F_statistic": f_stat,
            "p_value": p_value,
            "-log10(p_value)": log_p,
            "trajectory": trj,
            "nGenes": len(df.index),
            "nTimepoints": len(tp_cols),
            "Timepoints": ",".join(tp_cols),
            "Genes": genes,
        })

    result_df = pd.DataFrame(pattern_stats)
    result_df = result_df[result_df["p_value"] < pval_th].sort_values("p_value")
    return result_df
    

import scipy.stats as stats
from .cluster_func import _calc_gap

def pattern_eval(obj):
    res_df = get_sig_patterns(obj, pval_th=1e-3)
    pt_dist = []
    for i in res_df['Pattern_ID']:
        expr1 = l2norm(obj.merge_pattern_dict[i])
        avg_expr1 = expr1.mean(axis=0).values
        tmp=[]
        for j in res_df['Pattern_ID']:
            expr2 = l2norm(obj.merge_pattern_dict[j])
            avg_expr2 = expr2.mean(axis=0).values
            
            rho, p_val = stats.pearsonr(avg_expr1, avg_expr2)
            tmp.append(rho)
        pt_dist.append(tmp)

    pt_dist = pd.DataFrame(pt_dist, index=res_df["trajectory"], columns=res_df["trajectory"])
    
    linked = linkage(pt_dist, "ward")
    K = _calc_gap(linked)
    clusters = fcluster(linked, K, criterion="maxclust")

    pt_df = res_df[["Pattern_ID","trajectory"]].copy()
    pt_df["cluster"] = clusters
    
    obj.sig_patterns = res_df
    obj.pattern_dist = pt_dist
    obj.module_df = pt_df
