import numpy as np
import pandas as pd
import scanpy as sc
import leidenalg
import igraph as ig
import copy

import random
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


import seaborn as sns
import matplotlib.colors as mcolors


from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage, fcluster

from kneed import KneeLocator

from joblib import Parallel, delayed

from .core import *
from .preproc import *

# Get label index
def _get_label(labels, names, K):
    label_index = [[] for _ in range(K)]
    for name, lbl in zip(names, labels):
        label_index[lbl].append(name)
    return label_index


## --------------- Cell clustering --------------- ##
# Cell type labeling
def add_annotation(obj, time, do_filter=True):
    adata = obj.tp_data_dict[time]
    adata.layers["raw"] = adata.X.copy()

    cname = adata.obs.columns[0]
    obj.params.cell_type_label = cname

    # 1) Filter out low-count cell types (optional)
    counts = adata.obs[cname].value_counts()
    if do_filter:
        valid_ct = counts[counts >= obj.params.filter_cell_n].index
    else:
        valid_ct = counts.index  # keep all observed CTs

    adata = adata[adata.obs[cname].isin(valid_ct)].copy()

    # 2) Convert cell types into integer labels
    unique_ct = sorted(valid_ct)
    ct2id = {ct: i for i, ct in enumerate(unique_ct)}
    clabel = adata.obs[cname].map(ct2id).astype(int).tolist()

    adata.obs["cluster_label"] = clabel
    K = len(unique_ct)

    cells = adata.obs_names.tolist()
    label_index = _get_label(clabel, cells, K)

    return adata, label_index, K, clabel



# Graph-based cell clustering using leiden
def cell_graph_clustering(obj, time):
    adata = obj.tp_data_dict[time]
    if "raw" not in adata.layers:
        adata.layers["raw"] = adata.X.copy()

    # 1) Normalize -> log -> scale
    if "norm" not in adata.layers:
        adata.layers["norm"] = adata.X.copy()
        sc.pp.normalize_total(adata, layer="norm")
        sc.pp.log1p(adata, layer="norm")
        
    if 'scaled' not in adata.layers:
        adata.layers['scaled'] = adata.layers['norm'].copy()
        sc.pp.scale(adata, max_value=10, layer="scaled")

    # 2) PCA + neighbors + Leiden
    adata.X = adata.layers["scaled"]
    sc.tl.pca(adata, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=obj.params.cn_neighbors, use_rep="X_pca")
    sc.tl.leiden(adata, resolution=obj.params.cn_cluster_resolution)
    
    # 3) Cluster labeling
    adata.X = adata.layers["raw"]
    
    clabel = adata.obs["leiden"].astype(int)
    adata.obs["cluster_label"] = clabel

    K = adata.obs["cluster_label"].nunique()
    cells = adata.obs_names.tolist()
    label_index = _get_label(clabel, cells, K)

    return adata, label_index, K, clabel


def create_color_dict(obj):
    all_cts = []
    ct_label = obj.params.cell_type_label
    for _, dat in obj.tp_data_dict.items():
        all_cts.extend(dat.obs[ct_label].unique())
    
    unique_cts = sorted(set(all_cts))
    palette = sns.color_palette('Set2',len(unique_cts))
    celltype_colors = dict(zip(unique_cts, palette))
    obj.celltype_colors = celltype_colors


# Cell clustering for each time point
def cell_clustering(obj, time):
    if obj.params.label_flag:
        adata, cli, K, clabel = add_annotation(obj, time, do_filter=True)
    else:
        adata, cli, K, clabel = cell_graph_clustering(obj, time)
    
    obj.tp_data_dict[time] = adata
    obj.cell_optimal_k[time] = K
    obj.cell_cluster_label[time] = clabel
    obj.cell_label_index[time] = cli


## --------------- Gene clustering --------------- ##
def knn_based_gene_clustering(
    X,
    obs,
    target_cluster,
    gene_names=None,
    cell_label_col="cluster_label",
    min_cells=10,
    n_pcs_max=30,
    k_gene_max=15,
    res_list=(0.2, 0.4, 0.6),
    seed=1234,
    force_split_single_cluster=False,
):
    """
    PCA + cosine-kNN + Leiden gene clustering within a single cell cluster.
    Returns: list of gene clusters (each cluster is a list of genes).
    """

    # 1) Select cells in the target cluster
    mask = (obs[cell_label_col].values == target_cluster)
    n_cells = int(mask.sum())
    if n_cells < min_cells:
        return []

    X_sub = X[mask]  # (n_cells, n_genes)
    if not isinstance(X_sub, np.ndarray):
        X_sub = X_sub.toarray()

    # genes x cells
    G = X_sub.T
    n_genes = G.shape[0]

    if gene_names is not None and len(gene_names) != n_genes:
        raise ValueError(
            f"gene_names length ({len(gene_names)}) != n_genes ({n_genes})."
        )

    # 2) log2(x+1) + gene-wise z-score
    G_log = np.log2(G + 1.0)
    mu = G_log.mean(axis=1, keepdims=True)
    sd = G_log.std(axis=1, keepdims=True) + 1e-8
    G_final = (G_log - mu) / sd
    G_final = np.nan_to_num(G_final)

    # 3) PCA embedding (genes as observations)
    n_components = min(n_pcs_max, n_cells - 1, n_genes - 1)
    if n_components < 2:
        return []

    pca = PCA(n_components=n_components, random_state=0)
    G_pca = pca.fit_transform(G_final)  # (n_genes, n_components)

    # 4) gene-gene kNN graph (k based on n_genes!)
    k = min(k_gene_max, n_genes - 1)
    if k < 2:
        return []

    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(G_pca)
    knn_graph = nn.kneighbors_graph(G_pca, mode="connectivity")

    # symmetrize for undirected Leiden
    knn_graph = knn_graph.maximum(knn_graph.T)
    src, tar = knn_graph.nonzero()
    if len(src) == 0:
        return []

    g = ig.Graph(n=n_genes, edges=list(zip(src.tolist(), tar.tolist())), directed=False)

    # 5) Leiden: choose best modularity across resolutions
    best_part, best_mod = None, -np.inf
    for res in res_list:
        part = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=float(res),
            seed=seed,
        )
        if part.modularity > best_mod:
            best_mod, best_part = part.modularity, part

    if best_part is None:
        return []

    labels = best_part.membership

    # Optional fallback (not recommended for bootstrap stability)
    if force_split_single_cluster and len(set(labels)) == 1:
        part = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=float(max(res_list)) * 2.0,
            seed=seed,
        )
        labels = part.membership

    # 6) Convert to list-of-gene-lists
    K = max(labels) + 1
    glabel_idx = [[] for _ in range(K)]
    for gi, lbl in enumerate(labels):
        glabel_idx[lbl].append(gene_names[gi] if gene_names is not None else gi)

    # remove empties
    glabel_idx = [grp for grp in glabel_idx if len(grp) > 0]
    return glabel_idx


## --------------- bootstrapping based clustering --------------- ##


# Create object for bootstrapping
def _extract_dat(obj):
    return {
        'tp_data_dict':obj.tp_data_dict,
        'tp_data_num':obj.tp_data_num,
        'genes':obj.genes,
        'params':obj.params
    }

# Build mini GTra's object
def _build_boot_obj(dat):
    from .core import GTraObject
    obj = GTraObject() # type: ignore
    obj.tp_data_dict = dat["tp_data_dict"]
    obj.tp_data_num = dat["tp_data_num"]
    obj.genes = dat["genes"]
    obj.params = dat["params"]
    return obj

def _create_random_cells(obj, time, frac=0.8, min_keep=None):
    """
    Randomly subsample cells within each (annotated) cell type, but
    guarantee a minimum number of retained cells per type to prevent
    disappearing cell types in bootstrap runs.

    min_keep:
      - If None, uses obj.params.filter_cell_n + 1 (safe for add_annotation filtering)
    """
    trial_cells = []
    dat = obj.tp_data_dict[time]
    ct_col = obj.params.cell_type_label

    if min_keep is None:
        # ensure each CT survives add_annotation() filtering (> filter_cell_n)
        min_keep = int(obj.params.filter_cell_n) + 1

    for _, mat in dat.obs.groupby(ct_col):
        cell_ids = list(mat.index)
        n = len(cell_ids)
        if n == 0:
            continue

        N = int(n * frac)
        # guarantee at least min_keep, but never exceed n
        N = max(N, min_keep)
        N = min(N, n)
        
        if N == n:
            trial_cells.extend(cell_ids)
        else:
            trial_cells.extend(random.sample(cell_ids, N))
    return trial_cells


# Perform gene clustering for each time point
def process_timepoint(dat, tp):
    """
    Process a single time point for GTra bootstrapping.

    Steps:
        1. Build a temporary bootstrap object.
        2. Randomly subsample cells for the target time point.
        3. Perform cell-level clustering (Leiden or annotation-based).
        4. Extract the subset expression matrix and metadata.
        5. Perform gene-level clustering for each discovered cell cluster.

    Returns:
        A dictionary containing:
            - tp: time point index
            - K: number of detected cell clusters
            - clabel: cell cluster label vector
            - cli: list of cell indices per cluster
            - glabels: list of gene-cluster assignments per cell cluster
    """
    
    mini = _build_boot_obj(dat)

    # 1) Random cell selection
    trial_cells = _create_random_cells(mini, tp)
    if len(trial_cells) == 0:
        return dict(tp=tp, K=0, clabel=None, cli=None, glabels=None)
    
    # Slice AnnData safetly
    adata = mini.tp_data_dict[tp]
    mini.tp_data_dict[tp] = adata[adata.obs_names.isin(trial_cells)].copy()

    # 2) Cell clustering
    cell_clustering(mini, tp)
    K = mini.cell_optimal_k[tp]
    clabel = mini.cell_cluster_label[tp]
    cli = mini.cell_label_index[tp]

    # 3) Prepare numeric matrix
    X = mini.tp_data_dict[tp].X
    obs = mini.tp_data_dict[tp].obs
    gene_names = mini.tp_data_dict[tp].var_names.tolist()

    # 4) Gene clustering
    glabels = []
    for cid in range(K):
        gcl = knn_based_gene_clustering(X, obs, cid, gene_names)
        if len(gcl) == 0:
            # fallback: 최소 1개 module 보장 (전체 genes 또는 HVG subset 등)
            gcl = [gene_names]   # 또는 [list(intersection genes)] 등
        glabels.append(gcl)
    
    return dict(
        tp=tp,
        K=K,
        clabel = clabel,
        cli=cli,
        glabels=glabels
    )

# Perform Step 1 and 2
def Run_step1_and_2(dat):
    boot_obj = _build_boot_obj(dat)
    T = boot_obj.tp_data_num
    
    # Stage 1: timepoint-level sequential
    results = []
    for tp in range(T):
        res = process_timepoint(dat, tp)
        results.append(res)
    # Store cluster results to object
    for res_tp in results:
        tp = res_tp["tp"]
        boot_obj.gene_label_info[tp] = res_tp["glabels"]
        boot_obj.cell_cluster_label[tp] = res_tp["clabel"]
        boot_obj.cell_label_index[tp] = res_tp["cli"]
        boot_obj.cell_optimal_k[tp] = res_tp["K"]
    
    # Stage 2: Edge score & rank test
    boot_obj.cell_type_info = concat_meta(boot_obj)
    all_edges = []
    for t in range(T-1):
        boot_obj.cal_edge_score(t, t+1)
        boot_obj.edge_rank_test(t, t+1)
        
        all_edges.extend(boot_obj.selected_edges[t])
    
    boot_obj.node_info = pd.DataFrame(all_edges)
    boot_obj.node_cnt = len(all_edges)
    
    res = save_stat_res(boot_obj)
    gcinfo = get_gcinfo(boot_obj)
    
    del boot_obj
    
    return res, gcinfo

# Store score distribution information
def _score_distribution(obj):
    """
    obj.score_dict: {interval: {'s_t': [scores]}}
    boj.tp_data_dict[tp].obs: cluster_label -> cell types
    """
    
    ct_label_dict = dict()
    for tp in range(obj.tp_data_num):
        cell_clustering(obj, tp)
        obs = obj.tp_data_dict[tp].obs
        cl_col = "cluster_label"
        ct_col = obj.params.cell_type_label

        # Map each cluster label -> (majority) cell-type name.
        # Select the two columns explicitly so the mapping is robust to any
        # extra metadata columns the user may carry in obs.
        if ct_col == cl_col:
            ct_label = {str(cl): cl for cl in obs[cl_col].unique()}
        else:
            vc = obs[[cl_col, ct_col]].value_counts()  # sorted by count desc
            ct_label = {}
            for (cl, ct), _ in vc.items():
                key = str(cl)
                if key not in ct_label:   # first occurrence = majority cell type
                    ct_label[key] = ct
        ct_label_dict[tp] = ct_label
    
    x = copy.deepcopy(obj.score_dict)
    intervals = x.keys()
    iter_static = []
    for it in intervals:
        for st, vals in x[it].items():
            tok = st.split('_')
            if obj.params.label_flag:
                source = ct_label_dict[it][tok[0]]
                target = ct_label_dict[it+1][tok[1]]
            else:
                source = f't{str(it)}_{tok[0]}'
                target = f't{str(it+1)}_{tok[1]}'

            for v in vals:
                iter_static.append([it, source, target, v])
    dist_df = pd.DataFrame(
        iter_static, columns=["Interval", "source","target", "score"]
    )
    return dist_df
    

def statistical_testing(obj, N=50, n_cores=8):
    """
    Completely optimized statistical testing pipeline.
    - Bootstrap iterations run outer loop
    - Each iteration internally parallel over timepoints
    - Merging vectorized
    - Candidate edge extraction accelerated
    """
    
    dat =_extract_dat(obj)
    
    # 1) Run N bootstrap iterations
    # res = Parallel(n_jobs=n_cores, backend='threading')(
    #     delayed(Run_step1_and_2)(dat)
    #     for _ in range(N)
    #     )
    
    res = Parallel(n_jobs=n_cores, backend="loky")(  # <= threading -> loky
        delayed(Run_step1_and_2)(dat) for _ in range(N)
    )

    # 2) Build CCM (correspondence matrix)
    obj.ccmatrix = get_ccmatrix(res)
    
    # 3) Merge edge information (score + occurrence)
    merged_dat, merged_score = get_scoreinfo(res)
    obj.score_dict = merged_score
    
    # 4) Compute probability (% occurrence)
    cnt_dict = {}
    for inter, subdict in merged_dat.items():
        cnt_dict[inter] = {edge: (sum(vals)/N)*100
                                  for edge, vals in subdict.items()}
        obj.cnt_dict = cnt_dict
    
    # 5) Select candidate edges by threshold
    th = obj.params.static_th
    candidate_dict = {}
    for inter, subdict in cnt_dict.items():
        cand = [edge for edge, v in subdict.items() if v >= th]
        candidate_dict[inter] = cand
    
    obj.candidate_dict = candidate_dict
    obj.static_flag = True
    
    # 6) Compute score distribution
    obj.dist_df = _score_distribution(obj)
    
    # 7) Compute p-values
    obj.pval_df = cal_pvals(obj.dist_df)
    

def _calc_gap(linked, min_k=2, max_k=None):
    """
    Select optimal K by largest distance jump in hc
    """
    d = linked[:, 2] # linkage distance
    delta_d = np.diff(d)
    jump_idx = np.argmax(delta_d)
    
    # linkage result has n-1 merges
    n = linked.shape[0] + 1
    optimal_k = n - jump_idx
    
    if max_k is None:
        max_k = n // 2
    optimal_k = max(min_k, min(optimal_k, max_k))
    
    return optimal_k


def remap_timepoint_states(obj, time):
    # 실제 존재하는 cluster key
    actual_keys = sorted(obj.ccmatrix[time].keys())
    key_map = {old: new for new, old in enumerate(actual_keys)}

    # 1) ccmatrix remap
    new_cc = {}
    for old, new in key_map.items():
        new_cc[new] = obj.ccmatrix[time][old]
    obj.ccmatrix[time] = new_cc

    # 2) cell_label_index remap
    old_cli = obj.cell_label_index[time]
    new_cli = []
    for old in actual_keys:
        if old < len(old_cli):
            new_cli.append(old_cli[old])
        else:
            new_cli.append([])
    obj.cell_label_index[time] = new_cli

    # 3) cell_optimal_k update
    obj.cell_optimal_k[time] = len(actual_keys)

    # 4) cluster labels in obs remap
    adata = obj.tp_data_dict[time]
    if "cluster_label" in adata.obs.columns:
        adata.obs["cluster_label"] = adata.obs["cluster_label"].map(key_map)
        
def cc_clustering(obj):
    cc_dict = obj.ccmatrix.copy()
    genes = obj.genes.copy()

    for time in range(obj.tp_data_num):
        remap_timepoint_states(obj, time)
        actual_keys = sorted(cc_dict[time].keys())   # 실제 존재하는 cluster만
        clabel_clusters = []

        for old_clabel in actual_keys:
            cc = cc_dict[time][old_clabel]

            if cc is None or getattr(cc, "shape", (0, 0))[0] < 2:
                clabel_clusters.append([genes])
                continue

            linked = linkage(cc, "ward")
            K = _calc_gap(linked)
            clusters = fcluster(linked, K, criterion="maxclust")

            clustered_genes = []
            for cl in range(1, max(clusters) + 1):
                tmp = [gene for gene, label in zip(genes, clusters) if label == cl]
                if len(tmp) > 0:
                    clustered_genes.append(tmp)

            clabel_clusters.append(clustered_genes)

        obj.gene_label_info[time] = clabel_clusters
        obj.cell_optimal_k[time] = len(clabel_clusters)
