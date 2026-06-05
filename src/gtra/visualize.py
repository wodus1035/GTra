import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf  # draw_patterns uses PdfPages; submodule must be imported explicitly
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.colors as mcolors
import plotly.graph_objects as go

import networkx as nx

import itertools

import re
import os

from scipy.cluster.hierarchy import linkage, fcluster
from matplotlib.legend_handler import HandlerPatch



from .cluster_func import _calc_gap


cmaps = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F',
         '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC']

# Draw consensus matrix of each cell type
def draw_ccmatrix(obj, cc, gnames, title="GC"):
    """
    Visualize static edge score distributions using KDE plots.

    This function plots kernel density estimates (KDEs) of edge scores
    computed from static edge statistics tests in GTra. Distributions are
    faceted by time interval and source cell type, with target cell types
    shown as different colors.

    Parameters
    ----------
    obj : GTraObject
        GTra object containing edge statistics results and plotting metadata.
        Required attributes include `dist_df`, `celltype_colors`, and `params`.

    Notes
    -----
    The function assumes that `obj.dist_df` is already populated by a prior
    edge statistics test. If not, a warning message is printed.
    """
    safe_title = re.sub(r'[\\/*?:"<>|]', "_", title).replace("/","_")
    linked = linkage(cc, "ward")

    optimal_k = _calc_gap(linked)

    clusters = fcluster(linked, optimal_k, criterion="maxclust")
    cluster_df = pd.DataFrame(clusters, index=gnames, columns=["cluster"])

    gid_info = dict(cluster_df["cluster"])
    cluster_colors = [cmaps[gid_info[i]] for i in gnames]

    g = sns.clustermap(
        cc,
        cmap="vlag",
        figsize=(7,7),
        method="ward",
        col_colors=cluster_colors,
        xticklabels=False,
        yticklabels=False
    )

    cluster_legend = [mpatches.Patch(color=cmaps[i], label=f'GC{i}')
                      for i in sorted(set(gid_info.values()))]

    g.ax_col_dendrogram.legend(
        handles=cluster_legend,
        title="Gene Clusters",
        loc="upper left",
        bbox_to_anchor=(1, 1.0),
        fontsize=9,
        title_fontsize=10,
        frameon=False
        )
    
    col_colors_raw = g.col_colors

    color_row = col_colors_raw  # shape: (N,)
    color_row = np.array(color_row)
    col_order = g.dendrogram_col.reordered_ind
    ordered_colors = color_row[col_order]

    boundaries = []
    for color, group in itertools.groupby(enumerate(ordered_colors), key=lambda x: tuple(x[1])):
        group = list(group)
        start = group[0][0]
        end = group[-1][0]
        boundaries.append((start, end, color))
    
    g.fig.suptitle(safe_title,fontsize=10, y=1.02)

    ax = g.ax_heatmap
    edge_color='white'
    face_alpha=0.07
    for start, end, color in boundaries:
        size = end - start + 1
        rect = mpatches.Rectangle(
            (start, start), size, size,
            linewidth=2.5,
            edgecolor=edge_color,
            facecolor=color + (face_alpha,),
            linestyle='-',
            joinstyle='round'
        )    
        ax.add_patch(rect)
    
    ccoutput = f'{obj.params.output_dir}/ccmatrix/'
    os.makedirs(ccoutput, exist_ok=True)
    g.fig.savefig(f"{ccoutput}/{safe_title}_cm.png", bbox_inches="tight")



# Draw statistic testing results
def draw_edge_stat(obj):
    """
    Plot static edge score distributions using kernel density estimation (KDE).

    This function visualizes the distribution of edge scores obtained from
    static edge statistics tests in GTra. KDE plots are faceted by time
    interval and source cell type, with target cell types distinguished by color.
    The resulting figure is saved as a PDF file.
    """
    if len(obj.score_dict) == 0:
        print("Perform the edge statistics test first!!!..")
    else:
        dist_df = obj.dist_df.copy()
        dist_g = sns.FacetGrid(
            dist_df, row="Interval", col="source", sharey=False, sharex=False,
            hue="target", palette=obj.celltype_colors
        )

        dist_g = dist_g.map(
            sns.kdeplot, "score", fill=False, warn_singular=False
        )

        for ax in dist_g.axes.flat:
            row_val = ax.get_title().split('|')[0].split('=')[1].strip()
            col_val = ax.get_title().split('|')[1].split('=')[1].strip()
            ax.set_title(f"{col_val} at Interval {row_val}", fontsize=12)

        
        os.makedirs(obj.params.output_dir, exist_ok=True)

        outputs = f"{obj.params.output_dir}/{obj.params.output_name}"
        dist_g.add_legend(loc="upper left", bbox_to_anchor=(.95, 1), title="Cell types")
        dist_g.savefig(
            f"{outputs}_static_res.pdf"
        )


##################################################################################
## Draw state transition graph between adjacent time points (Gene cluster) [START]
##################################################################################
def _draw_edges(ax, G, pos, node_color):
    for u, v, d in G.edges(data=True):
        if u == v:
            continue

        weight = -np.log10(d["p-value"])
        # lw = np.clip(0.5 + 0.3 * weight, 0.8, 6.0)
        # alpha = np.clip(0.2 + 0.025 * weight, 0.3, 0.9)
        lw = np.clip(0.8 + 0.9 * np.sqrt(weight), 1.2, 6.5)
        alpha = np.clip(0.35 + 0.08 * np.sqrt(weight), 0.4, 0.95)
        rad = 0.05 * ((hash(u) % 5) - 2) / 2

        ax.annotate(
            "",
            xy=pos[v], xytext=pos[u],
            arrowprops=dict(
                arrowstyle="-|>",
                color=node_color.get(u, "#999999"),
                lw=lw,
                alpha=alpha,
                shrinkA=10, shrinkB=10,
                connectionstyle=f"arc3,rad={rad}",
            )
        )


def _draw_nodes(ax, G, pos, node_color, self_nodes):
    for n, (x, y) in pos.items():
        c = node_color.get(n, "#999999")

        # node
        ax.scatter(x, y, s=1300, color=c, alpha=0.25, zorder=1)
        ax.scatter(x, y, s=700, color=c, alpha=0.95,
                   edgecolors="white", linewidth=0.8, zorder=2)

        # cell type label
        ax.text(x, y - 0.22, n,
                ha="center", va="top",
                fontsize=10, zorder=3)

        # self persistence 표시
        if n in self_nodes:
            ax.text(
                x, y,
                "self",
                ha="center", va="center",
                fontsize=9,
                color="white",
                fontweight="bold",
                zorder=4
            )


def draw_transition_graph(obj, p_th=0.05):
    """
    Draw directed transition graphs between cell types across time intervals.

    Significant transitions (p < p_th) are visualized as directed edges,
    where edge thickness and transparency reflect statistical significance.
    """

    # === time interval labels ===
    if len(obj.params.time_point_label) > 1:
        time_points = obj.params.time_point_label
    else:
        time_points = [f"T{i}" for i in range(1, obj.tp_data_num + 1)]

    interval_conv = {
        i: f"{time_points[i]} → {time_points[i+1]}"
        for i in range(len(time_points) - 1)
    }

    # === filter significant edges ===
    df = obj.pval_df.loc[obj.pval_df["adj_p-value"] < p_th].copy()
    if df.empty:
        print("No significant transitions found.")
        return
    
    # === Check answer path ====
    fname = obj.params.answer_path_dir
    if os.path.isfile(fname):
        answer_df = pd.read_csv(fname, sep=",")
        answer_edges = set(
            zip(answer_df["source"], answer_df["target"])
        )
            
        df = df[
            df.apply(lambda r: (r["source"], r["target"]) in answer_edges, axis=1)
        ].copy()

    intervals = sorted(df["Interval"].unique())

    # === node colors ===
    all_nodes = sorted(set(df["source"]).union(df["target"]))
    node_color = {n: obj.celltype_colors.get(n, "#999999") for n in all_nodes}

    # === figure ===
    fig, axes = plt.subplots(
        1, len(intervals),
        figsize=(5.8 * len(intervals), 5),
        constrained_layout=False
    )
    plt.subplots_adjust(
        left=0.04,
        right=0.98,
        top=0.90,
        bottom=0.08,
        wspace=0.2
        )
    if len(intervals) == 1:
        axes = [axes]

    for ax, interval in zip(axes, intervals):
        sub_df = df[df["Interval"] == interval]

        G = nx.from_pandas_edgelist(
            sub_df,
            source="source",
            target="target",
            edge_attr="p-value",
            create_using=nx.DiGraph()
        )

        pos = nx.shell_layout(G)

        self_nodes = set(
            sub_df.loc[sub_df["source"] == sub_df["target"], "source"]
        )

        _draw_edges(ax, G, pos, node_color)
        _draw_nodes(ax, G, pos, node_color, self_nodes)

        # --- FIX: axis limit padding ---
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]

        pad = 0.35  # 여백 (node + text + self 고려)

        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)


        ax.set_title(interval_conv.get(interval, str(interval)),
                     fontsize=15, weight="bold")
        ax.axis("off")
        
    output_dir = f"{obj.params.output_dir}/{obj.params.output_name}_tansition.pdf"
    plt.savefig(output_dir)

    plt.show()

##################################################################################
## Draw state transition graph between adjacent time points (Cell types) [END]
##################################################################################


####################################################################################
## Draw state transition matrix between adjacent time points (Gene clusters) [START]
####################################################################################
def _parse_node_label(df):
    """Extract time, cell type, gene cluster from source/target."""
    df[['t1_time', 't1_ct', 't1_gc']] = df['source'].str.extract(r"t(\d+)_(\d+)_(\d+)")
    df[['t2_time', 't2_ct', 't2_gc']] = df['target'].str.extract(r"t(\d+)_(\d+)_(\d+)")
    cols = ['t1_ct', 't1_gc', 't2_ct', 't2_gc']
    df[cols] = df[cols].astype(int)
    return df


def _map_celltype_names(df, type_to_cluster):
    df['t1_ct_name'] = df['t1_ct'].map(type_to_cluster).fillna("CT" + df['t1_ct'].astype(str))
    df['t2_ct_name'] = df['t2_ct'].map(type_to_cluster).fillna("CT" + df['t2_ct'].astype(str))
    return df


def _build_pivot(df):
    df['intensity'] = 1 / (df['rank'] + 1e-6)
    df['source_label'] = df['t1_ct_name'] + "_G" + (df['t1_gc'] + 1).astype(str)
    df['target_label'] = df['t2_ct_name'] + "_G" + (df['t2_gc'] + 1).astype(str)

    return (
        df.pivot_table(
            index='source_label',
            columns='target_label',
            values='intensity',
            aggfunc='mean'
        ).fillna(0)
    )
# === circle을 legend에서 유지하기 위한 핸들러 ===
def make_legend_circle(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
    # facecolor를 안전하게 추출
    fc = orig_handle.get_facecolor()
    if hasattr(fc, "__len__"):
        if len(fc) == 1:
            fc = fc[0]
    fc = mcolors.to_rgba(fc)  # ✅ float 튜플 → RGBA 확정

    return mpatches.Circle((width / 2, height / 2),
                           radius=height / 2.5,
                           facecolor=fc,
                           edgecolor='black',
                           lw=0.4)

def draw_gg_matrix(obj):
    # === interval label (한 번만) ===
    if len(obj.params.time_point_label) > 1:
        time_points = obj.params.time_point_label
    else:
        time_points = [f'T{i}' for i in range(1, obj.tp_data_num + 1)]

    interval_labels = {
        i: f'{time_points[i]} → {time_points[i+1]}'
        for i in range(len(time_points) - 1)
    }

    ct_color_map = obj.celltype_colors

    for interval_idx, edge_list in obj.selected_edges.items():
        # === DataFrame 생성 ===
        df = pd.DataFrame(
            edge_list,
            columns=["source", "target", "jaccard", "cosine", "rank"]
        )

        # === Parsing ===
        df = _parse_node_label(df)

        # === cell type 이름 매핑 ===
        try:
            in_ct = obj.tp_data_dict[interval_idx].obs.drop_duplicates()
            type_to_cluster = dict(
                zip(in_ct['cluster_label'], in_ct[obj.params.cell_type_label])
            )
        except Exception:
            type_to_cluster = {}

        df = _map_celltype_names(df, type_to_cluster)

        # === Pivot matrix ===
        pivot_df = _build_pivot(df)

        title = interval_labels.get(interval_idx, f"Interval {interval_idx}")

        # === Heatmap ===
        n_row, n_col = pivot_df.shape
        cell_size, base = 0.33, .8
        fig_w, fig_h = base + n_col * cell_size, base + n_row * cell_size

        fig, ax = plt.subplots(
            figsize=(fig_w, fig_h),
            constrained_layout=True
            )
        
        vals = pivot_df.values
    
        
        sns.heatmap(
            pivot_df,
            cmap = "viridis",
            ax=ax,
            cbar_kws={
                'label': 'Transition strength',
                'shrink': 0.65
                }
                )

        ax.axis('off')
        ax.set_aspect('equal')

        # === row / col annotation ===
        row_ct = [i.split('_')[0] for i in pivot_df.index]
        row_gc = [i.split('_')[1] for i in pivot_df.index]
        col_ct = [i.split('_')[0] for i in pivot_df.columns]
        col_gc = [i.split('_')[1] for i in pivot_df.columns]

        for i, (ct, gc) in enumerate(zip(row_ct, row_gc)):
            y = i + 0.5
            color = ct_color_map.get(ct, "gray")
            ax.add_patch(plt.Circle((-0.6, y), 0.25, color=color,
                                     ec='black', lw=0.4, clip_on=False, alpha=0.6))
            ax.text(-0.6, y, gc, ha='center', va='center', fontsize=9)

        for j, (ct, gc) in enumerate(zip(col_ct, col_gc)):
            x = j + 0.5
            color = ct_color_map.get(ct, "gray")
            ax.add_patch(plt.Circle((x, n_row + 0.6), 0.25, color=color,
                                     ec='black', lw=0.4, clip_on=False, alpha=0.6))
            ax.text(x, n_row + 0.6, gc, ha='center', va='center', fontsize=9)
        
        # === Legends ===
        used_celltypes = sorted(set(row_ct) | set(col_ct))
        filtered_ct_color_map = {ct: obj.celltype_colors[ct] for ct in used_celltypes if ct in obj.celltype_colors}

        patches = [
            mpatches.Circle((0, 0), radius=0.22,
                            facecolor=color,  # 여기서 color는 이미 (r,g,b)
                            label=ct, ec='black', lw=0.4)
            for ct, color in filtered_ct_color_map.items()
        ]


        # figure 밖 오른쪽 배치
        legend_x = .9 + (2.0 / fig_w)

        fig.legend(
            handles=patches,
            title="Cell types",
            loc="upper right",
            bbox_to_anchor=(legend_x, 0.95),
            frameon=False,
            handler_map={mpatches.Circle: HandlerPatch(patch_func=make_legend_circle)},
            handletextpad=0.3,
            handlelength=1.0,
            labelspacing=0.3,
            borderaxespad=0.2,
            fontsize=10
            )

        # === labels ===
        ax.text(-1.2, n_row / 2, "Source gene clusters",
                rotation=90, ha='center', va='center', fontsize=10)
        ax.text(n_col / 2, n_row + 1.4, "Target gene clusters",
                ha='center', va='center', fontsize=10)

        plt.title(title, pad=10)
        plt.show()

##################################################################################
## Draw state transition matrix between adjacent time points (Gene clusters) [END]
##################################################################################

# Draw time-series gene expression patterns
def draw_patterns(obj):
    from .utils import l2norm, vect_mu, convert_path_name, make_gene_set_frame
    from .utils import plotting_patterns

    print("Plotting time-series gene expression patterns...")

    output_name = f"{obj.params.output_dir}/{obj.params.output_name}_patterns.pdf"
    pdf = matplotlib.backends.backend_pdf.PdfPages(output_name)

    # Figure positions
    row_n, col_n, pos = 3, 3, 0
    fig, ax = plt.subplots(figsize=(14,10), nrows=row_n, ncols=col_n)

    # Trajectory keys
    pt_keys = list(obj.merge_pattern_dict.keys())
    time_len = obj.tp_data_num

    # Gene set data frame
    gene_set_df = pd.DataFrame()

    # In-memory record of {pattern key -> trajectory name} for the plotted
    # (significant) patterns, so module_evaluation() no longer depends on
    # round-tripping through the *_pattern_genes.csv file.
    key_map = {}

    # Plotting time-series gene expression patterns
    fc_th = 1.2
    for idx, key in enumerate(pt_keys):
        # Normalization
        pt_df = l2norm(obj.merge_pattern_dict[key])
        cent = vect_mu(pt_df)

        if len(pt_df.columns) != time_len: continue
        if max(cent) / min(cent) < fc_th: continue

        # Customizing a personalized list of specific time points for each user
        if len(obj.params.time_point_label) != 0:
            pt_df.columns = obj.params.time_point_label

        # Store cell-state trajectory info and gene sets
        convert_name = convert_path_name(obj, key)
        start_cells = convert_name[: convert_name.find("-")]
        gene_set_df = make_gene_set_frame(idx, gene_set_df, pt_df, key, convert_name)
        key_map[key] = convert_name

        # Update position
        if pos // col_n == col_n:
            fig.tight_layout()
            pdf.savefig(fig)
            plt.show()
            fig.clf()
            fig, ax = plt.subplots(figsize=(14,10), nrows=row_n, ncols=col_n)
            pos = 0
        
        # Plotting gene expression patterns
        plotting_patterns(pt_df, key, start_cells, ax, pos)
        pos+=1
    
    # Store gene set information for cell trajectory
    obj.pattern_key_map = key_map
    gene_set_df.to_csv(f'{obj.params.output_dir}/{obj.params.output_name}_pattern_genes.csv',sep=",")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.show()
    pdf.close()
    
    
from .utils import build_sankey_df_from_pvals, rgb01_to_rgbstr, extract_time_and_celltype

def draw_trajectory(obj):
    color_mapping = obj.celltype_colors.copy()
    
    df = build_sankey_df_from_pvals(obj, min_gn=5, pval_th=1e-5)
    df["Interval"] = df["Interval"].astype(int)

    interval_to_time = {i: f"T{i}" for i in sorted(df["Interval"].unique())}
    next_time = {f"T{i}": f"T{i+1}" for i in sorted(df["Interval"].unique())}

    df["source_label"] = df.apply(lambda row: 
        f"{interval_to_time[row['Interval']]}: {row['source']}", axis=1)
    df["target_label"] = df.apply(lambda row: 
        f"{next_time[interval_to_time[row['Interval']]]}: {row['target']}", axis=1)

    labels = sorted(set(df["source_label"]).union(set(df["target_label"])),
                    key=lambda x: extract_time_and_celltype(x))
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    node_colors = [color_mapping[label.split(": ")[1]] for label in labels]

    time_labels = sorted(set(l.split(":")[0] for l in labels))
    celltype_order = sorted(color_mapping.keys())

    cell_type_per_time = {
        t: [ct for ct in celltype_order if ct in  set(l.split(": ")[1] for l in labels if l.startswith(t))]
        for t in time_labels
        }

    node_colors = [rgb01_to_rgbstr(c) for c in node_colors]

    spacing_factor = 1.1

    node_x, node_y = [], []

    for label in labels:
        t, ct = label.split(': ')
        ct_list = cell_type_per_time[t]
        x = time_labels.index(t) / (len(time_labels) - 1) if len(time_labels) > 1 else 0.5
        y_rank = ct_list.index(ct)
        y = (y_rank / max(1, len(ct_list) - 1)) * spacing_factor
        node_x.append(x)
        node_y.append(1 - y)

    df["source_index"] = df["source_label"].map(label_to_index)
    df["target_index"] = df["target_label"].map(label_to_index)
    link_colors = [node_colors[idx] for idx in df["source_index"]]

    visible_labels = []
    for label in labels:
        t, _ = label.split(': ')
        visible_labels.append(label.split(': ')[1])



    fig = go.Figure(data=[go.Sankey(
        arrangement = "fixed",
        node= dict(
            pad=15,
            thickness=10,
            line=dict(color="black", width=0.8),
            label=visible_labels,
            color=node_colors,
            x=node_x,
            y=node_y
            ),
        link = dict(
            source=df["source_index"],
            target=df["target_index"],
            value=df["GN"],
            color=link_colors
            )
    )])

    fig.update_layout(title_text="", 
                    font_size=10,
                    height=400,
                    margin=dict(l=80, r=80, t=80, b=80))

    time_anno = obj.params.time_point_label
    for i, t in enumerate(time_anno):
        x = i / (len(time_anno) - 1) if len(time_anno) > 1 else 0.5
        fig.add_annotation(
            x=x, y=-0.32,  # y는 다이어그램 하단에 위치하도록 음수로 조정
            text=t,
            showarrow=False,
            xref="paper", yref="paper",
            font=dict(size=14),
            align="center"
        )

    outputdir = f"{obj.params.output_dir}/{obj.params.output_name}_trajectory_sankey.pdf"
    fig.write_image(outputdir)
    fig.show()    


def draw_module_cluster(obj):
    n_colors = obj.cell_type_info.iloc[:,0].nunique()
    cb_palette = sns.color_palette("Set2", n_colors)
    
    res_df = obj.sig_patterns
    pt_dist = obj.pattern_dist
    pt_df = obj.module_df
    clusters = pt_df['cluster'].values
    

    c_lut = {i:cb_palette[i] for i in range(max(clusters) + 1)}
    trend = pd.Series(clusters, index=res_df.index)

    col_colors = trend.map(c_lut)

    g = sns.clustermap(
        pt_dist, 
        col_colors=col_colors.to_numpy(),
        xticklabels=False,
        yticklabels=False,
        cmap='vlag', 
        method="ward",
        figsize=(6, 6)
        )

    g.ax_heatmap.set_xlabel("Modules",fontsize=15)    # x축 제목
    g.ax_heatmap.set_ylabel("Modules",fontsize=15)    # y축 제목
    g.cax.set_ylabel("Correlation", rotation=270, labelpad=10, fontsize=11)

    cluster_legend = [mpatches.Patch(color=c_lut[i], label=f'C{i}')
                    for i in sorted(np.unique(pt_df["cluster"].values))]

    g.ax_col_dendrogram.legend(
                handles=cluster_legend,
                title="Module \nclusters",
                loc='upper left',
                bbox_to_anchor=(1, 1.0),
                fontsize=9,
                title_fontsize=10,
                frameon=False
                )

    col_colors_raw = g.col_colors

    color_row = np.array(col_colors_raw)
    col_order = g.dendrogram_col.reordered_ind
    ordered_colors = color_row[col_order]

    boundaries = []
    for color, group in itertools.groupby(enumerate(ordered_colors), 
                                          key=lambda x: tuple(x[1])):
        group = list(group)
        start = group[0][0]
        end = group[-1][0]
        boundaries.append((start, end, color))

    ax = g.ax_heatmap
    edge_color='white'
    face_alpha=0.07
    for start, end, color in boundaries:
        size = end - start + 1
        rect = mpatches.Rectangle(
            (start, start), size, size,
            linewidth=3.5,
            edgecolor=edge_color,
            facecolor=color + (face_alpha,),
            linestyle='-',
            joinstyle='round'
        )    
        ax.add_patch(rect)


from .utils import l2norm

def draw_rep_patterns(obj):
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'legend.fontsize': 10,
        'xtick.labelsize': 16,
        'ytick.labelsize': 10,
        'font.family': 'sans-serif',
        'axes.edgecolor': 'black',
        'axes.linewidth': 0.8
    })

    sns.set(style="white", context="paper")

    n_colors = obj.cell_type_info.iloc[:,0].nunique()
    cb_palette = sns.color_palette("Set2", n_colors)

    pt_df = obj.module_df.copy()
    n_clusters = pt_df["cluster"].nunique()
    fig, axes = plt.subplots(n_clusters, 1, figsize=(4, 2.5 * n_clusters), sharex=True)

    if n_clusters == 1:
        axes = [axes]

    pt_id_c = dict()

    for ax, (c, d) in zip(axes, pt_df.groupby("cluster")):
        tmp = []
        expr_list = []
        union_genes = set()

        for pi in d["Pattern_ID"]:
            tmp.append(pi)
            expr = l2norm(obj.merge_pattern_dict[pi])
            avg_expr = expr.mean(axis=0)
            expr_list.append(avg_expr)

            union_genes.update(list(expr.index))

        pt_id_c[c] = tmp

        expr_df = pd.concat(expr_list, axis=1)
        mean_expr = expr_df.mean(axis=1)
        std_expr = expr_df.std(axis=1)

        ax.plot(obj.params.time_point_label, mean_expr.values,
                label=None,
                marker='o',
                color=cb_palette[c-1],
                linewidth=2)

        ax.fill_between(obj.params.time_point_label,
                        mean_expr.values - std_expr.values,
                        mean_expr.values + std_expr.values,
                        color=cb_palette[c-1],
                        alpha=0.2, linewidth=0)

        ax.set_title(f"C{c} (n={len(union_genes)})", fontsize=13, fontweight='bold')

        ax.set_ylabel("Expression", fontsize=11)
        ax.grid(False)
        ax.tick_params(axis='x', labelsize=12)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[-1].set_xlabel("Time points", fontsize=11)

    plt.tight_layout()

    output_dir = f"{obj.params.output_dir}/{obj.params.output_name}_MC_patterns.pdf"
    plt.savefig(output_dir)
    plt.show()
