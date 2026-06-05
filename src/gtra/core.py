import numpy as np
import pandas as pd
import scanpy as sc

import anndata as ad
import networkx as nx

from tqdm import tqdm

from scipy.stats import wilcoxon
from soyclustering import SphericalKMeans
from scipy.sparse import csr_matrix

from sklearn.metrics.pairwise import cosine_similarity

from .cluster_func import statistical_testing
from .preproc import *
from .utils import *
from .cluster_func import *

class GTraObject(object):
    class GTraParam:
        __slots__ = (
            "low_gene_num", "low_cell_num", "mito_percent", "hvg_n", "norm_flag",
            "label_flag", "gene_norm_flag", "cn_neighbors", "gn_neighbors", "cn_cluster_resolution",
            "gn_cluster_resolution", "filter_cell_n", "dist_threshold", "sw", "dw", "top_rank",
            "static_th", "answer_path_type", "answer_path_dir", "cell_type_label", "time_point_label",
            "min_pattern_gene", "output_dir","output_name")

        # Basic parameters
        def __init__(self):
            # For data preprocessing
            self.low_gene_num = 200
            self.low_cell_num = 3
            self.mito_percent = 0.2
            self.hvg_n = 2000
            self.norm_flag = True
            
            # Step 1: GTra's consensus clustering
            self.label_flag = False
            self.gene_norm_flag = True
            self.cn_neighbors = 15 # For cells (kNN)
            self.gn_neighbors = 15 # For genes (kNN)
            self.cn_cluster_resolution = 0.5
            self.gn_cluster_resolution = .3
            self.filter_cell_n = 10

            # Step 2: Select candidate edges and construct trajectories
            self.dist_threshold = .5 # For cosine distance
            self.sw, self.dw = .3, .7 # sw: jaccard, dw: cosine
            self.top_rank = 10 # candidate's edges
            self.static_th = 90 # threshold of statistical testing
            self.answer_path_type = "" # organism or dataset
            self.answer_path_dir = "" # directory contained answer path info
            self.cell_type_label = ""
            self.time_point_label = [] # [day+1, day+5, ...]
            self.min_pattern_gene = 20

            # Save directory
            self.output_dir = "./"
            self.output_name = "GTra"
    
    # GTraObject parameters
    def __init__(self):
        self.params = self.GTraParam()

        ## =========== Step 1's parameters =========== ##
        self.tp_data_dict = {}
        self.tp_data_num = 0 # total count of time points
        # For cell clustering
        self.cell_cluster_label = {}
        self.cell_optimal_k = {}
        self.cell_label_index = {}
        self.celltype_colors = {}

        # For gene clustering
        self.genes = {}
        self.tp_genes_dict = {}
        self.gene_label_info = {}
        self.ccmatrix = {}

        ## =========== Step 2's parameters =========== ##
        self.ctc_list, self.gtg_list = {}, {}
        self.edge_info = {}
        self.selected_edges = {}
        self.node_info, self.node_cnt = pd.DataFrame(), 0
        self.net_info = [[[]]]
        self.answer_path = pd.DataFrame() # standard trajectory info [source -> target]
        self.cnt_dict, self.score_dict = {}, {} # statistical testing
        self.candidate_dict = {} # edges passed statistical testing
        self.dist_df, self.pval_df = pd.DataFrame(), pd.DataFrame()
        self.pval_th = 0.05
        self.static_flag = False

        ## =========== Step 3's parameters =========== ##
        self.path_gene_sets = {}
        self.path_candidates = {}
        self.merge_pattern_dict = {}
        self.merge_pattern_within_distance= {}
        self.merge_path_dict = {}
        self.cluster_centers = {}
        self.cell_type_info = pd.DataFrame()
        self.f = lambda x: x
        self.merge_node_info = pd.DataFrame.from_records(
            list(map(self.f, [])), columns = ["from", "to"]
        )
        self.merge_net_info = [[[]]]
        
        ## Pattern evaluation
        self.sig_patterns = pd.DataFrame()
        self.pattern_dist = pd.DataFrame()
        self.module_df = pd.DataFrame()
        
    
    # Upload time-series scRNA-seq dataset
    def upload_time_scRNA(self, *args):
        """
        Upload scRNA-seq data for each time point.
        Ensures all time points use the same gene list and same ordering.
        """
        if len(args) == 2: # args[0]: matrix, args[1]: obs
            self.params.label_flag = True # celltype label check
            adata = sc.AnnData(args[0], obs=args[1])
        else:
            adata = sc.AnnData(args[0])
        
        if self.tp_data_num == 0:
            self.genes = adata.var_names.tolist()
        
        self.tp_data_dict[self.tp_data_num] = adata
        self.tp_data_num += 1

    # Filtering low-expressed genes
    def select_genes(self):
        """
        Select filtered genes and enforce a unified gene set across all time points.

        Ensures that every time point uses the same genes in the same order.
        Missing genes in any time point are automatically added as zero vectors.
        """

        gene_sets = [set(filter_genes(self.tp_data_dict[tp]))
                     for tp in range(self.tp_data_num)]
        fgenes = list(gene_sets[0].intersection(*gene_sets[1:]))
        
        for tp in range(self.tp_data_num):
            adata = self.tp_data_dict[tp]
            missing = set(fgenes) - set(adata.var_names)
            if missing:
                # Densify consistently (X may be sparse or dense) before stacking
                Xd = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
                zeros = np.zeros((adata.n_obs, len(missing)))
                X = np.hstack([Xd, zeros])
                new_genes = list(adata.var_names) + list(missing)
                pos = {g: i for i, g in enumerate(new_genes)}
                X = X[:, [pos[g] for g in fgenes]]
                self.tp_data_dict[tp] = sc.AnnData(X, obs=adata.obs,
                                                   var=pd.DataFrame(index=fgenes))
            else:
                self.tp_data_dict[tp] = adata[:, fgenes]
        self.genes = fgenes

    ###############################################################################
    ## ============ Step 1: Identifying cell type-specific clusters ============ ##
    ###############################################################################

    # Perform consensus clustering
    def find_gclusters(self, N=50):
        print("Step 1: Identifying cell type-specific gene clusters...")
        if self.params.label_flag == False:
            for tp in range(self.tp_data_num):
                cell_clustering(self, tp)
            self.params.label_flag = True
            self.params.cell_type_label = 'cluster_label'
                
        statistical_testing(self, N=N)
        cc_clustering(self)
        create_color_dict(self)
    
    ###########################################################################
    ## ============= Step 2: Construct cell-state trajectories ============= ##
    ###########################################################################
    
    def cal_edge_score(self, tp1, tp2):
        # Load scRNA data
        t1_df = self.tp_data_dict[tp1].to_df().T
        t2_df = self.tp_data_dict[tp2].to_df().T

        # Load gene sets
        t1_genes = self.gene_label_info[tp1]
        t2_genes = self.gene_label_info[tp2]

        # Sample and gene cluster index (init)
        t1_s, t1_g, t2_s, t2_g = 0, 0, 0, 0

        # Get optimal threshold
        optimal_th = JS_threshold_test(self, tp1, tp2)

        # Comparison intersected gene set between adjacent time points
        edge_info = []
        for t1_s in range(len(t1_genes)):
            for t1_g in range(len(t1_genes[t1_s])):
                for t2_s in range(len(t2_genes)):
                    for t2_g in range(len(t2_genes[t2_s])):
                        sim, inter_genes = Jaccard(
                            t1_genes[t1_s][t1_g], t2_genes[t2_s][t2_g]
                        )
                        if sim < optimal_th: continue

                        dist = cal_cos_dist(
                            self, tp1, tp2, t1_s, t2_s, t1_df, t2_df, inter_genes
                        )
                        if dist is None or dist == -1: 
                            continue

                        cent = float(dist)
                        edge_info.append([t1_s, t1_g, t2_s, t2_g, sim, cent])

        self.edge_info[tp1] = edge_info
    
        
    # Rank test for candidate edges
    def edge_rank_test(self, tp1, tp2):
        # Get edge info for previous time point data
        edge_info_dict = get_edge_info(self, tp1)

        # Edge candidates that have passed statistical tests
        if len(self.candidate_dict):
            candidated_edges = self.candidate_dict[tp1]
            check_edges = {}
            for i in candidated_edges:
                tok = i.split("_")
                s, t = int(tok[0]), int(tok[1])
                if check_edges.get(s) is None:
                    check_edges[s] = [t]
                else:
                    check_edges[s].append(t)

        # Ranking candidate edges [Modi: 25-11-21]
        RANK_PARAM = -1
        edge_info_results = []
        for t1_s, edge_info in edge_info_dict.items():
            conv_edge = cal_rank(edge_info, self.params.sw, self.params.dw)
            sort_conv_edge = np.array(sorted(conv_edge, key=lambda x: x[RANK_PARAM]))

            # Edge candidates that have passed statistical tests
            if len(self.candidate_dict) and (check_edges.get(t1_s)):
                sort_conv_edge = np.array(
                    [list(i) for i in sort_conv_edge if i[1] in check_edges[t1_s]]
                )

            top_rank = min(self.params.top_rank, len(sort_conv_edge))

            # If answer path information exist then ~
            if (self.static_flag == True):
                edge_info_results = check_standard_path(
                    self, tp1, t1_s, top_rank, sort_conv_edge, edge_info_results
                )
            else:
                for rank in range(top_rank):
                    label_info = list(map(int, sort_conv_edge[rank, :3]))
                    score_info = list(sort_conv_edge[rank, 3:])
                    edge_info_results.append([t1_s] + label_info + score_info)

        # Convert edge info name
        conv_edge_info = []
        for einfo in edge_info_results:
            source = f"t{str(tp1)}_{str(einfo[0])}_{str(einfo[1])}"
            target = f"t{str(tp2)}_{str(einfo[2])}_{str(einfo[3])}"
            conv_edge_info.append([source, target, einfo[4], einfo[5], einfo[-1]])

        # Save candidate edges
        self.selected_edges[tp1] = conv_edge_info
        
    
    # Select edges
    def select_candidate_edges(self):
        self.cell_type_info = concat_meta(self)
        
        display_name = "Step 2: Constructing cell-state trajectories.."
        for tp in tqdm(
            range(self.tp_data_num-1),
            total=self.tp_data_num-1,
            desc=display_name,
            ncols=100,
            ascii=" =",
            leave=True
        ):
            self.cal_edge_score(tp, tp+1)
            self.edge_rank_test(tp, tp+1)
        
        # Save edge info
        records = []
        for tp in range(self.tp_data_num - 1):
            records.extend(self.selected_edges[tp])
        
        self.node_info = pd.DataFrame(records, columns=[
            "from", "to", "sim", "cos", "rank_val"
        ])
    
    # Construct cell-state trajectories
    def construct_trajectories(self):
        self.select_candidate_edges()
        
        # Create sub-graphs
        g = nx.from_pandas_edgelist(
            self.node_info, "from", "to", create_using=nx.DiGraph()
        )
        
        sub_graphs = list(g.subgraph(c) for c in nx.weakly_connected_components(g))
        
        # Get candidate path info
        self.net_info = get_networks(self, sub_graphs)
            
    
    ############################################################################
    ## ============= Step 3: Gene expression pattern clustering ============= ##
    ############################################################################

    # Time-series pattern clustering
    def pattern_clustering(self):

        self.merge_path_dict = group_cell_type_trajectory(self.net_info)

        path_label = 0
        displays = "Step 3: Detecting time-series pattern clustering..."
        mpdv = list(self.merge_path_dict.values())
        for i in tqdm(
            range(len(mpdv)),
            total=len(mpdv),
            desc=displays,
            ncols=100,
            ascii=" =",
            leave=True,
        ):
            cluster_label = 0
            paths = mpdv[i]
            for path in paths:
                key = f"{str(path_label)}_{str(cluster_label)}"
                expr = merge_expr(self, path)

                if len(expr) < self.params.min_pattern_gene:
                    self.merge_pattern_dict[key] = expr
                    self.cluster_centers[key] = expr.mean().values
                    continue
                try:
                    optimal_k = elbow_method(expr)
                    spk = SphericalKMeans(
                        n_clusters=optimal_k, random_state=25, max_iter=100
                    ).fit(csr_matrix(expr))
                
                except:
                    self.merge_pattern_dict[key] = expr
                    self.cluster_centers[key] = expr.mean().values
                    continue

                for k in range(optimal_k):
                    gene_cluster = [i for i, c in enumerate(spk.labels_) if c == k]
                    key_k = f"{key}_{str(k)}"
                    self.merge_pattern_dict[key_k] = expr.iloc[gene_cluster, :]
                    self.cluster_centers[key_k] = spk.cluster_centers_
            
                cluster_label += 1
            path_label += 1
    
        self.merge_pattern_dict = pattern_filtering(self) # Filtering low-quality patterns
        new_mp_dict = merge_sim_patterns(self) # Merge selected patterns
        self.merge_pattern_dict = renaming_pattern_id(new_mp_dict) # Convert pattern name
        save_pattern_centroid(self) # Save pattern centroid    
    
    
    # Module evaluation
    def module_evaluation(self):
        pattern_eval(self)
    
    # -------------------------- Visualization functions -------------------------- #
    
    # Plotting edge static results
    def plot_edge_statistic(self):
        from .visualize import draw_edge_stat
        draw_edge_stat(self)
    
    # Plotting cell-state transition graph
    def plot_cell_state_graph(self):
        from .visualize import draw_transition_graph
        draw_transition_graph(self)
    
    # Plotting gene cluster-gene-cluster matrix
    def plot_gg_matrix(self):
        from .visualize import draw_gg_matrix
        draw_gg_matrix(self)
        
    # Plotting time-series gene expression patterns
    def plot_patterns(self):
        from .visualize import draw_patterns
        draw_patterns(self)
    
    def plot_trajectory(self):
        from .visualize import draw_trajectory
        draw_trajectory(self)
    
    def plot_module_cluster(self):
        from .visualize import draw_module_cluster
        draw_module_cluster(self)
    
    def plot_rep_patterns(self):
        from .visualize import draw_rep_patterns
        draw_rep_patterns(self)