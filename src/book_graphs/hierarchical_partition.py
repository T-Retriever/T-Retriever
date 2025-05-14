import os
import pandas as pd
import numpy as np
import torch
from book_graphs.utils import PartitionTree
import json
import argparse
import scipy.sparse as sp
from scipy.sparse import csr_matrix, lil_matrix
import time
from sklearn.decomposition import PCA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()
parser.add_argument('--no_semantic', action='store_true')
parser.add_argument('--alpha', type=float, default=0.3)
parser.add_argument('--bandwidth', type=float, default=0.5)
parser.add_argument('--depth', type=int, default=5)
parser.add_argument('--embed_dim', type=int, default=50)
args = parser.parse_args()
DATASET_PATH = f'dataset/book_graphs'
NODES_FILE = f'{DATASET_PATH}/nodes.csv'
EDGES_FILE = f'{DATASET_PATH}/edges.csv'
NODES_EMB_FILE = f'{DATASET_PATH}/nodes_embedded.csv'
PARTITION_PATH = f'{DATASET_PATH}/partitions_{args.alpha}_{args.bandwidth}'
TREE_DEPTH = args.depth
ALPHA = args.alpha
USE_SEMANTIC = not args.no_semantic
BANDWIDTH = args.bandwidth
EMBED_DIM = args.embed_dim


os.makedirs(PARTITION_PATH, exist_ok=True)

def extract_cluster_assignment(T, tree_depth=10):
    depth_counts = {}
    depth_to_nodes = {}
    leaf_nodes = []
    for i, n in T.items():
        depth = n['depth']
        if depth not in depth_counts:
            depth_counts[depth] = 0
            depth_to_nodes[depth] = []
        depth_counts[depth] += 1
        depth_to_nodes[depth].append(i)
        if not n['children']:
            leaf_nodes.append(i)
    actual_tree_depth = max(depth_counts.keys()) if depth_counts else 0

    interLayerEdges = [[] for _ in range(actual_tree_depth + 1)]

    leaf_to_ancestors = {}
    for leaf in leaf_nodes:
        current = leaf
        path = []
        while current is not None:
            path.append(current)
            current = T[current]['parent'] if T[current]['parent'] is not None else None
        leaf_to_ancestors[leaf] = path
    for depth in range(actual_tree_depth):
        if depth not in depth_to_nodes or depth + 1 not in depth_to_nodes:
            continue
            
        current_layer_nodes = depth_to_nodes[depth]
        next_layer_nodes = depth_to_nodes[depth + 1]
        
        edges = []
        for child_id in current_layer_nodes:
            if child_id in T and T[child_id]['parent'] is not None:
                parent_id = T[child_id]['parent']
                if parent_id in next_layer_nodes:
                    child_idx = current_layer_nodes.index(child_id)
                    parent_idx = next_layer_nodes.index(parent_id)
                    edges.append((child_idx, parent_idx))
        
        if edges:
            interLayerEdges[depth + 1] = edges

    return [edges for edges in interLayerEdges[1:] if edges], leaf_to_ancestors

def extract_partitions(leaf_to_ancestors, depth_to_nodes, current_depth):
    current_layer_nodes = depth_to_nodes.get(current_depth, [])
    if not current_layer_nodes:
        return None
    partition_dict = {i: [] for i in range(len(current_layer_nodes))}
    for leaf, ancestors in leaf_to_ancestors.items():
        current_ancestor = None
        for ancestor in ancestors:
            if ancestor in current_layer_nodes:
                current_ancestor = ancestor
                break
                
        if current_ancestor is not None:
            partition_idx = current_layer_nodes.index(current_ancestor)
            partition_dict[partition_idx].append(leaf)

    total_nodes = sum(len(nodes) for nodes in partition_dict.values())
    empty_partitions = [idx for idx, nodes in partition_dict.items() if not nodes]
    return partition_dict

def get_all_partitions(T, leaf_to_ancestors):
    depth_to_nodes = {}
    max_depth = 0
    for node_id, node in T.items():
        depth = node['depth']
        if depth not in depth_to_nodes:
            depth_to_nodes[depth] = []
        depth_to_nodes[depth].append(node_id)
        max_depth = max(max_depth, depth)
    partitions_per_layer = []
    for depth in range(max_depth + 1):
        partition_dict = extract_partitions(leaf_to_ancestors, depth_to_nodes, depth)
        if partition_dict is not None:
            partitions_per_layer.append(partition_dict)
    
    return partitions_per_layer

def update_node(tree_node):
    root_id = None
    for i, node in tree_node.items():
        if node.parent is None:
            root_id = i
            break
    
    if root_id is None:
        return {}
    
    Tree = {
        i: {
            'children': list(node.children) if node.children else [],
            'partition': node.partition,
            'parent': node.parent,
            'ID': i,
            'depth': 0 
        } for i, node in tree_node.items()
    }
    Tree[root_id]['depth'] = 0  
    queue = [(root_id, 0)]
    visited = set([root_id])
    
    while queue:
        node_id, current_depth = queue.pop(0)
        for child_id in Tree[node_id]['children']:
            if child_id in Tree: 
                Tree[child_id]['depth'] = current_depth + 1
                if child_id not in visited:
                    visited.add(child_id)
                    queue.append((child_id, current_depth + 1))

    depth_counts = {}
    for node_id, node in Tree.items():
        depth = node['depth']
        if depth not in depth_counts:
            depth_counts[depth] = 0
        depth_counts[depth] += 1

    max_depth = max(depth_counts.keys()) if depth_counts else 0
    return Tree

def load_data(use_semantic):
    start_time = time.time()
    nodes_df = pd.read_csv(NODES_FILE)
    num_nodes = len(nodes_df)
    node_ids = list(nodes_df['node_id'])
    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    idx_to_node_id = {idx: node_id for idx, node_id in enumerate(node_ids)}
    edges_df = pd.read_csv(EDGES_FILE)
    rows = []
    cols = []
    for _, row in edges_df.iterrows():
        src_idx = node_id_to_idx.get(row['src'])
        dst_idx = node_id_to_idx.get(row['dst'])
        if src_idx is not None and dst_idx is not None:
            rows.append(src_idx)
            cols.append(dst_idx)
            rows.append(dst_idx)
            cols.append(src_idx)

    data = [1] * len(rows)
    adj_matrix = csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    edge_index = torch.tensor([rows, cols], dtype=torch.long, device=device)
    node_embeddings = {}
    if use_semantic:
        try:
            node_emb_df = pd.read_csv(NODES_EMB_FILE)
            all_embeddings = []
            valid_indices = []
            for _, row in node_emb_df.iterrows():
                node_id = row['node_id']
                if node_id in node_id_to_idx:
                    idx = node_id_to_idx[node_id]
                    embedding_str = row['node_embedding']
                    if isinstance(embedding_str, str) and embedding_str.startswith('[') and embedding_str.endswith(']'):
                        embedding_str = embedding_str[1:-1]
                        embedding_values = [float(x.strip()) for x in embedding_str.split(',')]
                        all_embeddings.append(embedding_values)
                        valid_indices.append(idx)
            
            if all_embeddings:
                all_embeddings = np.array(all_embeddings)
                pca = PCA(n_components=EMBED_DIM)
                reduced_embeddings = pca.fit_transform(all_embeddings)
                for idx, embedding in zip(valid_indices, reduced_embeddings):
                    node_embeddings[idx] = torch.tensor(embedding, dtype=torch.float32, device=device)

            else:
                use_semantic = False
                
        except Exception as e:
            use_semantic = False
    
    end_time = time.time()

    return adj_matrix, node_embeddings, edge_index, node_id_to_idx, idx_to_node_id, nodes_df

def save_partitions(partitions_per_layer, nodes_df, node_id_to_idx, idx_to_node_id):
    if not partitions_per_layer:
        num_nodes = len(node_id_to_idx)
        default_partition = {0: list(range(num_nodes))}
        partitions_per_layer = [default_partition]
    for layer, partitions in enumerate(partitions_per_layer):
        if not partitions:
            continue

        partition_file = f"{PARTITION_PATH}/layer_{layer+1}_partitions.csv"
        with open(partition_file, 'w') as f:
            f.write("partition_id,node_id\n")
            for partition_id, nodes in partitions.items():
                for node in nodes:
                    if isinstance(node, int) and node < len(nodes_df):
                        original_node_id = idx_to_node_id.get(node, node)
                    else:
                        original_node_id = node
                    f.write(f"{partition_id},{original_node_id}\n")

def main():
    start_time = time.time()
    adj_matrix, node_embeddings, edge_index, node_id_to_idx, idx_to_node_id, nodes_df = load_data(USE_SEMANTIC)
    if USE_SEMANTIC and not node_embeddings:
        alpha = 1.0
        use_semantic = False
    else:
        alpha = ALPHA if USE_SEMANTIC else 1.0
        use_semantic = USE_SEMANTIC

    init_start = time.time()
    partition_tree = PartitionTree(adj_matrix=adj_matrix, 
                                 node_embeddings=node_embeddings,
                                 alpha=alpha,
                                 bandwidth=BANDWIDTH)

    build_start = time.time()
    max_partition_size = max(int(adj_matrix.shape[0] * 0.5), 1000)
    partition_tree.build_coding_tree(
        k=TREE_DEPTH, 
        balance_factor=0.1,
        max_size_ratio=1000.0, 
        max_partition_size=max_partition_size
    )
    update_start = time.time()
    T = update_node(partition_tree.tree_node)

    extract_start = time.time()
    _, leaf_to_ancestors = extract_cluster_assignment(T, tree_depth=TREE_DEPTH)

    partition_start = time.time()
    partitions_per_layer = get_all_partitions(T, leaf_to_ancestors)

    save_start = time.time()
    save_partitions(partitions_per_layer, nodes_df, node_id_to_idx, idx_to_node_id)
    
    end_time = time.time()

if __name__ == "__main__":
    main()
