import json
from utils import PartitionTree
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
import os
import torch
import copy
from tqdm import tqdm


DATASET_PATH = 'dataset/webqsp_s'
NODES_FILE = f'{DATASET_PATH}/nodes'
EDGES_FILE = f'{DATASET_PATH}/edges'
NODES_EMB = f'{DATASET_PATH}/nodes_embedded'
PARTITION_PATH = f'{DATASET_PATH}/partitions'
TREE_DEPTH = 5
BETA = 0.5

class DataLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.nodes_data = {}
        self.edges_data = {}
        self.node_embeddings = {}
        
        
    def load_graph(self, graph_id):
        try:
            nodes_file = f'{NODES_FILE}/{graph_id}.csv'
            edges_file = f'{EDGES_FILE}/{graph_id}.csv'
            nodes_emb_file = f'{NODES_EMB}/{graph_id}.csv'
            
            for file_path in [nodes_file, edges_file, nodes_emb_file]:
                if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                    raise ValueError(f"File {file_path} does not exist or is empty")
            
            nodes_df = pd.read_csv(nodes_file)
            nodes_emb_df = pd.read_csv(nodes_emb_file)
            
            edges_df = pd.read_csv(edges_file)
            
            if len(nodes_df) == 0 or len(edges_df) == 0:
                raise ValueError(f"Graph {graph_id} has no nodes or edges")
            
            num_nodes = len(nodes_df)
            adj_matrix = np.zeros((num_nodes, num_nodes))
            edge_index = [[], []]
            
            for _, row in edges_df.iterrows():
                src, dst = int(row['src']), int(row['dst'])
                adj_matrix[src, dst] = 1
                adj_matrix[dst, src] = 1
                edge_index[0].extend([src, dst])
                edge_index[1].extend([dst, src])
            
            node_embeddings = {}
            for _, row in nodes_emb_df.iterrows():
                node_id = int(row['node_id'])
                try:
                    embedding = eval(row['node_embedding'])
                    node_embeddings[node_id] = np.array(embedding)
                except (SyntaxError, ValueError, TypeError) as e:
                    node_embeddings[node_id] = np.zeros(128)
            
            return {
                'adj_matrix': adj_matrix,
                'edge_index': edge_index,
                'node_embeddings': node_embeddings,
                'nodes_df': nodes_df,
                'edges_df': edges_df
            }
        except Exception as e:
            raise

def update_depth(tree):
    wait_update = [k for k, v in tree.items() if v.children is None]
    while wait_update:
        for nid in wait_update:
            node = tree[nid]
            if node.children is None:
                node.child_h = 0
            else:
                node.child_h = tree[list(node.children)[0]].child_h + 1
        wait_update = set([tree[nid].parent for nid in wait_update if tree[nid].parent])

def update_node(tree):
    update_depth(tree)
    
    d_id = [(v.child_h, v.ID) for k, v in tree.items()]
    d_id.sort()
    
    new_tree = {}
    for k, v in tree.items():
        n = copy.deepcopy(v)
        n.ID = d_id.index((n.child_h, n.ID))
        
        if n.parent is not None:
            n.parent = d_id.index((n.child_h+1, n.parent))
            
        if n.children is not None:
            n.children = [d_id.index((n.child_h-1, c)) for c in n.children]
            
        n = n.__dict__
        n['depth'] = n['child_h']
        new_tree[n['ID']] = n
        
    return new_tree

def extract_cluster_assignment(T, tree_depth=2):
    layer_idx = [0]
    for layer in range(tree_depth+1):
        layer_nodes = [i for i, n in T.items() if n['depth']==layer]
        layer_idx.append(layer_idx[-1] + len(layer_nodes))
    interLayerEdges = [[] for i in range(tree_depth+1)]
    for i, n in T.items():
        if n['depth'] == 0:
            continue
        n_idx = n['ID'] - layer_idx[n['depth']]
        c_base = layer_idx[n['depth']-1]
        interLayerEdges[n['depth']].extend([(n_idx, c-c_base) for c in n['children']])
    return interLayerEdges[1:]

def extract_partitions(S_edge_index, num_partitions, old_partition_dict):
    new_partition_dict = {i: [] for i in range(num_partitions)}
    
    for i in range(S_edge_index.shape[1]):
        target_partition = S_edge_index[0, i].item()
        node = S_edge_index[1, i].item()
        if old_partition_dict==None:
            new_partition_dict[target_partition].append(node)
        else:
            new_partition_dict[target_partition].extend(old_partition_dict[node])
    
    return new_partition_dict

def get_all_partitions(all_S_edge_indices):
    partitions_per_layer = []
    
    for i, S_edge_index in enumerate(all_S_edge_indices):
        num_partitions = len(set(S_edge_index[0].numpy()))
        if i==0:
            old_partition_dict = None
        else:
            old_partition_dict = partitions_per_layer[-1]
        new_partition_dict = extract_partitions(S_edge_index, num_partitions, old_partition_dict=old_partition_dict)
        partitions_per_layer.append(new_partition_dict)
    
    return partitions_per_layer

def convert_partitions_to_tensor(partitions_per_layer, num_nodes):
    partition_tensors = []
    
    for layer_partitions in partitions_per_layer:
        partition_tensor = torch.full((num_nodes,), -1, dtype=torch.long)
        
        for partition_id, nodes in layer_partitions.items():
            for node in nodes:
                if 0 <= node < num_nodes:
                    partition_tensor[node] = partition_id
        
        partition_tensors.append(partition_tensor)
    
    return partition_tensors

def process_graph(graph_id, tree_depth=TREE_DEPTH, beta=BETA):
    try:
        data_loader = DataLoader(DATASET_PATH)
        graph_data = data_loader.load_graph(graph_id)
        
        if graph_data['adj_matrix'].size == 0 or len(graph_data['node_embeddings']) == 0:
            raise ValueError(f"图 {graph_id} 数据无效：邻接矩阵为空或节点嵌入为空")
            
        try:
            partition_tree = PartitionTree(
                adj_matrix=graph_data['adj_matrix'],
                node_embeddings=graph_data['node_embeddings'],
                edge_index=graph_data['edge_index'],
                beta=beta
            )
            
            partition_tree.build_coding_tree(tree_depth)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise ValueError(f"error: {e}")
        
        T = update_node(partition_tree.tree_node)
        
        leaf_partitions = {i: [i] for i in range(graph_data['adj_matrix'].shape[0])}
        
        S_edge_index_list = extract_cluster_assignment(T, tree_depth=tree_depth)
        
        all_nodes = [(i, i) for i in range(graph_data['adj_matrix'].shape[0])]
        if len(all_nodes) == 0:
            raise ValueError(f"error: {graph_id} graph has no nodes")
            
        S_edge_index_list.insert(0, all_nodes)
        
        S_edge_index_tensors = [torch.LongTensor(S_edge_index_list[i]).T for i in range(len(S_edge_index_list))]
        
        partitions_per_layer = get_all_partitions(S_edge_index_tensors)
        
        num_nodes = graph_data['adj_matrix'].shape[0]
        partition_tensors = convert_partitions_to_tensor(partitions_per_layer, num_nodes)
        
        node_attrs = {}
        for _, row in graph_data['nodes_df'].iterrows():
            node_id = int(row['node_id'])
            node_attr = row['node_attr']
            node_attrs[node_id] = node_attr
        
        edge_attrs = {}
        for idx, row in graph_data['edges_df'].iterrows():
            src = int(row['src'])
            dst = int(row['dst'])
            edge_attr = row['edge_attr']
            edge_attrs[(src, dst)] = edge_attr
        
        return partition_tensors, partition_tree, node_attrs, edge_attrs
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise

def process_all_graphs(tree_depth=TREE_DEPTH, beta=BETA):
    os.makedirs(PARTITION_PATH, exist_ok=True)
    
    graph_files = [f for f in os.listdir(NODES_FILE) if f.endswith('.csv')]
    graph_ids = [int(f.split('.')[0]) for f in graph_files]
    graph_ids.sort()
    
    results = {}
    
    for graph_id in tqdm(graph_ids):
        try:
            partition_tensors, partition_tree, node_attrs, edge_attrs = process_graph(graph_id, tree_depth, beta)
            
            partition_result = {
                'partitions': partition_tensors,
                'node_attrs': node_attrs,
                'edge_attrs': edge_attrs
            }
            
            torch.save(partition_result, f'{PARTITION_PATH}/{graph_id}.pt')
            
            results[graph_id] = {
                'num_layers': len(partition_tensors),
                'num_nodes': len(node_attrs),
                'num_edges': len(edge_attrs),
                'partitions_summary': [
                    {
                        'layer': i,
                        'num_partitions': len(set(tensor.tolist())),
                        'partition_sizes': {
                            str(p_id): (tensor == p_id).sum().item() 
                            for p_id in set(tensor.tolist()) if p_id >= 0
                        }
                    }
                    for i, tensor in enumerate(partition_tensors)
                ]
            }
            
        except Exception as e:
            pass
    
    return results

def view_partition_result(graph_id):
    result_path = f'{PARTITION_PATH}/{graph_id}.pt'
    if not os.path.exists(result_path):
        return
    
    partition_result = torch.load(result_path)
    partitions = partition_result['partitions']
    node_attrs = partition_result['node_attrs']
    edge_attrs = partition_result['edge_attrs']

if __name__ == "__main__":
    results = process_all_graphs(tree_depth=TREE_DEPTH, beta=BETA)
    
    with open(f'{DATASET_PATH}/partition_summary.json', 'w') as f:
        json.dump(results, f)
    
    if len(results) > 0:
        first_graph_id = list(results.keys())[0]
        view_partition_result(first_graph_id)
