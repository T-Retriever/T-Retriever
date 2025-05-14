import copy
import math
import heapq
import numba as nb
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine, cdist
from scipy.stats import gaussian_kde
import multiprocessing as mp
from functools import partial
import random
import torch


def get_id():
    i = 0
    while True:
        yield i
        i += 1


def graph_parse(adj_matrix):
    g_num_nodes = adj_matrix.shape[0]
    adj_table = {}
    VOL = 0
    node_vol = []
    for i in range(g_num_nodes):
        n_v = 0
        adj = set()
        for j in range(g_num_nodes):
            if adj_matrix[i, j] != 0:
                n_v += adj_matrix[i, j]
                VOL += adj_matrix[i, j]
                adj.add(j)
        adj_table[i] = adj
        node_vol.append(n_v)
    return g_num_nodes, VOL, node_vol, adj_table


@nb.jit(nopython=True)
def cut_volume(adj_matrix, p1, p2):
    c12 = 0
    for i in range(len(p1)):
        for j in range(len(p2)):
            c = adj_matrix[p1[i], p2[j]]
            if c != 0:
                c12 += c
    return c12


def LayerFirst(node_dict, start_id):
    stack = [start_id]
    while len(stack) != 0:
        node_id = stack.pop(0)
        yield node_id
        if node_dict[node_id].children:
            for c_id in node_dict[node_id].children:
                stack.append(c_id)


def merge(new_ID, id1, id2, cut_v, node_dict):
    new_partition = node_dict[id1].partition + node_dict[id2].partition
    v = node_dict[id1].vol + node_dict[id2].vol
    g = node_dict[id1].g + node_dict[id2].g - 2 * cut_v
    child_h = max(node_dict[id1].child_h, node_dict[id2].child_h) + 1
    new_node = PartitionTreeNode(ID=new_ID, partition=new_partition, children={id1, id2},
                                 g=g, vol=v, child_h=child_h, child_cut=cut_v)
    node_dict[id1].parent = new_ID
    node_dict[id2].parent = new_ID
    node_dict[new_ID] = new_node


def compressNode(node_dict, node_id, parent_id):
    p_child_h = node_dict[parent_id].child_h
    node_children = node_dict[node_id].children
    node_dict[parent_id].child_cut += node_dict[node_id].child_cut
    node_dict[parent_id].children.remove(node_id)
    node_dict[parent_id].children = node_dict[parent_id].children.union(
        node_children)
    for c in node_children:
        node_dict[c].parent = parent_id
    com_node_child_h = node_dict[node_id].child_h
    node_dict.pop(node_id)

    if (p_child_h - com_node_child_h) == 1:
        while True:
            max_child_h = max(
                [node_dict[f_c].child_h for f_c in node_dict[parent_id].children])
            if node_dict[parent_id].child_h == (max_child_h + 1):
                break
            node_dict[parent_id].child_h = max_child_h + 1
            parent_id = node_dict[parent_id].parent
            if parent_id is None:
                break


def child_tree_deepth(node_dict, nid):
    node = node_dict[nid]
    deepth = 0
    while node.parent is not None:
        node = node_dict[node.parent]
        deepth += 1
    deepth += node_dict[nid].child_h
    return deepth


def CompressDelta(node1, p_node):
    a = node1.child_cut
    v1 = node1.vol
    v2 = p_node.vol

    # 添加安全检查，避免对数计算错误
    try:
        if v1 <= 0 or v2 <= 0:
            return 0
        return a * math.log(v2 / v1)
    except (ValueError, ZeroDivisionError):
        return 0


class PartitionTreeNode():
    def __init__(self, ID, partition, vol, g, children: set = None, parent=None, child_h=0, child_cut=0):
        self.ID = ID
        self.partition = partition
        self.parent = parent
        self.children = children
        self.vol = vol
        self.g = g
        self.merged = False
        self.child_h = child_h
        self.child_cut = child_cut

    def __str__(self):
        return "{" + "{}:{}".format(self.__class__.__name__, self.gatherAttrs()) + "}"

    def gatherAttrs(self):
        return ",".join("{}={}"
                        .format(k, getattr(self, k))
                        for k in self.__dict__.keys())


class PartitionTree():

    def __init__(self, adj_matrix, node_embeddings=None, edge_index=None, alpha=0.5, bandwidth=0.5):
        self.adj_matrix = adj_matrix
        self.tree_node = {}
        self.g_num_nodes, self.VOL, self.node_vol, self.adj_table = graph_parse(
            adj_matrix)
        self.id_g = get_id()
        self.leaves = []
        self.build_leaves()

        self.node_embeddings = node_embeddings
        self.edge_index = edge_index
        self.alpha = alpha 
        self.bandwidth = bandwidth  
        self.epsilon = 1e-10

        self._semantic_cache = {}
        self._consistency_cache = {}
        self._structure_cache = {}

        self.base_structure_threshold = 0.01 
        self.large_partition_threshold = 100 
        self.sampling_size = 100 
        self.parallel_threshold = 50  
        self.cache_block_size = 100 

        self.enable_semantic = node_embeddings is not None 

        self._pool = None  

        self._precomputed = False

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def build_leaves(self):
        for vertex in range(self.g_num_nodes):
            ID = next(self.id_g)
            v = self.node_vol[vertex]
            leaf_node = PartitionTreeNode(
                ID=ID, partition=[vertex], g=v, vol=v)
            self.tree_node[ID] = leaf_node
            self.leaves.append(ID)

    def build_sub_leaves(self, node_list, p_vol):
        subgraph_node_dict = {}
        ori_ent = 0
        for vertex in node_list:
            try:
                if self.tree_node[vertex].g <= 0 or self.VOL <= 0 or self.tree_node[vertex].vol <= 0 or p_vol <= 0:
                    continue
                ori_ent += -(self.tree_node[vertex].g / self.VOL)\
                    * math.log2(self.tree_node[vertex].vol / p_vol)
            except (ValueError, ZeroDivisionError):
                continue

            sub_n = set()
            vol = 0
            for vertex_n in node_list:
                c = self.adj_matrix[vertex, vertex_n]
                if c != 0:
                    vol += c
                    sub_n.add(vertex_n)
            sub_leaf = PartitionTreeNode(
                ID=vertex, partition=[vertex], g=vol, vol=vol)
            subgraph_node_dict[vertex] = sub_leaf
            self.adj_table[vertex] = sub_n

        return subgraph_node_dict, ori_ent

    def build_root_down(self):
        root_child = self.tree_node[self.root_id].children
        subgraph_node_dict = {}
        ori_en = 0
        g_vol = self.tree_node[self.root_id].vol
        for node_id in root_child:
            node = self.tree_node[node_id]
            try:
                if node.g <= 0 or g_vol <= 0 or node.vol <= 0:
                    continue
                ori_en += -(node.g / g_vol) * math.log2(node.vol / g_vol)
            except (ValueError, ZeroDivisionError):
                continue

            new_n = set()
            for nei in self.adj_table[node_id]:
                if nei in root_child:
                    new_n.add(nei)
            self.adj_table[node_id] = new_n

            new_node = PartitionTreeNode(
                ID=node_id, partition=node.partition, vol=node.vol, g=node.g, children=node.children)
            subgraph_node_dict[node_id] = new_node

        return subgraph_node_dict, ori_en

    def entropy(self, node_dict=None):
        if node_dict is None:
            node_dict = self.tree_node
        ent = 0
        for node_id, node in node_dict.items():
            if node.parent is not None:
                node_p = node_dict[node.parent]
                node_vol = node.vol
                node_g = node.g
                node_p_vol = node_p.vol
                try:
                    if node_vol <= 0 or node_p_vol <= 0 or self.VOL <= 0:
                        continue
                    ent += - (node_g / self.VOL) * \
                        math.log2(node_vol / node_p_vol)
                except (ValueError, ZeroDivisionError):
                    continue
        return ent

    def parallel_compute_gains(self, node_pairs, nodes_dict, adj_matrix, g_vol, structure_threshold):
        results = []
        for id1, id2 in node_pairs:
            n1 = nodes_dict[id1]
            n2 = nodes_dict[id2]
            try:
                p1 = np.array(n1.partition)
                p2 = np.array(n2.partition)
                cut_v = np.sum(adj_matrix[p1][:, p2])
                v1, v2 = n1.vol, n2.vol
                g1, g2 = n1.g, n2.g
                v12 = v1 + v2
                if v1 <= 0 or v2 <= 0 or v12 <= 0 or g_vol <= 0:
                    continue
                log_term1 = np.log2(v12 / v1) if v12 > v1 > 0 else 0
                log_term2 = np.log2(v12 / v2) if v12 > v2 > 0 else 0
                log_term3 = np.log2(g_vol / v12) if g_vol > v12 > 0 else 0

                structure_delta = ((v1 - g1) * log_term1 +
                                   (v2 - g2) * log_term2 -
                                   2 * cut_v * log_term3) / g_vol

                if structure_delta <= structure_threshold:
                    results.append((structure_delta, id1, id2, cut_v))
            except Exception as e:
                print(
                    f"Warning: Error in computing gains for nodes {id1} and {id2}: {e}")
                continue
        return results

    def __build_k_tree(self, g_vol, nodes_dict: dict, k=None):
        min_heap = []
        cmp_heap = []
        nodes_ids = list(nodes_dict.keys())
        new_id = None
        self.precompute_semantic_entropy(nodes_dict)
        num_processes = mp.cpu_count()
        pool = mp.Pool(num_processes)
        node_pairs = []
        for i, id1 in enumerate(nodes_ids):
            for id2 in self.adj_table[id1]:
                if id2 > id1:
                    node_pairs.append((id1, id2))
        batch_size = max(1, len(node_pairs) // (num_processes * 4))
        batches = [node_pairs[i:i + batch_size]
                   for i in range(0, len(node_pairs), batch_size)]
        compute_func = partial(self.parallel_compute_gains,
                               nodes_dict=nodes_dict,
                               adj_matrix=self.adj_matrix,
                               g_vol=g_vol,
                               structure_threshold=self.base_structure_threshold)

        results = []
        for batch in batches:
            results.extend(compute_func(batch))
        for result in results:
            heapq.heappush(min_heap, result)

        def process_new_node_edges(new_id, adj_nodes):
            results = []
            adj_nodes_array = np.array(list(adj_nodes))
            n2 = nodes_dict[new_id]
            p2 = np.array(n2.partition)

            for i in range(0, len(adj_nodes_array), 100):
                batch_nodes = adj_nodes_array[i:i+100]
                batch_partitions = [
                    np.array(nodes_dict[ID].partition) for ID in batch_nodes]

                for j, ID in enumerate(batch_nodes):
                    if nodes_dict[ID].merged:
                        continue

                    try:
                        n1 = nodes_dict[ID]
                        p1 = batch_partitions[j]

                        cut_v = np.sum(self.adj_matrix[p1][:, p2])

                        partition_size = len(p1) + len(p2)
                        if partition_size <= 0:
                            continue
                        dynamic_threshold = self.base_structure_threshold * \
                            (1 + math.log(partition_size))

                        v1, v2 = n1.vol, n2.vol
                        g1, g2 = n1.g, n2.g
                        v12 = v1 + v2

                        if v1 <= 0 or v2 <= 0 or v12 <= 0 or g_vol <= 0:
                            continue
                        log_term1 = np.log2(v12 / v1) if v12 > v1 > 0 else 0
                        log_term2 = np.log2(v12 / v2) if v12 > v2 > 0 else 0
                        log_term3 = np.log2(
                            g_vol / v12) if g_vol > v12 > 0 else 0

                        structure_delta = ((v1 - g1) * log_term1 +
                                           (v2 - g2) * log_term2 -
                                           2 * cut_v * log_term3) / g_vol

                        if structure_delta <= dynamic_threshold:
                            new_diff = self.CombineDelta(n1, n2, cut_v, g_vol)
                            results.append((new_diff, ID, new_id, cut_v))

                    except Exception as e:
                        print(
                            f"Warning: Error in processing edge between nodes {ID} and {new_id}: {e}")
                        continue

            return results

        unmerged_count = len(nodes_ids)
        merge_count = 0

        while unmerged_count > 1:
            if len(min_heap) == 0:
                break

            diff, id1, id2, cut_v = heapq.heappop(min_heap)
            if nodes_dict[id1].merged or nodes_dict[id2].merged:
                continue

            merge_count += 1
            if merge_count % 100 == 0:
                pass

            nodes_dict[id1].merged = True
            nodes_dict[id2].merged = True
            new_id = next(self.id_g)
            merge(new_id, id1, id2, cut_v, nodes_dict)
            self.adj_table[new_id] = self.adj_table[id1].union(
                self.adj_table[id2])
            for i in self.adj_table[new_id]:
                self.adj_table[i].add(new_id)
            if nodes_dict[id1].child_h > 0:
                heapq.heappush(cmp_heap, [CompressDelta(
                    nodes_dict[id1], nodes_dict[new_id]), id1, new_id])
            if nodes_dict[id2].child_h > 0:
                heapq.heappush(cmp_heap, [CompressDelta(
                    nodes_dict[id2], nodes_dict[new_id]), id2, new_id])
            unmerged_count -= 1

            adj_nodes = list(self.adj_table[new_id])
            total_adj = len(adj_nodes)

            results = process_new_node_edges(new_id, adj_nodes)
            new_edges = len(results)
            for result in results:
                heapq.heappush(min_heap, result)

        root = new_id

        if unmerged_count > 1:
            assert len(min_heap) == 0
            unmerged_nodes = {i for i, j in nodes_dict.items() if not j.merged}
            new_child_h = max(
                [nodes_dict[i].child_h for i in unmerged_nodes]) + 1

            new_id = next(self.id_g)
            new_node = PartitionTreeNode(ID=new_id, partition=list(nodes_ids), children=unmerged_nodes,
                                         vol=g_vol, g=0, child_h=new_child_h)
            nodes_dict[new_id] = new_node

            for i in unmerged_nodes:
                nodes_dict[i].merged = True
                nodes_dict[i].parent = new_id
                if nodes_dict[i].child_h > 0:
                    heapq.heappush(cmp_heap, [CompressDelta(
                        nodes_dict[i], nodes_dict[new_id]), i, new_id])
            root = new_id

        if k is not None:
            compress_count = 0
            while nodes_dict[root].child_h > k:
                compress_count += 1
                if compress_count % 100 == 0:
                    pass
                diff, node_id, p_id = heapq.heappop(cmp_heap)
                if child_tree_deepth(nodes_dict, node_id) <= k:
                    continue
                children = nodes_dict[node_id].children
                compressNode(nodes_dict, node_id, p_id)
                if nodes_dict[root].child_h == k:
                    break
                for e in cmp_heap:
                    if e[1] == p_id:
                        if child_tree_deepth(nodes_dict, p_id) > k:
                            e[0] = CompressDelta(
                                nodes_dict[e[1]], nodes_dict[e[2]])
                    if e[1] in children:
                        if nodes_dict[e[1]].child_h == 0:
                            continue
                        if child_tree_deepth(nodes_dict, e[1]) > k:
                            e[2] = p_id
                            e[0] = CompressDelta(
                                nodes_dict[e[1]], nodes_dict[p_id])
                heapq.heapify(cmp_heap)
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

        return root

    def check_balance(self, node_dict, root_id):
        root_c = copy.deepcopy(node_dict[root_id].children)
        for c in root_c:
            if node_dict[c].child_h == 0:
                self.single_up(node_dict, c)

    def single_up(self, node_dict, node_id):
        new_id = next(self.id_g)
        p_id = node_dict[node_id].parent
        grow_node = PartitionTreeNode(ID=new_id, partition=node_dict[node_id].partition, parent=p_id,
                                      children={node_id}, vol=node_dict[node_id].vol, g=node_dict[node_id].g)
        node_dict[node_id].parent = new_id
        node_dict[p_id].children.remove(node_id)
        node_dict[p_id].children.add(new_id)
        node_dict[new_id] = grow_node
        node_dict[new_id].child_h = node_dict[node_id].child_h + 1

        self.adj_table[new_id] = set(self.adj_table[node_id])
        for i in list(self.adj_table[node_id]):
            self.adj_table[i].add(new_id)

    def root_down_delta(self):
        if len(self.tree_node[self.root_id].children) < 3:
            return 0, None, None
        subgraph_node_dict, ori_entropy = self.build_root_down()
        g_vol = self.tree_node[self.root_id].vol
        new_root = self.__build_k_tree(
            g_vol=g_vol, nodes_dict=subgraph_node_dict, k=2)
        self.check_balance(subgraph_node_dict, new_root)

        new_entropy = self.entropy(subgraph_node_dict)
        delta = (ori_entropy - new_entropy) / \
            len(self.tree_node[self.root_id].children)
        return delta, new_root, subgraph_node_dict

    def leaf_up_entropy(self, sub_node_dict, sub_root_id, node_id):
        ent = 0
        for sub_node_id in LayerFirst(sub_node_dict, sub_root_id):
            if sub_node_id == sub_root_id:
                sub_node_dict[sub_root_id].vol = self.tree_node[node_id].vol
                sub_node_dict[sub_root_id].g = self.tree_node[node_id].g

            elif sub_node_dict[sub_node_id].child_h == 1:
                node = sub_node_dict[sub_node_id]
                inner_vol = node.vol - node.g
                partition = node.partition
                ori_vol = sum(self.tree_node[i].vol for i in partition)
                ori_g = ori_vol - inner_vol
                node.vol = ori_vol
                node.g = ori_g
                node_p = sub_node_dict[node.parent]
                try:
                    if node.g <= 0 or self.VOL <= 0 or node.vol <= 0 or node_p.vol <= 0:
                        continue
                    ent += -(node.g / self.VOL) * \
                        math.log2(node.vol / node_p.vol)
                except (ValueError, ZeroDivisionError):
                    continue
            else:
                node = sub_node_dict[sub_node_id]
                node.g = self.tree_node[sub_node_id].g
                node.vol = self.tree_node[sub_node_id].vol
                node_p = sub_node_dict[node.parent]
                try:
                    if node.g <= 0 or self.VOL <= 0 or node.vol <= 0 or node_p.vol <= 0:
                        continue
                    ent += -(node.g / self.VOL) * \
                        math.log2(node.vol / node_p.vol)
                except (ValueError, ZeroDivisionError):
                    continue
        return ent

    def leaf_up(self):
        h1_id = set()
        h1_new_child_tree = {}
        id_mapping = {}
        for l in self.leaves:
            p = self.tree_node[l].parent
            h1_id.add(p)
        delta = 0
        for node_id in h1_id:
            candidate_node = self.tree_node[node_id]
            sub_nodes = candidate_node.partition
            if len(sub_nodes) == 1:
                id_mapping[node_id] = None
            if len(sub_nodes) == 2:
                id_mapping[node_id] = None
            if len(sub_nodes) >= 3:
                sub_g_vol = candidate_node.vol - candidate_node.g
                subgraph_node_dict, ori_ent = self.build_sub_leaves(
                    sub_nodes, candidate_node.vol)
                sub_root = self.__build_k_tree(
                    g_vol=sub_g_vol, nodes_dict=subgraph_node_dict, k=2)
                self.check_balance(subgraph_node_dict, sub_root)
                new_ent = self.leaf_up_entropy(
                    subgraph_node_dict, sub_root, node_id)
                delta += (ori_ent - new_ent)
                h1_new_child_tree[node_id] = subgraph_node_dict
                id_mapping[node_id] = sub_root
        delta = delta / self.g_num_nodes
        return delta, id_mapping, h1_new_child_tree

    def leaf_up_update(self, id_mapping, leaf_up_dict):
        for node_id, h1_root in id_mapping.items():
            if h1_root is None:
                children = copy.deepcopy(self.tree_node[node_id].children)
                for i in children:
                    self.single_up(self.tree_node, i)
            else:
                h1_dict = leaf_up_dict[node_id]
                self.tree_node[node_id].children = h1_dict[h1_root].children
                for h1_c in h1_dict[h1_root].children:
                    assert h1_c not in self.tree_node
                    h1_dict[h1_c].parent = node_id
                h1_dict.pop(h1_root)
                self.tree_node.update(h1_dict)
        self.tree_node[self.root_id].child_h += 1

    def root_down_update(self, new_id, root_down_dict):
        self.tree_node[self.root_id].children = root_down_dict[new_id].children
        for node_id in root_down_dict[new_id].children:
            assert node_id not in self.tree_node
            root_down_dict[node_id].parent = self.root_id
        root_down_dict.pop(new_id)
        self.tree_node.update(root_down_dict)
        self.tree_node[self.root_id].child_h += 1

    def build_coding_tree(self, k=2, mode='v2'):
        if k == 1:
            return
        if mode == 'v1' or k is None:
            self.root_id = self.__build_k_tree(self.VOL, self.tree_node, k=k)
        elif mode == 'v2':
            self.root_id = self.__build_k_tree(self.VOL, self.tree_node, k=2)
            self.check_balance(self.tree_node, self.root_id)

            if self.tree_node[self.root_id].child_h < 2:
                self.tree_node[self.root_id].child_h = 2

            flag = 0
            current_height = self.tree_node[self.root_id].child_h

            while self.tree_node[self.root_id].child_h < k:
                current_height = self.tree_node[self.root_id].child_h

                if flag == 0:
                    leaf_up_delta, id_mapping, leaf_up_dict = self.leaf_up()
                    root_down_delta, new_id, root_down_dict = self.root_down_delta()

                elif flag == 1:
                    leaf_up_delta, id_mapping, leaf_up_dict = self.leaf_up()

                elif flag == 2:
                    root_down_delta, new_id, root_down_dict = self.root_down_delta()
                else:
                    raise ValueError

                if leaf_up_delta < root_down_delta:
                    flag = 2
                    self.root_down_update(new_id, root_down_dict)
                else:
                    flag = 1
                    self.leaf_up_update(id_mapping, leaf_up_dict)

                    if root_down_delta != 0:
                        for root_down_id, root_down_node in root_down_dict.items():
                            if root_down_node.child_h == 0:
                                root_down_node.children = self.tree_node[root_down_id].children

        count = 0
        for _ in LayerFirst(self.tree_node, self.root_id):
            count += 1
        assert len(self.tree_node) == count

    def batch_cosine_similarity(self, X):
        try:
            if X is None or X.size == 0:
                return np.ones((1, 1)) 
            if np.isnan(X).any():
                X = np.nan_to_num(X)
            norms = np.linalg.norm(X, axis=1)
            zero_mask = norms < 1e-10  
            norms[zero_mask] = 1

            X_normalized = X / norms[:, np.newaxis]
            X_normalized[zero_mask] = 0
            similarity_matrix = np.dot(X_normalized, X_normalized.T)
            similarity_matrix = np.clip(similarity_matrix, -1.0, 1.0)

            return similarity_matrix
        except Exception as e:
            return np.ones((X.shape[0], X.shape[0]))

    def _init_parallel_pool(self):
        if self._pool is None:
            self._pool = mp.Pool(processes=mp.cpu_count())

    def _calculate_partial_semantic_entropy(self, partition_chunk):
        try:
            node_consistency = self.calculate_node_semantic_consistency(
                partition_chunk)
            p = len(partition_chunk) / max(1, self.g_num_nodes)
            S = node_consistency
            S = max(self.epsilon, S)  
            return -p * S * math.log2(S)
        except (ValueError, ZeroDivisionError):
            return 0

    def _sample_partition(self, partition, sample_size=None):
        if not sample_size:
            if len(partition) > self.large_partition_threshold:
                sample_size = min(self.sampling_size,
                                  int(math.sqrt(len(partition)) * math.log(len(partition))))
            else:
                return partition

        if len(partition) > sample_size:
            return random.sample(partition, sample_size)
        return partition

    def calculate_semantic_entropy(self, partition):
        if not partition or not self.enable_semantic:
            return 0
        partition_key = frozenset(partition)
        if partition_key in self._semantic_cache:
            return self._semantic_cache[partition_key]
        p = len(partition) / self.g_num_nodes

        if len(partition) > self.cache_block_size:
            sub_partitions = [partition[i:i+self.cache_block_size]
                              for i in range(0, len(partition), self.cache_block_size)]
            sub_entropies = []
            sub_weights = []

            for sub in sub_partitions:
                sub_p = len(sub) / self.g_num_nodes
                sub_weights.append(sub_p / p)  
                sub_entropy = self.calculate_semantic_entropy(sub)
                sub_entropies.append(sub_entropy)

            result = sum(w * e for w, e in zip(sub_weights, sub_entropies))
            self._semantic_cache[partition_key] = result
            return result
        node_consistency = self.calculate_node_semantic_consistency(partition)
        S = node_consistency
        S = max(self.epsilon, S)  
        result = -p * S * math.log2(S)
        self._semantic_cache[partition_key] = result
        return result

    def calculate_node_semantic_consistency(self, partition):
        if self.node_embeddings is None or not partition:
            return 1.0
        partition_key = frozenset(partition)
        cache_key = 'node_' + str(hash(partition_key))
        if cache_key in self._consistency_cache:
            return self._consistency_cache[cache_key]
        if len(partition) > self.large_partition_threshold:
            sample_size = min(self.sampling_size,
                              int(math.sqrt(len(partition)) * math.log(len(partition))))
            sampled_partition = random.sample(partition, sample_size)
        else:
            sampled_partition = partition

        try:
            if len(sampled_partition) > 50:  
                result = self.calculate_node_semantic_consistency_fast(
                    sampled_partition)
            else:
                valid_nodes = []
                for node_id in sampled_partition:
                    if node_id in self.node_embeddings:
                        valid_nodes.append(node_id)

                if not valid_nodes:
                    return 1.0

                embeddings = np.vstack([self.node_embeddings[i]
                                       for i in valid_nodes])
                similarities = self.batch_cosine_similarity(embeddings)
                mask = np.triu(np.ones_like(similarities), k=1).astype(bool)
                result = np.mean(similarities[mask]) if np.any(mask) else 1.0

        except Exception as e:
            result = 1.0
        self._consistency_cache[cache_key] = result
        return result

    def calculate_node_semantic_consistency_fast(self, partition):
        if self.node_embeddings is None or not partition:
            return 1.0

        try:
            valid_nodes = []
            for node_id in partition:
                if node_id in self.node_embeddings:
                    valid_nodes.append(node_id)

            if not valid_nodes:
                return 1.0

            embeddings = np.vstack([self.node_embeddings[i]
                                   for i in valid_nodes])

            norms = np.linalg.norm(embeddings, axis=1)
            zero_mask = norms == 0
            norms[zero_mask] = 1 

            embeddings_normalized = embeddings / norms[:, np.newaxis]
            embeddings_normalized[zero_mask] = 0

            vector_sum = np.sum(embeddings_normalized, axis=0)
            n = len(valid_nodes)
            if n <= 1:
                return 1.0

            sum_cos = (np.sum(vector_sum**2) - n) / 2

            num_pairs = n * (n - 1) / 2
            avg_cos = sum_cos / num_pairs

            return max(0, min(1, avg_cos)) 

        except Exception as e:
            return 1.0

    def calculate_semantic_entropy(self, embeddings, bandwidth=0.5):
        if embeddings is None or len(embeddings) == 0:
            return 0

        try:
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()

            kde = gaussian_kde(embeddings.T, bw_method=bandwidth)
            densities = kde(embeddings.T)

            entropy = -np.mean(np.log(densities))

            return entropy
        except Exception as e:
            print(
                f"Warning: Error in semantic density entropy calculation: {e}")
            return 0

    def CombineDelta(self, node1, node2, cut_v, g_vol):
        v1, v2 = node1.vol, node2.vol
        g1, g2 = node1.g, node2.g
        v12 = v1 + v2
        try:
            if v1 <= 0 or v2 <= 0 or v12 <= 0 or g_vol <= 0:
                return float('inf')

            structure_delta = ((v1 - g1) * math.log(v12 / v1, 2) +
                               (v2 - g2) * math.log(v12 / v2, 2) -
                               2 * cut_v * math.log(g_vol / v12, 2)) / g_vol
        except (ValueError, ZeroDivisionError):
            return float('inf')
        if not self.enable_semantic or self.node_embeddings is None:
            return structure_delta

        partition_size = len(node1.partition) + len(node2.partition)
        dynamic_threshold = self.base_structure_threshold * \
            (1 + math.log(partition_size) if partition_size > 0 else 1)

        if structure_delta > dynamic_threshold:
            return float('inf')

        try:
            embeddings1 = []
            for node_idx in node1.partition:
                if node_idx in self.node_embeddings:
                    embeddings1.append(self.node_embeddings[node_idx])

            embeddings2 = []
            for node_idx in node2.partition:
                if node_idx in self.node_embeddings:
                    embeddings2.append(self.node_embeddings[node_idx])
            if not embeddings1 or not embeddings2:
                return structure_delta
            embeddings1_matrix = np.vstack(embeddings1)
            embeddings2_matrix = np.vstack(embeddings2)

            semantic_entropy1 = self.calculate_semantic_entropy(
                embeddings1_matrix, self.bandwidth)
            semantic_entropy2 = self.calculate_semantic_entropy(
                embeddings2_matrix, self.bandwidth)
            p1 = len(node1.partition) / max(1, self.g_num_nodes)
            p2 = len(node2.partition) / max(1, self.g_num_nodes)
            p12 = p1 + p2

            weighted_entropy_before = (
                p1 * semantic_entropy1 + p2 * semantic_entropy2) / p12
            combined_embeddings = np.vstack(
                [embeddings1_matrix, embeddings2_matrix])

            semantic_entropy_after = self.calculate_semantic_entropy(
                combined_embeddings, self.bandwidth)

            semantic_delta = semantic_entropy_after - weighted_entropy_before

            delta = self.alpha * structure_delta + \
                (1 - self.alpha) * semantic_delta

            return delta

        except Exception as e:
            return structure_delta  

    def precompute_semantic_entropy(self, nodes_dict):
        if self._precomputed or not self.enable_semantic:
            return
        if len(nodes_dict) > 100:
            self._init_parallel_pool()
            tasks = []
            for node_id, node in nodes_dict.items():
                if not node.merged:
                    tasks.append(node.partition)
            batch_size = 50
            for i in range(0, len(tasks), batch_size):
                batch = tasks[i:i+batch_size]
                for partition in batch:
                    self.calculate_node_semantic_consistency(partition)
                    if self.node_embeddings is not None:
                        pass
        else:
            for node_id, node in nodes_dict.items():
                if not node.merged:
                    self.calculate_node_semantic_consistency(node.partition)
                    if self.node_embeddings is not None:
                        pass

        self._precomputed = True

    def total_entropy(self, node_dict=None):
        if node_dict is None:
            node_dict = self.tree_node
        structural_ent = self.entropy(node_dict)
        semantic_ent = 0
        if self.enable_semantic:
            for node_id, node in node_dict.items():
                if node.parent is not None and hasattr(node, 'partition'):
                    embeddings = []
                    for node_idx in node.partition:
                        if node_idx in self.node_embeddings:
                            embeddings.append(self.node_embeddings[node_idx])
                    if embeddings:
                        embeddings_matrix = np.vstack(embeddings)
                        partition_semantic_entropy = self.calculate_semantic_entropy(
                            embeddings_matrix, self.bandwidth)
                        weight = len(node.partition) / self.g_num_nodes
                        semantic_ent += weight * partition_semantic_entropy
        total_ent = self.alpha * structural_ent + \
            (1 - self.alpha) * semantic_ent

        return total_ent


def load_graph(dname):
    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}

    with open('datasets/%s/%s.txt' % (dname, dname.replace('-', '')), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if l not in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                g.add_node(j, tag=row[0])
                if tmp == len(row):
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array(
                        [float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
            else:
                node_features = None

            assert len(g) == n
            g_list.append({'G': g, 'label': l})
    return g_list
