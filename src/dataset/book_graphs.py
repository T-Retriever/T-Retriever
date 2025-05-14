import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
import os
from matplotlib.lines import Line2D

os.makedirs('dataset/book_graphs/output', exist_ok=True)
df = pd.read_csv('dataset/book_graphs/Children.csv')
label_counts = Counter(df['label'])
nodes_df = df[['node_id', 'category', 'text']].copy()
nodes_df['node_attr'] = nodes_df.apply(
    lambda row: f"Category: {row['category']}; {row['text']}", axis=1)
nodes_df = nodes_df[['node_id', 'node_attr']]
nodes_df.to_csv('dataset/book_graphs/output/nodes.csv', index=False)
edges = []
edge_set = set()

for _, row in df.iterrows():
    src = row['node_id']
    try:
        neighbours = ast.literal_eval(row['neighbour'])
        if isinstance(neighbours, (list, tuple)):
            for dst in neighbours:
                edge = (min(src, dst), max(src, dst))
                if edge not in edge_set:
                    edge_set.add(edge)
                    edges.append({'src': edge[0], 'dst': edge[1]})
        elif isinstance(neighbours, int):
            dst = neighbours
            edge = (min(src, dst), max(src, dst))
            if edge not in edge_set:
                edge_set.add(edge)
                edges.append({'src': edge[0], 'dst': edge[1]})
    except (ValueError, SyntaxError) as e:
        pass

edges_df = pd.DataFrame(edges)
edges_df.to_csv('dataset/book_graphs/output/edges.csv', index=False)