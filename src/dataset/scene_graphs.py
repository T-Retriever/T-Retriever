import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.data.data import Data
from src.utils.lm_modeling import load_model, load_text2embedding
from sklearn.model_selection import train_test_split


model_name = 'sbert'
path = 'dataset/scene_graphs'
path_nodes = f'{path}/nodes'
path_edges = f'{path}/edges'

def textualize_graph(data):
    objectid2nodeid = {object_id: idx for idx, object_id in enumerate(data['objects'].keys())}
    nodes = []
    edges = []
    for objectid, object in data['objects'].items():
        node_attr = f'name: {object["name"]}'
        x, y, w, h = object['x'], object['y'], object['w'], object['h']
        if len(object['attributes']) > 0:
            node_attr = node_attr + '; attribute: ' + (', ').join(object["attributes"])
        node_attr += '; (x,y,w,h): ' + str((x, y, w, h))
        nodes.append({'node_id': objectid2nodeid[objectid], 'node_attr': node_attr})
        for rel in object['relations']:
            src = objectid2nodeid[objectid]
            dst = objectid2nodeid[rel['object']]
            edge_attr = rel['name']
            edges.append({'src': src, 'edge_attr': edge_attr, 'dst': dst})

    return nodes, edges


def step_one():
    dataset = json.load(open('dataset/gqa/train_sceneGraphs.json'))

    os.makedirs(path_nodes, exist_ok=True)
    os.makedirs(path_edges, exist_ok=True)

    for imageid, object in tqdm(dataset.items(), total=len(dataset)):
        node_attr, edge_attr = textualize_graph(object)
        pd.DataFrame(node_attr, columns=['node_id', 'node_attr']).to_csv(f'{path_nodes}/{imageid}.csv', index=False)
        pd.DataFrame(edge_attr, columns=['src', 'edge_attr', 'dst']).to_csv(f'{path_edges}/{imageid}.csv', index=False)



if __name__ == '__main__':
    step_one()
