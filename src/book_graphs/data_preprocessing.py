import pandas as pd
import os
import pandas as pd
import numpy as np
import json
import argparse
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def encode_text(text, remove_position=True):
    if not isinstance(text, str):
        return np.zeros(384).tolist()
    
    if remove_position:
        parts = text.split(';')
        semantic_parts = [p for p in parts if not p.strip().startswith('(x,y,w,h)')]
        cleaned_text = '; '.join(semantic_parts)
        return model.encode(cleaned_text).tolist()
    else:
        return model.encode(text).tolist()

def preprocess_nodes(input_folder, output_folder, remove_position=True):
    os.makedirs(output_folder, exist_ok=True)
    
    for file in tqdm(os.listdir(input_folder)):
        if file.endswith(".csv"):
            input_path = os.path.join(input_folder, file)
            output_path = os.path.join(output_folder, file)

            df = pd.read_csv(input_path)
            df['node_embedding'] = df['node_attr'].apply(lambda x: encode_text(x, remove_position))

            df.to_csv(output_path, index=False)

def preprocess_edges(input_folder, output_folder, remove_position=True):
    os.makedirs(output_folder, exist_ok=True)
    
    for file in tqdm(os.listdir(input_folder)):
        if file.endswith(".csv"):
            input_path = os.path.join(input_folder, file)
            output_path = os.path.join(output_folder, file)

            df = pd.read_csv(input_path)
            if 'edge_attr' in df.columns:
                df['edge_embedding'] = df['edge_attr'].apply(lambda x: encode_text(x, remove_position))
            else:
                df['edge_embedding'] = [[] for _ in range(len(df))]

            df.to_csv(output_path, index=False)

def process_book_graphs_dataset(dataset_path="dataset/book_graphs"):
    input_path = os.path.join(dataset_path, "nodes.csv")
    output_path = os.path.join(dataset_path, "nodes_embedded.csv")
    
    if os.path.exists(input_path):
        df = pd.read_csv(input_path)
        df['node_embedding'] = df['node_attr'].apply(lambda x: encode_text(x, remove_position=False))
        df.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="dataset/webqsp")
    parser.add_argument("--dataset_type", type=str, default="default", 
                        choices=["default", "book_graphs"])
    parser.add_argument("--process_edges", action="store_true")

    args = parser.parse_args()
    
    if args.dataset_type == "book_graphs":
        process_book_graphs_dataset(args.dataset_path)
    else:
        nodes_folder = os.path.join(args.dataset_path, "nodes")
        nodes_output_folder = os.path.join(args.dataset_path, "nodes_embedded")
        
        preprocess_nodes(nodes_folder, nodes_output_folder)
        
        if args.process_edges:
            edges_folder = os.path.join(args.dataset_path, "edges")
            edges_output_folder = os.path.join(args.dataset_path, "edges_embedded")
            
            preprocess_edges(edges_folder, edges_output_folder)
