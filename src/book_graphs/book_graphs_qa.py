import os
import torch
import pandas as pd
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import re
import time
import difflib
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
from collections import defaultdict

class BookGraphsQA:
    
    def __init__(self, 
                 dataset_path='dataset/book_graphs',
                 embed_model='sentence-transformers/all-MiniLM-L6-v2',
                 llm_model='meta-llama/Llama-2-7b-chat-hf',
                 device=None,
                 cache_dir='cache',
                 use_cache=True,
                 top_k=3):
        self.dataset_path = dataset_path
        self.nodes_file = os.path.join(dataset_path, 'nodes.csv')
        self.edges_file = os.path.join(dataset_path, 'edges.csv')
        self.questions_file = os.path.join(dataset_path, 'question.csv')
        self.partition_dir = os.path.join(dataset_path, 'partitions')
        self.summary_dir = os.path.join(dataset_path, 'subgraph_summaries')
        self.qa_result_dir = os.path.join(dataset_path, 'qa_results')
        
        os.makedirs(self.summary_dir, exist_ok=True)
        os.makedirs(self.qa_result_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.embed_model = SentenceTransformer(embed_model)
        self.embed_model.to(self.device)
        
        access_token = "hf_XXXXXX"  # Replace with your Hugging Face access token
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model, token=access_token)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model,
            token=access_token,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.top_k = top_k
    
    def generate_summary_prompt(self, nodes_info, edges_info):
        processed_nodes = []
        for node in nodes_info:
            node_id = node['node_id']
            
            attr_str = str(node['attr'])
            attr_str = re.sub(r'Description:\s*([^;]+)(;|$)', r'\1\2', attr_str)
            
            node_desc = f"Node ID: {node_id}, Attributes: {attr_str.strip()}"
            processed_nodes.append(node_desc)
        
        processed_edges = []
        for edge in edges_info:
            edge_desc = f"Relation: Node{edge['src']} --[{edge['rel']}]--> Node{edge['dst']}"
            processed_edges.append(edge_desc)
        prompt = f"""You are an expert graph analyzer tasked with creating a detailed summary of subgraphs.

Below is a subgraph extracted from a larger knowledge graph or scene graph. The nodes in the graph are information about the book, such as category, title, etc., and the edges indicate two books that are often purchased together or browsed together. Please analyze this subgraph carefully and create a comprehensive summary that captures:
1. The key entities (nodes) present in the subgraph
2. The important relationships between these entities
3. The overall meaning or context represented by this subgraph

Node Information:
{chr(10).join(processed_nodes)}

Relationship Information:
{chr(10).join(processed_edges)}

Your summary should be factual, detailed, and focus on all available information in the subgraph. Include specific attributes and relationships that might be important for answering questions about this subgraph later.

Please output only the summary itself without additional explanations or formatting.

Subgraph Summary:"""
        
        return prompt
    
    def generate_qa_prompt(self, question, subgraph_summaries):
        prompt = f"""You are an expert question-answering system that provides precise answers based on graph data summaries.

I will provide you with:
1. Multiple subgraph summaries that might contain relevant information
2. A question to answer

Your task is to carefully analyze the information in these subgraph summaries and provide a concise, accurate answer to the question.

SUBGRAPH SUMMARIES:
"""
        
        for i, summary in enumerate(subgraph_summaries, 1):
            prompt += f"\nSummary {i}:\n{summary['summary']}\n"
        
        prompt += f"""
QUESTION: {question}

EXPECTED ANSWER FORMAT:
- Answer should be short and direct
- Focus only on information present in the subgraph summaries
- Be specific and avoid vague statements

Answer:"""
        
        return prompt
    
    def generate_verification_prompt(self, question, generated_answer, standard_answer):
        prompt = f"""As an expert evaluator, your task is to determine whether a generated answer correctly responds to a question based on a standard reference answer.

QUESTION: {question}

GENERATED ANSWER: {generated_answer}

REFERENCE ANSWER: {standard_answer}

Evaluate whether the GENERATED ANSWER is semantically correct compared to the REFERENCE ANSWER. 
Consider the following:
1. The generated answer might use different wording but convey the same meaning
2. The generated answer might be more specific or detailed than the reference
3. The generated answer might correctly address only part of the reference answer

Please respond with ONLY "Correct" or "Incorrect".

Evaluation:"""
        
        return prompt
    
    def load_graph_data(self, graph_id=0):
        try:
            if not os.path.exists(self.nodes_file):
                raise FileNotFoundError(f"节点文件不存在: {self.nodes_file}")
            nodes_df = pd.read_csv(self.nodes_file)
            
            if not os.path.exists(self.edges_file):
                edges_df = pd.DataFrame(columns=['src', 'dst', 'rel'])
            else:
                edges_df = pd.read_csv(self.edges_file)
            
            return nodes_df, edges_df
        except Exception as e:
            import traceback
            traceback.print_exc()
            return pd.DataFrame(), pd.DataFrame()
    
    def extract_subgraph_info_from_csv(self, layer, partition_id):
        try:
            partition_file = os.path.join(self.dataset_path, f'combined_tree_0.3_0.5/combined_layer_{layer}.csv')
            if not os.path.exists(partition_file):
                return [], []

            partition_df = pd.read_csv(partition_file)
            partition_nodes = partition_df[partition_df['partition_id'] == partition_id]['node_id'].tolist()
            
            if not partition_nodes:
                return [], []
                
            nodes_df, edges_df = self.load_graph_data()

            nodes_info = []
            for node_id in partition_nodes:
                node_rows = nodes_df[nodes_df['node_id'] == node_id]
                if not node_rows.empty:
                    node_info = {
                        'node_id': int(node_id),
                        'attr': node_rows.iloc[0]['node_attr']
                    }
                    nodes_info.append(node_info)

            edges_info = []
            filtered_edges = edges_df[
                edges_df['src'].isin(partition_nodes) & 
                edges_df['dst'].isin(partition_nodes)
            ]

            for _, edge_row in filtered_edges.iterrows():
                edge_info = {
                    'src': int(edge_row['src']),
                    'dst': int(edge_row['dst']),
                    'rel': edge_row.get('edge_attr', '')
                }
                edges_info.append(edge_info)

            return nodes_info, edges_info

        except Exception as e:
            import traceback
            traceback.print_exc()
            return [], []
    
    def generate_llm_response(self, prompt, max_tokens=512):
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm.device)
            
            gen_config = {
                "max_new_tokens": max_tokens,
                "temperature": 0.3,
                "top_p": 0.85,
                "do_sample": True,
                "repetition_penalty": 1.2,
                "pad_token_id": self.tokenizer.pad_token_id
            }
            
            with torch.no_grad():
                outputs = self.llm.generate(**inputs, **gen_config)
            
            response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            response = response.strip()
            prefixes = ["I think ", "I believe ", "Based on the information, ", 
                       "The answer is ", "According to the subgraph, "]
            for prefix in prefixes:
                if response.startswith(prefix):
                    response = response[len(prefix):]
                    
            suffixes = [". This is because", ". The reason is", ". Based on"]
            for suffix in suffixes:
                if suffix in response:
                    response = response.split(suffix)[0]
                    
            return response.strip()
            
        except Exception as e:
            return "error"
    
    def generate_subgraph_summaries(self):
        combined_tree_dir = os.path.join(self.dataset_path, 'combined_tree_0.3_0.5')
        if not os.path.exists(combined_tree_dir):
            return {}
            
        summary_file = os.path.join(self.summary_dir, 'subgraph_summaries.json')
        
        if self.use_cache and os.path.exists(summary_file):
            with open(summary_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        summaries = {}
        
        for filename in os.listdir(combined_tree_dir):
            if filename.startswith('combined_layer_') and filename.endswith('.csv'):
                layer_idx = int(filename.replace('combined_layer_', '').replace('.csv', ''))
                layer_file_path = os.path.join(combined_tree_dir, filename)
                
                if not os.path.exists(layer_file_path):
                    continue
                
                partition_df = pd.read_csv(layer_file_path)
                partition_ids = partition_df['partition_id'].unique()
                
                for partition_id in tqdm(partition_ids):
                    nodes_info, edges_info = self.extract_subgraph_info_from_csv(layer_idx, partition_id)
                    
                    if not nodes_info or not edges_info:
                        continue
                    
                    prompt = self.generate_summary_prompt(nodes_info, edges_info)
                    summary = self.generate_llm_response(prompt)
                    
                    key = f"layer_{layer_idx}_partition_{partition_id}"
                    summaries[key] = {
                        "summary": summary,
                        "layer": layer_idx,
                        "partition_id": partition_id,
                        "node_count": len(nodes_info),
                        "edge_count": len(edges_info)
                    }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summaries, f, ensure_ascii=False, indent=2)
        
        return summaries

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summaries, f, ensure_ascii=False, indent=2)

        return summaries
    
    def compute_similarity(self, question, summaries):
        if not summaries:
            return []
        
        question_embedding = self.embed_model.encode([question])[0]
        
        summary_texts = []
        summary_keys = []
        
        for key, summary_info in summaries.items():
            summary_text = summary_info["summary"]
            node_count = summary_info.get("node_count", 0)
            edge_count = summary_info.get("edge_count", 0)
            
            if node_count > 5 and edge_count > 5:
                summary_text = summary_text + " " + summary_text
                
            summary_texts.append(summary_text)
            summary_keys.append(key)
        
        summary_embeddings = self.embed_model.encode(summary_texts)
        
        similarities = cosine_similarity([question_embedding], summary_embeddings)[0]
        
        results = []
        for i, key in enumerate(summary_keys):
            summary_info = summaries[key]
            node_count = summary_info.get("node_count", 0)
            edge_count = summary_info.get("edge_count", 0)
            base_similarity = float(similarities[i])
            size_factor = min(1.0, (node_count + edge_count) / 30.0) * 0.1
            adjusted_similarity = base_similarity * (1 + size_factor)
            
            results.append({
                "key": key,
                "similarity": adjusted_similarity,
                "base_similarity": base_similarity,
                **summaries[key]
            })
        
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        return results[:self.top_k]
    

    
    def verify_answer_with_llm(self, question, generated_answer, standard_answer):
        prompt = self.generate_verification_prompt(question, generated_answer, standard_answer)
        
        verification = self.generate_llm_response(prompt, max_tokens=100).lower().strip()
        
        if "incorrect" in verification:
            return False
        else:
            return True
    
    def answer_question(self, question, answer, full_answer, layer=None, partition_id=None, all_summaries=None):
        try:
            if all_summaries is None:
                all_summaries = self.generate_subgraph_summaries()
                
            relevant_subgraphs = self.compute_similarity(question, all_summaries)
            
            if not relevant_subgraphs:
                return {
                    "generated_answer": "none",
                    "is_correct_llm": False,
                    "retrieved_subgraphs": []
                }
            
            qa_prompt = self.generate_qa_prompt(question, relevant_subgraphs)
            
            generated_answer = self.generate_llm_response(qa_prompt)
            
            is_correct_llm = self.verify_answer_with_llm(question, generated_answer, answer)
            
            return {
                "generated_answer": generated_answer,
                "is_correct_llm": is_correct_llm,
                "retrieved_subgraphs": relevant_subgraphs
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "generated_answer": "处理时出错",
                "is_correct_llm": False,
                "retrieved_subgraphs": []
            }
    
    def process_questions(self):
        all_summaries = self.generate_subgraph_summaries()
        if not all_summaries:
            return
        
        if not os.path.exists(self.questions_file):
            return
        
        try:
            questions_df = pd.read_csv(self.questions_file)
            
            required_columns = ['q_id', 'question', 'answer']
            missing_columns = [col for col in required_columns if col not in questions_df.columns]
            if missing_columns:
                return
            
            total_questions = len(questions_df)
            
            results = []
            correct_count = 0
            
            for idx, row in tqdm(questions_df.iterrows(), total=total_questions):
                try:
                    q_id = row['q_id']
                    question = row['question']
                    answer = row['answer']
                    full_answer = row.get('full_answer', answer)
                    image_id = row.get('image_id', 0)
                    layer = row.get('layer', None)
                    partition_id = row.get('partition_id', None)
                    
                    result_file = os.path.join(self.qa_result_dir, f'question_{q_id}.json')
                    if os.path.exists(result_file) and self.use_cache:
                        with open(result_file, 'r', encoding='utf-8') as f:
                            result = json.load(f)
                    else:
                        result = self.answer_question(
                            question, 
                            answer, 
                            full_answer, 
                            layer, 
                            partition_id, 
                            all_summaries
                        )
                        result.update({
                            "q_id": q_id,
                            "question": question,
                            "answer": answer,
                            "full_answer": full_answer,
                            "image_id": image_id,
                            "layer": layer,
                            "partition_id": partition_id
                        })
                        
                        with open(result_file, 'w', encoding='utf-8') as f:
                            json.dump(result, f, ensure_ascii=False, indent=2)
                    
                    if result.get('is_correct_llm', False):
                        correct_count += 1
                    
                    results.append(result)
                    
                except Exception as e:
                    import traceback
                    traceback.print_exc()
            
            total = len(results)
            
            acc_llm = correct_count / total if total > 0 else 0
            
            summary_result = {
                "acc_llm": acc_llm,
                "total_questions": total,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(os.path.join(self.qa_result_dir, 'summary_result.json'), 'w', encoding='utf-8') as f:
                json.dump(summary_result, f, ensure_ascii=False, indent=2)
            
            return acc_llm, results
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return 0, []

def print_section(title):
    pass

def check_environment():
    import torch
    gpu_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if gpu_available else 0
    
    data_ok = os.path.exists('dataset')
    
    return gpu_available and data_ok

def check_dataset(dataset_path):
    required_files = ['nodes.csv', 'edges.csv', 'question.csv']
    for file_name in required_files:
        file_path = os.path.join(dataset_path, file_name)
        if not os.path.exists(file_path):
            return False
    
    partition_dir = os.path.join(dataset_path, 'partitions')
    if not os.path.exists(partition_dir):
        return False
    
    partition_file = os.path.join(partition_dir, '0.pt')
    if not os.path.exists(partition_file):
        return False
    
    import pandas as pd
    try:
        df = pd.read_csv(os.path.join(dataset_path, 'question.csv'))
    except Exception as e:
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='BookGraphs QA System')
    
    parser.add_argument('--dataset', type=str, default='book_graphs')
    parser.add_argument('--cache_dir', type=str, default='cache')
    parser.add_argument('--no_cache', action='store_true')
    
    parser.add_argument('--llm_model', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--embed_model', type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    
    parser.add_argument('--top_k', type=int, default=3)
    
    parser.add_argument('--question_id', type=int)
    parser.add_argument('--summary_only', action='store_true')
    parser.add_argument('--check_only', action='store_true')
    
    parser.add_argument('--device', type=str)
    parser.add_argument('--float16', action='store_true')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    if not check_environment():
        return
    
    dataset_path = f'dataset/{args.dataset}'
    if not os.path.exists(dataset_path):
        return
    
    if not check_dataset(dataset_path):
        return
    
    if args.check_only:
        return
    
    try:
        qa_system = BookGraphsQA(
            dataset_path=dataset_path,
            embed_model=args.embed_model,
            llm_model=args.llm_model,
            device=args.device,
            cache_dir=args.cache_dir,
            use_cache=not args.no_cache,
            top_k=args.top_k
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return
    
    if args.summary_only:
        summaries = qa_system.generate_subgraph_summaries()
        return
    
    if args.question_id is not None:
        try:
            all_summaries = qa_system.generate_subgraph_summaries()
            
            questions_file = qa_system.questions_file
            questions_df = pd.read_csv(questions_file)
            
            question_row = questions_df[questions_df['q_id'] == args.question_id]
            if question_row.empty:
                return
            
            row = question_row.iloc[0]
            question = row['question']
            answer = row['answer']
            full_answer = row.get('full_answer', answer)
            layer = row.get('layer', None)
            partition_id = row.get('partition_id', None)
            
            # 使用所有子图摘要来回答问题
            result = qa_system.answer_question(
                question, 
                answer, 
                full_answer, 
                layer, 
                partition_id, 
                all_summaries
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
        return
    
    try:
        acc_llm, results = qa_system.process_questions()
        
        end_time = time.time()
        run_time = end_time - start_time
        hours, remainder = divmod(run_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()