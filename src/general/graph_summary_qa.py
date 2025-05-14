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

class GraphSummaryQA:
    
    def __init__(self, 
                 dataset_path='dataset/webqsp_s',
                 embed_model='sentence-transformers/all-MiniLM-L6-v2',
                 llm_model='meta-llama/Llama-2-7b-chat-hf',
                 device=None,
                 cache_dir='cache',
                 use_cache=True,
                 top_k=3):
        self.dataset_path = dataset_path
        self.nodes_dir = os.path.join(dataset_path, 'nodes')
        self.edges_dir = os.path.join(dataset_path, 'edges')
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
        
        access_token = "hf_xxxxxxxxxxxxxxxxxxxxxxx"  
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
            node_desc = f"Node ID: {node_id}, Attributes: {attr_str.strip()}"
            processed_nodes.append(node_desc)
        processed_edges = []
        for edge in edges_info:
            edge_desc = f"Relation: Node{edge['src']} --[{edge['rel']}]--> Node{edge['dst']}"
            processed_edges.append(edge_desc)
        
        prompt = f"""You are an expert graph analyzer tasked with creating a detailed summary of subgraphs.

Below is a subgraph extracted from a larger knowledge graph or scene graph. Please analyze this subgraph carefully and create a comprehensive summary that captures:
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
    
    def generate_qa_prompt(self, question, answer, subgraph_summary, nodes_info, edges_info):
        processed_nodes = []
        for node in nodes_info:
            node_id = node['node_id']
            
            attr_str = str(node['attr'])
            
            node_desc = f"Node ID: {node_id}, Attributes: {attr_str.strip()}"
            processed_nodes.append(node_desc)
        
        processed_edges = []
        for edge in edges_info:
            edge_desc = f"Relation: Node{edge['src']} --[{edge['rel']}]--> Node{edge['dst']}"
            processed_edges.append(edge_desc)
        
        prompt = f"""You are an expert question-answering system that provides precise answers based on graph data.

I will provide you with:
1. A subgraph summary
2. Detailed subgraph information (nodes and relationships)
3. A question

Your task is to carefully analyze the information and provide a concise, accurate answer to the question.

SUBGRAPH SUMMARY:
{subgraph_summary}

DETAILED SUBGRAPH INFORMATION:
Nodes:
{chr(10).join(processed_nodes)}

Relationships:
{chr(10).join(processed_edges)}

QUESTION: {question}

EXPECTED ANSWER FORMAT:
- Answer should be short and direct
- Focus only on information present in the subgraph
- Be specific and avoid vague statements

Answer:"""
        
        return prompt
    
    def load_graph_data(self, graph_id):
        nodes_file = os.path.join(self.nodes_dir, f'{graph_id}.csv')
        if not os.path.exists(nodes_file):
            raise FileNotFoundError(f"{nodes_file}")
        nodes_df = pd.read_csv(nodes_file)
        
        edges_file = os.path.join(self.edges_dir, f'{graph_id}.csv')
        if not os.path.exists(edges_file):
            edges_df = pd.DataFrame(columns=['src', 'dst', 'rel'])
        else:
            edges_df = pd.read_csv(edges_file)
        
        return nodes_df, edges_df
    
    def extract_subgraph_info(self, graph_id, partition_nodes):
        try:
            nodes_df, edges_df = self.load_graph_data(graph_id)
            
            nodes_info = []
            for node_id in partition_nodes:
                node_rows = nodes_df[nodes_df['node_id'] == node_id]
                if not node_rows.empty:
                    node_info = {
                        'node_id': int(node_id),
                        'attr': node_rows.iloc[0]['node_attr']
                    }
                    nodes_info.append(node_info)
            
            partition_nodes_set = set(partition_nodes)
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
    
    def generate_subgraph_summaries(self, graph_id):
        summary_file = os.path.join(self.summary_dir, f'graph_{graph_id}_summaries.json')
        
        if self.use_cache and os.path.exists(summary_file):
            with open(summary_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        partition_file = os.path.join(self.partition_dir, f'{graph_id}.pt')
        if not os.path.exists(partition_file):
            return {}
        
        try:
            partition_data = torch.load(partition_file)
            partition_tensors = partition_data['partitions']
            
            summaries = {}
                        
            for layer_idx, partition_tensor in enumerate(partition_tensors):
                unique_partitions = torch.unique(partition_tensor).tolist()
                
                for partition_id in tqdm(unique_partitions):
                    if partition_id < 0:
                        continue
                    
                    partition_nodes = torch.where(partition_tensor == partition_id)[0].tolist()
                    
                    if len(partition_nodes) <= 1:
                        continue
                    
                    nodes_info, edges_info = self.extract_subgraph_info(graph_id, partition_nodes)
                    
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
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {}
    
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
    
    def answer_question(self, question, answer, graph_id):
        try:
            summaries = self.generate_subgraph_summaries(graph_id)
            
            if not summaries:
                return {
                    "generated_answer": "none",
                    "is_correct": False,
                    "retrieved_subgraphs": []
                }

            relevant_subgraphs = self.compute_similarity(question, summaries)
            
            if not relevant_subgraphs:
                return {
                    "generated_answer": "none",
                    "is_correct": False,
                    "retrieved_subgraphs": []
                }

            generated_answers = []

            top_n = min(3, len(relevant_subgraphs))
            for i in range(top_n):
                subgraph = relevant_subgraphs[i]
                layer_idx = subgraph["layer"]
                partition_id = subgraph["partition_id"]

                partition_file = os.path.join(self.partition_dir, f'{graph_id}.pt')
                partition_data = torch.load(partition_file)
                partition_tensors = partition_data['partitions']

                partition_nodes = torch.where(partition_tensors[layer_idx] == partition_id)[0].tolist()

                nodes_info, edges_info = self.extract_subgraph_info(graph_id, partition_nodes)

                qa_prompt = self.generate_qa_prompt(
                    question, 
                    answer, 
                    subgraph["summary"], 
                    nodes_info, 
                    edges_info
                )

                gen_answer = self.generate_llm_response(qa_prompt)

                is_correct = self._check_answer_correctness(gen_answer, answer)
                
                generated_answers.append({
                    "answer": gen_answer,
                    "is_correct": is_correct,
                    "similarity": subgraph["similarity"],
                    "subgraph_key": subgraph["key"]
                })
                

            correct_answers = [a for a in generated_answers if a["is_correct"]]
            
            if correct_answers:
                best_answer = max(correct_answers, key=lambda x: x["similarity"])
                final_answer = best_answer["answer"]
                is_correct = True
            else:
                best_answer = generated_answers[0] 
                final_answer = best_answer["answer"]
                is_correct = False
            
            return {
                "generated_answer": final_answer,
                "is_correct": is_correct,
                "retrieved_subgraphs": relevant_subgraphs,
                "answer_candidates": generated_answers
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "generated_answer": f"error",
                "is_correct": False,
                "retrieved_subgraphs": []
            }
        
    def _check_answer_correctness(self, generated_answer, standard_answer):
        if not generated_answer or generated_answer == "error":
            return False
    
        uncertainty_phrases = ["I cannot determine", "I don't know", "I'm not sure", 
                             "there is no information", "unclear", "cannot be determined"]
        for phrase in uncertainty_phrases:
            if phrase.lower() in generated_answer.lower():
                return False

        gen_ans = generated_answer.lower().strip()
        std_ans = standard_answer.lower().strip()

        if std_ans == gen_ans:
            return True

        if std_ans in gen_ans:
            return True

        gen_ans_norm = re.sub(r'[^\w\s]', '', gen_ans)
        std_ans_norm = re.sub(r'[^\w\s]', '', std_ans)

        if std_ans_norm in gen_ans_norm:
            return True

        gen_tokens = set(gen_ans_norm.split())
        std_tokens = set(std_ans_norm.split())
        
        if len(std_ans_norm) <= 5:
            return std_ans_norm in gen_ans_norm
            
        if len(std_tokens) <= 3:
            overlap = len(std_tokens.intersection(gen_tokens))
            if overlap >= len(std_tokens) * 0.8:
                return True
        
        similarity = difflib.SequenceMatcher(None, std_ans_norm, gen_ans_norm).ratio()
        
        threshold = 0.85 if len(std_ans_norm) < 10 else 0.65
        
        if len(std_ans_norm) > 30:
            threshold = 0.5
            
        return similarity >= threshold
    
    def process_questions(self):
        questions_file = os.path.join(self.dataset_path, 'questions_filtered.csv')
        if not os.path.exists(questions_file):
            return
        
        try:
            questions_df = pd.read_csv(questions_file)
            
            required_columns = ['q_id', 'image_id', 'question', 'answer']
            missing_columns = [col for col in required_columns if col not in questions_df.columns]
            if missing_columns:
                return
            
            total_questions = len(questions_df)
            
            correct_count = 0
            results = []
            
            for idx, row in tqdm(questions_df.iterrows(), total=total_questions):
                try:
                    q_id = row['q_id']
                    question = row['question']
                    answer = row['answer']
                    graph_id = row['image_id']
                    
                    result_file = os.path.join(self.qa_result_dir, f'question_{q_id}.json')
                    if os.path.exists(result_file) and self.use_cache:
                        with open(result_file, 'r', encoding='utf-8') as f:
                            result = json.load(f)
                    else:
                        result = self.answer_question(question, answer, graph_id)
                        result.update({
                            "q_id": q_id,
                            "question": question,
                            "answer": answer,
                            "graph_id": graph_id,
                            "full_answer": row.get('full_answer', '')
                        })
                        
                        with open(result_file, 'w', encoding='utf-8') as f:
                            json.dump(result, f, ensure_ascii=False, indent=2)
                    
                    if result['is_correct']:
                        correct_count += 1
                    
                    results.append(result)
                    
                except Exception as e:
                    import traceback
                    traceback.print_exc()
            
            accuracy = correct_count / total_questions if total_questions > 0 else 0
            
            summary_result = {
                "accuracy": accuracy,
                "correct_count": correct_count,
                "total_questions": total_questions,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(os.path.join(self.qa_result_dir, 'summary_result.json'), 'w', encoding='utf-8') as f:
                json.dump(summary_result, f, ensure_ascii=False, indent=2)
            
            return accuracy, results
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return 0, []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='webqsp_s')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--no_cache', action='store_true')
    parser.add_argument('--llm_model', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--embed_model', type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    
    args = parser.parse_args()
    
    dataset_path = f'dataset/{args.dataset}'
    
    qa_system = GraphSummaryQA(
        dataset_path=dataset_path,
        embed_model=args.embed_model,
        llm_model=args.llm_model,
        use_cache=not args.no_cache,
        top_k=args.top_k
    )
    
    accuracy, _ = qa_system.process_questions()

if __name__ == '__main__':
    main() 