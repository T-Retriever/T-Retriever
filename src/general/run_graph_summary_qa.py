import argparse
import os
import sys
import time
from graph_summary_qa import GraphSummaryQA

def print_section(title):
    pass

def check_environment():
    import torch
    gpu_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if gpu_available else 0
    
    data_ok = os.path.exists('dataset')
    
    return gpu_available and data_ok

def check_dataset(dataset_path):
    required_dirs = ['nodes', 'edges', 'partitions']
    for dir_name in required_dirs:
        dir_path = os.path.join(dataset_path, dir_name)
        if not os.path.exists(dir_path):
            return False
    
    questions_file = os.path.join(dataset_path, 'questions.csv')
    if not os.path.exists(questions_file):
        return False
    else:
        import pandas as pd
        try:
            df = pd.read_csv(questions_file)
        except Exception as e:
            return False
    
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='webqsp_s')
    parser.add_argument('--cache_dir', type=str, default='cache')
    parser.add_argument('--no_cache', action='store_true')
    parser.add_argument('--llm_model', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--embed_model', type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--graph_id', type=int)
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
        qa_system = GraphSummaryQA(
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
    if args.graph_id is not None and args.summary_only:
        try:
            summaries = qa_system.generate_subgraph_summaries(args.graph_id)
        except Exception as e:
            import traceback
            traceback.print_exc()
        return
    if args.question_id is not None:
        try:
            questions_file = os.path.join(dataset_path, 'questions_filtered.csv')
            import pandas as pd
            questions_df = pd.read_csv(questions_file)
            question_row = questions_df[questions_df['q_id'] == args.question_id]
            if question_row.empty:
                return
            row = question_row.iloc[0]
            question = row['question']
            answer = row['answer']
            graph_id = row['image_id']

            result = qa_system.answer_question(question, answer, graph_id)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
        return
    try:
        accuracy, results = qa_system.process_questions()
        end_time = time.time()
        run_time = end_time - start_time
        hours, remainder = divmod(run_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 