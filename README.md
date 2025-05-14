# T-Retriever

## Environment requirement

```bash
conda create --name T_Retriever python=3.9 -y
conda activate T_Retriever

# https://pytorch.org/get-started/locally/
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia

python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.version.cuda)"
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

pip install transformers
pip install peft
pip install sentencepiece
pip install sentence-transformers
pip install pandas
pip install numpy
pip install scipy==1.12
pip install gensim
pip install tqdm
pip install ogb
pip install pcst_fast
pip install python-igraph leidenalg
pip install wandb
pip install datasets
pip install protobuf

```

## Download the Llama 2 Model
1. Go to Hugging Face: https://huggingface.co/meta-llama/Llama-2-7b-hf. You will need to share your contact information with Meta to access this model.
2. Sign up for a Hugging Face account (if you don’t already have one).
3. Generate an access token: https://huggingface.co/docs/hub/en/security-tokens.
4. Add your token to the code file
`conda activate hierarchical_rag`

## Data Preprocessing

- `dataset/scene_graphs/`:
  - `nodes/`
  - `edges/`
  - `questions.csv`

- `dataset/webqsp/`:https://huggingface.co/datasets/rmanluo/RoG-webqsp
  - `nodes/`
  - `edges/`
  - `questions.csv`

- `dataset/book_graphs/`:
  - `Children.csv`
  - `edges.csv`
  - `nodes.csv`
  - `questions.csv`
  
```
# scene_graphs
python -m src.dataset.scene_graphs

# webqsp
python -m src.dataset.webqsp

# book_graphs
python -m src.dataset.book_graphs 
```

## Run Code

### WebQSP & Scene Graphs

```bash
python src/general/data_preprocessing.py
python src/general/hierarchical_partition.py
python src/general/run_graph_summary_qa.py
```

### Book Graphs

```bash
python src/book_graphs/data_preprocessing.py
python src/book_graphs/hierarchical_partition.py
python src/book_graphs/book_graphs_qa.py
```

