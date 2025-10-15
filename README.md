# Denoising Knowledge Graphs for Retrieval Augmented Generation (DEG-RAG)

### Data config

1. Download dataset from https://drive.google.com/file/d/14nYYw-3FutumQnSRwKavIbG3LRSmIzDX/view
2. extract to `Data`

### Environment

```
conda env create -f experiment.yml -n graphrag
pip install json_repair pykeen
```

### Running the Script

The `scripts/run_all.sh` script is designed to automate the process of initializing and processing a knowledge graph (KG) using various methods and configurations. Below are the details on how to run the script and customize its parameters.

#### Default Parameters
- `dataset`: The dataset to use (default: `mini`).
- `method`: The method configuration file (default: `LightRAG`).
- `node_reduction`: The target reduction percentage for nodes (default: `0.40`).
- `edge_threshold`: The threshold for edge reasoning (default: `0.20`).
- `llm_base_url`: Base URL for the language model (default: `http://28.7.192.183:8081/v1/`).
- `embedding_base_url`: Base URL for embeddings (default: `http://28.7.195.165:8081/v1`).
- `cluster_type`: Type of clustering to use (default: `kmeans`).
- `embedding_type`: Type of embedding to use (default: `llm`).
- `similarity_mode`: Mode of similarity calculation (default: `node_only`).
- `merge_type`: Type of merging to perform (default: `reduction_only`).

#### Steps in the Script
1. **Initialize the KG**: Checks if the KG is already initialized and skips if so.
2. **Cluster Nodes**: Clusters nodes using the specified clustering method.
3. **Calculate Similar Nodes**: Computes similar nodes based on the specified similarity mode.
4. **Edge Cleaning**: Cleans edges in the graph.
5. **Merge Nodes**: Merges nodes based on similarity and edge thresholds.
6. **RAG Response**: Generates a response using the RAG method.
7. **Winrate Calculation**: Compares results to calculate winrate.

#### Example Command
To run the script with default parameters:
```bash
bash scripts/run_all.sh
```
To customize parameters, you can set them before running the script:
```bash
export dataset="your_dataset"
export method="your_method"
bash scripts/run_all.sh
```

#### Output
The results and outputs are stored in the `Result` directory, organized by dataset and experiment name. Key output files include:
- `judge_result.json`: Final evaluation results.
- `similar_nodes.json`: Similar nodes identified.
- `edge_reason.jsonl`: Edge cleaning results.
- `graph_storage_nx_data.graphml`: Merged graph file.

Ensure all dependencies are installed and configured as per the script requirements before running the script.