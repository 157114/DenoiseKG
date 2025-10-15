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

#### Steps in the Script
1. **Initialize the KG**: Checks if the KG is already initialized and skips if so.
2. **Cluster Nodes**: Clusters nodes using the specified clustering method.
3. **Calculate Similar Nodes**: Computes similar nodes based on the specified similarity mode.
4. **Edge Cleaning**: Cleans edges in the graph.
5. **Merge Nodes**: Merges nodes based on similarity and edge thresholds.
6. **RAG Response**: Generates a response using the RAG method.
7. **Winrate Calculation**: Compares results to calculate winrate.

#### Example Command
Before running the script, configure your API keys and urls for llm and embedding models in ./Option/Config2.yaml
```yaml
llm:
    llm_base_url: 'YOUR_LLM_BASE_URL'
    model: "YOUR_LLM_MODEL"
    api_key: "YOUR_API_KEY"
embedding:
    embedding_base_url: 'YOUR_EMBEDDING_BASE_URL'
    model: "YOUR_EMBEDDING_MODEL"
    api_key: "YOUR_API_KEY"
```

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

### Acknowledgement
This project is forked from DIGIMON https://github.com/JayLZhou/GraphRAG.