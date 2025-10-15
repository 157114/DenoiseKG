import os
import json
import time
import yaml
import argparse
import numpy as np
import networkx as nx
from tqdm import tqdm
from openai import OpenAI
from collections import defaultdict
from sklearn.cluster import KMeans

# Config
with open("./Option/Config2.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

embed_cfg = config.get("embedding", {})

api_key = embed_cfg.get("api_key")
base_url = embed_cfg.get("embedding_base_url")
model_name = embed_cfg.get("model")

# Setup client
client = OpenAI(
    api_key=api_key,
    base_url=base_url
)

def embed_batch(texts: list[str], batch_size: int = 50) -> list[list[float]]:
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    print(f"[INFO] Embedding in {total_batches} batches of size {batch_size}...")
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding Progress", unit="batch"):
        batch = texts[i:i + batch_size]
        max_retries = 3
        for retry in range(max_retries):
            try:
                response = client.embeddings.create(
                    model=model_name,
                    input=batch
                )
                embeddings = [e.embedding for e in response.data]
                all_embeddings.extend(embeddings)
                break
            except Exception as e:
                print(f"[ERROR] Failed on batch {i // batch_size + 1} (retry {retry + 1}): {e}")
                if retry < max_retries - 1:
                    time.sleep(2 * (retry + 1))  
                else:
                    print(f"[WARNING] Using zero embeddings for failed batch")
                    all_embeddings.extend([[0.0] * 1536] * len(batch))
    return all_embeddings

def parse_arguments():
    parser = argparse.ArgumentParser(description="Cluster nodes using K-means.")
    parser.add_argument('--dataset', type=str, default='mix', help='Class type for file paths')
    parser.add_argument('--k', type=int, default=10, help='Number of clusters for K-means')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for embedding')
    parser.add_argument('--graphml_path', type=str, default='./Result/mix/rkg_graph/graph_storage/graph_storage_nx_data.graphml', help='Path to the GraphML file')
    parser.add_argument('--k_means_path', type=str, default='./Result/mix/rkg_graph/k_means', help='K-means path')
    parser.add_argument('--orig_text_path', type=str, default='./Result/mix/rkg_graph/k_means/mix_original_texts.json', help='Original texts path')
    parser.add_argument('--orig_emb_path', type=str, default='./Result/mix/rkg_graph/k_means/mix_original_embeddings.npy', help='Original embeddings path')
    parser.add_argument('--output_path', type=str, default='./Result/mix/rkg_graph/k_means/mix_kmeans_similar_nodes.json', help='Output path')
    parser.add_argument('--cluster_map_path', type=str, default='./Result/mix/rkg_graph/k_means/mix_kmeans_cluster_map.json', help='Cluster map path')
    parser.add_argument('--cluster_type', type=str, default='entity_block', choices=['kmeans', 'entity_block', 'structure'], help='Type of clustering to use')
    return parser.parse_args()


def main():
    args = parse_arguments()
    print(args)
    dataset = args.dataset
    K = args.k
    batch_size = args.batch_size

    os.makedirs(args.k_means_path, exist_ok=True)
    GRAPHML_PATH = args.graphml_path
    ORIG_TEXT_PATH = args.orig_text_path
    ORIG_EMB_PATH = args.orig_emb_path
    OUTPUT_PATH = args.output_path
    CLUSTER_MAP_PATH = args.cluster_map_path

    # Load Graph
    if not os.path.exists(GRAPHML_PATH):
        raise FileNotFoundError(f"GraphML file not found: {GRAPHML_PATH}")
    G = nx.read_graphml(GRAPHML_PATH)

    # Filter nodes with 'description' or 'entity_name'
    nodes_with_desc = [
        n for n, d in G.nodes(data=True) 
        if d.get('description') or d.get('entity_name')
    ]
    if not nodes_with_desc:
        raise ValueError("No nodes with 'description' or 'entity_name' attribute found in the graph.")
    G_desc = G.subgraph(nodes_with_desc).copy()
    node_ids = list(G_desc.nodes())

    # Load or create original texts for embedding
    if os.path.exists(ORIG_TEXT_PATH):
        with open(ORIG_TEXT_PATH, "r", encoding="utf-8") as f:
            texts_to_embed = json.load(f)
        print(f"[INFO] Loaded cached descriptions from {ORIG_TEXT_PATH}")
    else:
        # Use description, or fall back to entity_name
        texts_to_embed = [
            G_desc.nodes[n].get('description') or G_desc.nodes[n].get('entity_name') 
            for n in node_ids
        ]
        with open(ORIG_TEXT_PATH, "w", encoding="utf-8") as f:
            json.dump(texts_to_embed, f, ensure_ascii=False)
        print(f"[INFO] Saved descriptions to {ORIG_TEXT_PATH}")

    # Load or create embeddings
    if os.path.exists(ORIG_EMB_PATH):
        embeddings = np.load(ORIG_EMB_PATH)
        print(f"[INFO] Loaded cached embeddings from {ORIG_EMB_PATH}")
    else:
        print(f"[INFO] Embedding {len(texts_to_embed)} node texts...")
        embeddings = embed_batch(texts_to_embed, batch_size=batch_size)
        embeddings = np.array(embeddings).astype('float32')
        np.save(ORIG_EMB_PATH, embeddings)
        print(f"[INFO] Saved embeddings to {ORIG_EMB_PATH}")

    # ---- Clustering ----
    if args.cluster_type == "kmeans":
        print(f"[INFO] Running global K-means clustering with K={K}...")
        kmeans = KMeans(n_clusters=K, random_state=42)
        labels = kmeans.fit_predict(embeddings)

        cluster_to_nodes = defaultdict(list)
        for idx, label in enumerate(labels):
            cluster_to_nodes[int(label)].append(node_ids[idx])

    elif args.cluster_type == "entity_block":
        print("[INFO] Running entity_block clustering...")
        cluster_to_nodes = defaultdict(list)
        labels = [-1] * len(node_ids)  # placeholder for global label assignment
        current_label = 0

        # Group nodes by entity_type
        entity_blocks = defaultdict(list)
        for idx, nid in enumerate(node_ids):
            entity_type = G_desc.nodes[nid].get("entity_type", "unknown")
            entity_blocks[entity_type].append(idx)

        for etype, indices in entity_blocks.items():
            block_size = len(indices)
            if block_size <= 512:
                for idx in indices:
                    labels[idx] = current_label
                cluster_to_nodes[current_label] = [node_ids[idx] for idx in indices]
                current_label += 1
            else:
                # run k-means inside this block
                k = max(2, int(np.sqrt(block_size / 10)))
                print(f" -> {etype}: {block_size} nodes, running KMeans with k={k}")
                block_embeddings = embeddings[indices]
                kmeans = KMeans(n_clusters=k, random_state=42)
                block_labels = kmeans.fit_predict(block_embeddings)
                for i, idx in enumerate(indices):
                    new_label = current_label + block_labels[i]
                    labels[idx] = new_label
                    cluster_to_nodes[new_label].append(node_ids[idx])
                current_label += k
    elif args.cluster_type == "structure":
        print("[INFO] Running structure-based clustering (connected components)...")
        cluster_to_nodes = defaultdict(list)
        labels = [-1] * len(node_ids)

        # Find connected components in the graph
        components = list(nx.connected_components(G_desc))

        for cluster_id, comp in enumerate(components):
            for nid in comp:
                idx = node_ids.index(nid)  #
                labels[idx] = cluster_id
                cluster_to_nodes[cluster_id].append(nid)
    else:
        raise ValueError(f"Invalid cluster_type: {args.cluster_type}")

    # Output: for each node, list all other nodes in the same cluster
    results = []
    for node_id, source_text, label in zip(node_ids, texts_to_embed, labels):
        similar_node_ids = [nid for nid in cluster_to_nodes[int(label)] if nid != node_id]
        results.append({
            "node_id": node_id,
            "source_text": source_text,
            "similar_nodes": similar_node_ids
        })

    # Save similar nodes list
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Cluster-based similar nodes saved to: {OUTPUT_PATH}")

    # Save cluster map
    if args.cluster_type == "entity_block":
        cluster_map = {}
        for label, nodes in cluster_to_nodes.items():
            entity_type = G_desc.nodes[nodes[0]].get("entity_type", "unknown") if nodes else "unknown"
            cluster_map[int(label)] = {
                "entity_type": entity_type,
                "nodes": nodes
            }
    else:
        # for cluster_type: kmeans and structure
        cluster_map = {int(label): nodes for label, nodes in cluster_to_nodes.items()}

    with open(CLUSTER_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(cluster_map, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Cluster map saved to: {CLUSTER_MAP_PATH}")


if __name__ == "__main__":
    main()