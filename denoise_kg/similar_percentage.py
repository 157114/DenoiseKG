import os
import json, json_repair
import time
import numpy as np
import networkx as nx
import yaml
from tqdm import tqdm
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
import argparse


# === Setup OpenAI-compatible client ===
print("[INFO] Setting up API client...")
try:
    with open("./Option/Config2.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    embed_cfg = config.get("embedding", {})
    api_key = embed_cfg.get("api_key")
    base_url = embed_cfg.get("embedding_base_url")
    model_name = embed_cfg.get("model")
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    print("✅ API client configured successfully.")
except Exception as e:
    print(f"❌ ERROR: Could not configure API client from Config2.yaml. Details: {e}")
    exit(1)


def create_rich_representation(node_id: str, node_data: dict) -> str:
    desc = node_data.get("description", "")
    entity_type = node_data.get("entity_type", "N/A")
    return f"Entity: {node_id}. Type: {entity_type}. Description: {desc}"

def embed_batch(texts: list[str], batch_size: int, embedding_type: str) -> np.ndarray:
    all_embeddings = []

    if embedding_type == "llm":
        embedding_model = model_name or "text-embedding-v3"
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding Progress", unit="batch"):
            batch = [str(t) for t in texts[i:i + batch_size] if t]
            if not batch:
                all_embeddings.extend([[0.0] * 1536] * len(texts[i:i + batch_size]))
                continue 
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = client.embeddings.create(model=embedding_model, input=batch)
                    embeddings = [e.embedding for e in response.data]
                    all_embeddings.extend(embeddings)
                    break
                except Exception as e:
                    print(f"[ERROR] Batch failed (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2 * (attempt + 1))
                    else:
                        print(f"[WARNING] Using zero embeddings for failed batch.")
                        embedding_dim = len(all_embeddings[0]) if all_embeddings else 1536
                        all_embeddings.extend([[0.0] * embedding_dim] * len(batch))
    else:
        raise ValueError(f"Embedding type {embedding_type} not supported.")
    return np.array(all_embeddings).astype('float32')

def get_average_neighbor_embedding(
    node_id: str,
    graph: nx.Graph,
    node_id_to_embedding: Dict[str, np.ndarray],
    node_data_map: Dict[str, Dict],
    max_neighbors: int,
    similarity_mode: str
) -> np.ndarray:
    embedding_dim = list(node_id_to_embedding.values())[0].shape[0]
    if node_id not in graph:
        return np.zeros(embedding_dim)

    neighbor_ids = list(graph.neighbors(node_id))[:max_neighbors]
    
    if not neighbor_ids:
        return np.zeros(embedding_dim)

    all_neighbor_embeddings = [node_id_to_embedding[nid] for nid in neighbor_ids if nid in node_id_to_embedding]
    if not all_neighbor_embeddings:
        all_neighbors_avg = np.zeros(embedding_dim)
    else:
        all_neighbors_avg = np.mean(np.array(all_neighbor_embeddings), axis=0)

    if similarity_mode == "neisubset_only" or similarity_mode == "node_neisubset":
        main_node_data = node_data_map.get(node_id, {})
        main_node_type = main_node_data.get("entity_type")
        
        subset_neighbor_embeddings = []
        if main_node_type:
            for neighbor_id in neighbor_ids:
                neighbor_data = node_data_map.get(neighbor_id)
                if (neighbor_data and 
                    neighbor_data.get("entity_type") == main_node_type and 
                    neighbor_id in node_id_to_embedding):
                    subset_neighbor_embeddings.append(node_id_to_embedding[neighbor_id])

        if not subset_neighbor_embeddings:
            subset_avg = np.zeros(embedding_dim)
        else:
            subset_avg = np.mean(np.array(subset_neighbor_embeddings), axis=0)
            
        return np.mean(np.array([all_neighbors_avg, subset_avg]), axis=0)
    else:
        return all_neighbors_avg

def parse_arguments():
    parser = argparse.ArgumentParser(description="Calculate node similarity and merge nodes.")
    parser.add_argument('--dataset_name', type=str, default='cs', help='Class type for file paths')
    parser.add_argument("--similarity_mode", type=str, choices=["node_only", "neighbor_only", "node_neighbor", "neisubset_only", "node_neisubset"], default="node_only", help="Type of node embdeeings")
    parser.add_argument('--target_reduction_percent', type=float, default=0.20, help='Target node reduction percentage')
    parser.add_argument('--max_neighbors', type=int, default=10, help='Maximum number of neighbors')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for embedding')
    parser.add_argument('--node_weight', type=float, default=0.5, help='Weight for node embedding')
    parser.add_argument('--neighbor_weight', type=float, default=0.5, help='Weight for neighbor embedding')
    parser.add_argument('--embedding_type', type=str, default='llm', choices=["llm", "TransE", "DistMult", "ComplEx", "RGCN", "CompGCN"], help='Embedding type')

    parser.add_argument('--graphml_path', type=str, default='./Result/mix/rkg_graph/graph_storage/graph_storage_nx_data.graphml', help='GraphML path')
    parser.add_argument('--node_data_path', type=str, default='./Result/mix/rkg_graph/k_means/mix_kmeans_similar_nodes.json', help='Node data path')
    parser.add_argument('--cluster_map_path', type=str, default='./Result/mix/rkg_graph/k_means/mix_kmeans10_cluster_map.json', help='Cluster map path')
    parser.add_argument('--similar_nodes_path', type=str, default='./Result/mix/rkg_graph/node_neighbors/cs_20.0pct_pairs_subset_True_nodes_only.json', help='Similar node path')
    parser.add_argument('--cache_dir', type=str, default='./Result/mix/rkg_graph/k_means', help='Cache directory')
    return parser.parse_args()


def main():
    args = parse_arguments()
    dataset_name = args.dataset_name
    SIMILARITY_MODE = args.similarity_mode
    TARGET_REDUCTION_PERCENT = args.target_reduction_percent
    MAX_NEIGHBORS = args.max_neighbors
    BATCH_SIZE = args.batch_size
    NODE_WEIGHT = args.node_weight
    NEIGHBOR_WEIGHT = args.neighbor_weight
    EMBEDDING_TYPE = args.embedding_type

    GRAPHML_PATH = args.graphml_path
    NODE_DATA_PATH = args.node_data_path
    CLUSTER_MAP_PATH = args.cluster_map_path
    CACHE_DIR = args.cache_dir

    output_path = args.similar_nodes_path
        
    print(f"--- Running in mode: {SIMILARITY_MODE} ---")
    print(f"--- Target Node Reduction: {TARGET_REDUCTION_PERCENT*100:.1f}% ---")

    # --- 1. Load Data ---
    print("\n[INFO] Loading data...")
    G = nx.read_graphml(GRAPHML_PATH)
    total_node_count = G.number_of_nodes()
    target_nodes_to_merge = int(total_node_count * TARGET_REDUCTION_PERCENT)
    print(f" -> Graph has {total_node_count} nodes. Target to merge: {target_nodes_to_merge} nodes.")

    with open(CLUSTER_MAP_PATH, "r", encoding="utf-8") as f:
        cluster_map = json.load(f)
    with open(NODE_DATA_PATH, "r", encoding="utf-8") as f:
        node_data_list = json.load(f)

    node_ids = [entry["node_id"] for entry in node_data_list]
    node_id_to_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    node_data_map = {entry["node_id"]: entry for entry in node_data_list}

    # --- 2. Generate and Cache Base Embeddings ---
    print("\n[INFO] Generating node embeddings...")

    METHOD = EMBEDDING_TYPE 

    if METHOD.lower() == "llm":
        base_embedding_cache_path = os.path.join(CACHE_DIR, f"{dataset_name}_rich_base_embeddings.npy")
        if os.path.exists(base_embedding_cache_path):
            print(f"[INFO] Loading cached LLM embeddings...")
            base_embeddings = np.load(base_embedding_cache_path)
        else:
            print("[INFO] Generating base text representations...")
            texts_to_embed = [
                create_rich_representation(node_id, node_data_map.get(node_id, {}))
                for node_id in tqdm(node_ids, desc="Generating Text")
            ]
            print(f"[INFO] Generating {len(texts_to_embed)} LLM embeddings via API...")
            base_embeddings = embed_batch(texts_to_embed, BATCH_SIZE, EMBEDDING_TYPE)
            os.makedirs(os.path.dirname(base_embedding_cache_path), exist_ok=True)
            np.save(base_embedding_cache_path, base_embeddings)
            print(f"[INFO] Saved LLM embeddings to cache.")

        node_id_to_embedding = {node_id: base_embeddings[i] for i, node_id in enumerate(node_ids)}

    elif METHOD.lower() in ["transe", "complex", "distmult", "rgcn", "compgcn"]:
        # Use KG embeddings
        from kg_embedding import compute_kg_embeddings

        print(f"[INFO] Generating {METHOD} embeddings from KG...")
        base_embeddings, node_id_to_index = compute_kg_embeddings(
            G,
            method=METHOD,
            embedding_dim=100,
            epochs=100,
            device="auto"
        )

        # Map node_id to embedding
        nodes_in_order = list(G.nodes())
        node_id_to_embedding = {node: base_embeddings[i] for i, node in enumerate(nodes_in_order)}

    else:
        raise ValueError(f"Invalid embedding method: {METHOD}")


    # --- 3. Create Final Embeddings ---
    print(f"\n[INFO] Applying '{SIMILARITY_MODE}' pooling strategy...")

    final_embeddings_list = []
    for i, node_id in enumerate(tqdm(node_ids, desc="Applying Pooling")):
        node_embedding = base_embeddings[i]
        
        if SIMILARITY_MODE == "node_only":
            final_embeddings_list.append(node_embedding)
            
        elif SIMILARITY_MODE == "neighbor_only":
            neighbor_avg_embedding = get_average_neighbor_embedding(
                node_id, G, node_id_to_embedding, node_data_map, MAX_NEIGHBORS, SIMILARITY_MODE
            )
            final_embeddings_list.append(neighbor_avg_embedding)

        elif SIMILARITY_MODE == "node_neighbor":
            neighbor_avg_embedding = get_average_neighbor_embedding(
                node_id, G, node_id_to_embedding, node_data_map, MAX_NEIGHBORS, SIMILARITY_MODE
            )
            combined_embedding = (NODE_WEIGHT * node_embedding) + (NEIGHBOR_WEIGHT * neighbor_avg_embedding)
            final_embeddings_list.append(combined_embedding)

        elif SIMILARITY_MODE == "neisubset_only":
            subset_avg_embedding = get_average_neighbor_embedding(
                node_id, G, node_id_to_embedding, node_data_map, MAX_NEIGHBORS, SIMILARITY_MODE
            )
            final_embeddings_list.append(subset_avg_embedding)

        elif SIMILARITY_MODE == "node_neisubset":
            subset_avg_embedding = get_average_neighbor_embedding(
                node_id, G, node_id_to_embedding, node_data_map, MAX_NEIGHBORS, SIMILARITY_MODE
            )
            combined_embedding = (NODE_WEIGHT * node_embedding) + (NEIGHBOR_WEIGHT * subset_avg_embedding)
            final_embeddings_list.append(combined_embedding)

        else:
            raise ValueError(f"Invalid similarity_mode: {SIMILARITY_MODE}")
    
    final_embeddings = np.array(final_embeddings_list)


    # --- 4. MODIFIED: Calculate All Similarities and Select Pairs to Meet Target ---
    all_potential_pairs = []
    print(f"\n[INFO] Calculating all within-cluster similarity scores...")
    for cluster_label, members in tqdm(cluster_map.items(), desc="Processing clusters"):
        # Support formats of all cluster_type
        if isinstance(members, dict):
            node_list = members.get("nodes", [])
        else:
            node_list = members

        indexed_nodes = [
            (nid, node_id_to_index[nid])
            for nid in node_list
            if nid in node_id_to_index
        ]
        max_idx = final_embeddings.shape[0]
        indexed_nodes = [(nid, idx) for nid, idx in indexed_nodes if idx < max_idx]

        if len(indexed_nodes) < 2:
            continue

        indices = [idx for _, idx in indexed_nodes]
        ids_in_order = [nid for nid, _ in indexed_nodes]

        emb_subset = final_embeddings[indices]
        sim_matrix = cosine_similarity(emb_subset)

        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                all_potential_pairs.append({
                    "node1": ids_in_order[i],
                    "node2": ids_in_order[j],
                    "similarity": sim_matrix[i, j],
                    "cluster": int(cluster_label)
                })

    # Sort all pairs from all clusters by similarity, highest first
    all_potential_pairs.sort(key=lambda x: x["similarity"], reverse=True)

    # --- Simulate merges to find the pairs that meet the reduction target ---
    print(f"\n[INFO] Selecting top pairs to achieve {TARGET_REDUCTION_PERCENT*100:.1f}% node reduction...")
    
    parent = {}
    def find_set(v):
        if v not in parent: parent[v] = v
        if v == parent[v]: return v
        parent[v] = find_set(parent[v])
        return parent[v]

    final_pairs = []
    merged_nodes_count = 0
    
    for pair in all_potential_pairs:
        if merged_nodes_count >= target_nodes_to_merge:
            break

        root1 = find_set(pair["node1"])
        root2 = find_set(pair["node2"])

        if root1 != root2:
            if root1 < root2:
                parent[root2] = root1
            else:
                parent[root1] = root2
            
            final_pairs.append(pair)
            merged_nodes_count += 1


    # --- 5. Save Results ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        # Round similarity for cleaner output
        for pair in final_pairs:
            pair["similarity"] = round(float(pair["similarity"]), 4)
        json.dump(final_pairs, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] Selected {len(final_pairs)} pairs to merge, resulting in {merged_nodes_count} merged nodes.")
    if total_node_count > 0:
        actual_reduction = (merged_nodes_count / total_node_count) * 100
        print(f" -> Actual Node Reduction: {actual_reduction:.2f}%")
        
    print(f"[INFO] Saved final pairs to {output_path}")

if __name__ == "__main__":
    main()