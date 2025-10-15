import os
import json
import networkx as nx
import argparse
import asyncio
import yaml
import random
import tiktoken
from typing import List, Dict, Optional
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from concurrent.futures import ProcessPoolExecutor
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Core.Utils.MergeSum import MergeEntity, MergeRelationship, DescriptionSummarizer

GRAPH_FIELD_SEP = "<SEP>"
CONFIG_PATH = "./Option/Config2.yaml"


def summarize_node_sync(args):
    """Synchronous wrapper function for multiprocessing node summarization."""
    node, description, api_key, base_url, model_name = args
    summarizer = DescriptionSummarizer(api_key, base_url, model_name)
    summary = asyncio.run(summarizer.summarize(node, description, text_type='entity_description'))
    return node, summary

def build_canonical_map(pairs: List[Dict]) -> Dict[str, str]:

    parent = {}
    def find_set(v): 
        if v not in parent: parent[v] = v
        if v == parent[v]: return v
        parent[v] = find_set(parent[v])
        return parent[v]
    def unite_sets(a, b): 
        a_root, b_root = find_set(a), find_set(b)
        if a_root != b_root:
            if a_root < b_root: parent[b_root] = a_root
            else: parent[a_root] = b_root
    for pair in pairs: unite_sets(pair['node1'], pair['node2'])
    return {node: find_set(node) for node in parent}

# --- MAIN FUNCTION WITH UPDATED SUMMARIZATION PASS ---
def merge_similar_nodes(dataset_name: str, graph_path: str, similar_nodes_path: str, output_path: str, threshold_edge_reason: float, similarity: float, merge_type: str, process_num: Optional[int] = None):
    print("--- Starting Node Merging Process ---")
    # Step 0: Load config and initialize summarizer
    summarizer = None
    print(f"\n[Step 0] Loading configuration from {CONFIG_PATH}...")
    with open(CONFIG_PATH, 'r') as f: config = yaml.safe_load(f)
    api_key = config.get('llm', {}).get('api_key')
    base_url = config.get('llm', {}).get('llm_base_url')
    model_name = config.get('llm', {}).get('model')
    summarizer = DescriptionSummarizer(api_key, base_url, model_name)
    print(" -> Summarizer initialized successfully.")

    # Step 1 & 2: Load data and build map
    print(f"\n[Step 1] Loading data...")
    G = nx.read_graphml(graph_path)
    with open(similar_nodes_path, 'r', encoding='utf-8') as f: similar_node_pairs = json.load(f)
    print(f"✅ Data loaded. Initial graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    print("\n[Step 2] Building canonical map for all similar nodes...")
    canonical_map = build_canonical_map(similar_node_pairs)
    print(f" -> Map built. {len({n for n, t in canonical_map.items() if n != t})} nodes will be merged.")

    # [Step 3 - PASS 1] Merge nodes and edges WITHOUT summarization
    print("\n[Step 3] Pass 1: Merging all nodes/edges and concatenating attributes...")

    merged_count = 0
    skipped_synonym_merges = 0
    if merge_type == "reduction_only":
        for node_to_merge, target_node in tqdm(canonical_map.items()):
            if node_to_merge == target_node: continue
            if node_to_merge in G and target_node in G:
                source_data, target_data = G.nodes[node_to_merge], G.nodes[target_node]
                merged_data = MergeEntity.merge_info(target_data, source_data, summarizer=None)
                nx.set_node_attributes(G, {target_node: merged_data})
                for neighbor in list(G.neighbors(node_to_merge)):
                    edge_data = G.get_edge_data(node_to_merge, neighbor)
                    if edge_data['tgt_id'] == node_to_merge: edge_data['tgt_id'] = target_node
                    if edge_data['src_id'] == node_to_merge: edge_data['src_id'] = target_node
                    if G.has_edge(target_node, neighbor):
                        existing_edge_data = G.get_edge_data(target_node, neighbor)
                        merged_edge_data = MergeRelationship.merge_info(existing_edge_data, edge_data)
                        G.add_edge(target_node, neighbor, **merged_edge_data)
                    else:
                        G.add_edge(target_node, neighbor, **edge_data)
                G.remove_node(node_to_merge)
                merged_count += 1

    elif merge_type == "reduction_synonym":
        print("Adding synonym edges only")
        with open(similar_nodes_path, "r", encoding="utf-8") as f:
            synonym_pairs = json.load(f)

        for pair in synonym_pairs:
            n1 = pair["node1"]
            n2 = pair["node2"]
            cluster = pair.get("cluster", -1)
            if n1 not in G.nodes or n2 not in G.nodes:
                print(f"Skipping: {n1} or {n2} not in graph")
                continue
            chunk_id_1 = G.nodes[n1].get("source_id", "unknown")
            chunk_id_2 = G.nodes[n2].get("source_id", "unknown")
            ids1 = set(chunk_id_1.split('<SEP>'))
            ids2 = set(chunk_id_2.split('<SEP>'))
            all_ids = sorted([id_str for id_str in ids1.union(ids2) if id_str])
            combined_chunk_ids = "<SEP>".join(all_ids)
            edge_data = {"src_id": n1, "tgt_id": n2, "relation_name": "synonym_of", "keywords": f"synonym cluster={cluster}",
                         "description": f"{n1} is the synonym of {n2}", "source_id": combined_chunk_ids, "weight": 1.0,}
            G.add_edge(n1, n2, **edge_data)
        print(f"✅ Synonym edges added: {len(synonym_pairs)}")

        print("\n Identifying 'synonym_of' relationships to protect from merging...")
        synonym_pairs = set()
        for u, v, data in G.edges(data=True):
            if data.get("relation_name") == "synonym_of":
                synonym_pairs.add(frozenset({u, v}))
        if synonym_pairs:
            print(f" -> Found {len(synonym_pairs)} 'synonym_of' pairs. These node pairs will be protected.")
        else:
            print(" -> No 'synonym_of' relationships found.")


        for node_to_merge, target_node in tqdm(canonical_map.items()):
            if node_to_merge == target_node: continue
            # Skip synonym_of edges
            if frozenset({node_to_merge, target_node}) in synonym_pairs:
                skipped_synonym_merges += 1
                continue

            if node_to_merge in G and target_node in G:
                source_data, target_data = G.nodes[node_to_merge], G.nodes[target_node]
                merged_data = MergeEntity.merge_info(target_data, source_data, summarizer=None)
                nx.set_node_attributes(G, {target_node: merged_data})
                for neighbor in list(G.neighbors(node_to_merge)):
                    edge_data = G.get_edge_data(node_to_merge, neighbor)
                    if edge_data['tgt_id'] == node_to_merge: edge_data['tgt_id'] = target_node
                    if edge_data['src_id'] == node_to_merge: edge_data['src_id'] = target_node
                    if G.has_edge(target_node, neighbor):
                        existing_edge_data = G.get_edge_data(target_node, neighbor)
                        merged_edge_data = MergeRelationship.merge_info(existing_edge_data, edge_data)
                        G.add_edge(target_node, neighbor, **merged_edge_data)
                    else:
                        G.add_edge(target_node, neighbor, **edge_data)
                G.remove_node(node_to_merge)
                merged_count += 1

    elif merge_type == "synonym_only":
        print("Adding synonym edges only")
        with open(similar_nodes_path, "r", encoding="utf-8") as f:
            synonym_pairs = json.load(f)
        for pair in synonym_pairs:
            n1 = pair["node1"]
            n2 = pair["node2"]
            cluster = pair.get("cluster", -1)
            if n1 not in G.nodes or n2 not in G.nodes:
                print(f"Skipping: {n1} or {n2} not in graph")
                continue
            chunk_id_1 = G.nodes[n1].get("source_id", "unknown")
            chunk_id_2 = G.nodes[n2].get("source_id", "unknown")
            ids1 = set(chunk_id_1.split('<SEP>'))
            ids2 = set(chunk_id_2.split('<SEP>'))
            all_ids = sorted([id_str for id_str in ids1.union(ids2) if id_str])
            combined_chunk_ids = "<SEP>".join(all_ids)
            edge_data = {"src_id": n1, "tgt_id": n2, "relation_name": "synonym_of", "keywords": f"synonym cluster={cluster}",
                         "description": f"{n1} is the synonym of {n2}", "source_id": combined_chunk_ids, "weight": 1.0,}
            G.add_edge(n1, n2, **edge_data)
        print(f"✅ Synonym edges added: {len(synonym_pairs)}")

    elif merge_type == "random_deletion":
        num_to_merge = int(len(G.nodes()) * similarity)
        print(f" -> Random merge mode: randomly merging {num_to_merge} nodes.")

        merged_count = 0
        nodes_to_merge = random.sample(list(G.nodes()), num_to_merge)

        for node_to_merge in tqdm(nodes_to_merge):
            # Pick a random target node
            target_node = random.choice([n for n in G.nodes if n != node_to_merge])
            if node_to_merge not in G or target_node not in G:
                continue

            source_data, target_data = G.nodes[node_to_merge], G.nodes[target_node]
            merged_data = MergeEntity.merge_info(target_data, source_data, summarizer=None)
            nx.set_node_attributes(G, {target_node: merged_data})

            for neighbor in list(G.neighbors(node_to_merge)):
                edge_data = G.get_edge_data(node_to_merge, neighbor)
                if edge_data['tgt_id'] == node_to_merge:
                    edge_data['tgt_id'] = target_node
                if edge_data['src_id'] == node_to_merge:
                    edge_data['src_id'] = target_node

                if G.has_edge(target_node, neighbor):
                    existing_edge_data = G.get_edge_data(target_node, neighbor)
                    merged_edge_data = MergeRelationship.merge_info(existing_edge_data, edge_data)
                    G.add_edge(target_node, neighbor, **merged_edge_data)
                else:
                    G.add_edge(target_node, neighbor, **edge_data)

            G.remove_node(node_to_merge)
            merged_count += 1
    else:
        raise ValueError(f"Invalid merge_type: {merge_type}")
        

    print(f"✅ Merging complete. {merged_count} nodes were merged.")

    # [Step 4 - PASS 2] Summarize long attributes for nodes AND relations
    print("\n[Step 4] Pass 2: Finding and summarizing long attributes...")
    if summarizer:
        # --- Summarize NODE descriptions ---
        print(" -> Summarizing long node descriptions...")
        nodes_to_summarize = [node for node, data in G.nodes(data=True) if len(summarizer.encoder.encode(data.get("description", ""))) > summarizer.config.token_check_threshold]
        if nodes_to_summarize:
            print(f" -> Found {len(nodes_to_summarize)} nodes to summarize.")
            # Prepare arguments for multiprocessing
            summarize_args = [(node, G.nodes[node].get("description"), api_key, base_url, model_name) for node in nodes_to_summarize]
            
            # Use ProcessPoolExecutor for parallel processing
            with ProcessPoolExecutor(max_workers=process_num) as executor:
                results = list(tqdm(executor.map(summarize_node_sync, summarize_args), 
                                  total=len(summarize_args), 
                                  desc="Summarizing nodes"))
            
            # Update graph with summarized descriptions
            for node, summary in results:
                nx.set_node_attributes(G, {node: {"description": summary}})
        else:
            print(" -> No nodes required summarization.")
        
        # --- Summarize RELATION descriptions and keywords ---
        print(" -> Summarizing long relation attributes (descriptions and keywords)...")
        relations_summarized_count = 0
        tasks = []
        for u, v, data in G.edges(data=True):
            summarized_this_edge = False
            # Check and summarize description
            rel_description = data.get("description", "")
            if rel_description and len(summarizer.encoder.encode(rel_description)) > summarizer.config.token_check_threshold:
                item_name = f"Relation from '{u}' to '{v}'"
                tasks.append(summarizer.summarize(item_name, rel_description, text_type='relation_description'))
                summarized_this_edge = True

            # Check and summarize keywords
            keywords = data.get("keywords", "")
            if keywords and len(summarizer.encoder.encode(keywords)) > summarizer.config.token_check_threshold:
                item_name = f"Keywords for relation from '{u}' to '{v}'"
                tasks.append(summarizer.summarize(item_name, keywords, text_type='keywords'))
                summarized_this_edge = True

        # Run all tasks with a progress bar
        summaries = asyncio.run(tqdm_asyncio.gather(*tasks, desc="Summarizing relations"))

        # Assign summaries back to the graph
        for (u, v, data), summary in zip(G.edges(data=True), summaries):
            if "description" in data:
                G[u][v]['description'] = summary
            if "keywords" in data:
                G[u][v]['keywords'] = summary
        
        if relations_summarized_count > 0:
            print(f" -> Summarized attributes for {relations_summarized_count} relations.")
        else:
            print(" -> No relations required summarization.")

        print("✅ Summarization complete.")
    else:
        print(" -> Summarizer not available, skipping summarization step.")

    # Step 5: Remove edges with reason score less than threshold_edge_reason
    print("\n[Step 5] Removing edges with reason score less than threshold_edge_reason...")
    file_name = f"edge_reason_{threshold_edge_reason:.2f}.jsonl"
    file_path = os.path.join(graph_path.split("graph_storage")[0], "edge_process", file_name)

    # Filter out empty or whitespace-only lines before parsing
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    edge_data = [json.loads(line) for line in lines if line.strip()]
    
    edges_to_remove = []
    for edge in edge_data:
        if edge is not None and 'score' in edge and edge['score'] is not None and edge['score'] < threshold_edge_reason:
            edges_to_remove.append((edge['source'], edge['destination']))
            
    for u, v in edges_to_remove:
        if G.has_edge(u, v):
            G.remove_edge(u, v)

    print(f" -> Removed {len(edges_to_remove)} edges.")

    # Step 5 & 6: Remove self-loops and save
    print("\n[Step 5] Removing self-loops...")
    self_loops = list(nx.selfloop_edges(G))
    if self_loops:
        print(f" -> Found and removed {len(self_loops)} self-loops.")
        G.remove_edges_from(self_loops)
    else: print(" -> No self-loops found.")
    print(f" -> Final graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    print(f"\n[Step 6] Saving new graph to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nx.write_graphml(G, output_path)
    print("✅ New graph saved successfully.")
    print("\n--- Process Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge similar nodes and summarize attributes in a GraphML file.")
    parser.add_argument("--similarity", type=float, default=0.2, help="The similarity percentage (e.g., 0.20).")
    parser.add_argument("--threshold_edge_reason", type=float, default=0.0, help="Threshold of the edge reason file (e.g., 0.20).")
    parser.add_argument("--dataset_name", type=str, default="cs", help="Name of the dataset.")
    parser.add_argument("--graph_path", type=str, default="Result/cs/rkg_graph/graph_storage/graph_storage_nx_data.graphml", help="Name of the graph file.")
    parser.add_argument("--similar_nodes_path", type=str, default="Result/cs/rkg_graph/node_neighbors/cs_0.40_0.65_llm_node_neighbor_reduction_only.json", help="Name of the similar nodes file.")
    parser.add_argument("--output_path", type=str, default="Result/cs/rkg_graph/graph_storage/graph_storage_nx_data_nodes_20.0_random1.graphml", help="Name of the output graph file.")
    parser.add_argument("--merge_type", type=str, choices=["reduction_only", "reduction_synonym", "synonym_only", "random_deletion"], default="random_deletion", help="Merge type")
    args = parser.parse_args()
    merge_similar_nodes(args.dataset_name, args.graph_path, args.similar_nodes_path, args.output_path, args.threshold_edge_reason, args.similarity, args.merge_type)