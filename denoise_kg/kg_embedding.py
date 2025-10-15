# kg_embedding.py
import networkx as nx
import numpy as np
from embedding_methods.TransE import TransE_embedding
from embedding_methods.ComplEx import ComplEx_embedding
from embedding_methods.DistMult import DistMult_embedding
from embedding_methods.CompGCN import CompGCN_embedding
from embedding_methods.RGCN import RGCN_embedding
import warnings

# Ignore the RGCN warning about reset_parameters
warnings.filterwarnings("ignore", message=".*has parameters, but no reset_parameters.*")


def compute_kg_embeddings(G, method="TransE", embedding_dim=100, epochs=100, device="auto"):
    """
    Compute KG embeddings for a NetworkX graph.

    Args:
        G (nx.Graph): Input knowledge graph.
        method (str): Embedding method ("TransE", "DistMult", "ComplEx", "RGCN", "CompGCN").
        embedding_dim (int): Dimension of embeddings.
        epochs (int): Number of training epochs.
        device (str): Device to train on ("cpu", "cuda", "auto").

    Returns:
        embeddings (np.ndarray): Array of shape (n_nodes, embedding_dim).
        node_id_to_index (dict): Mapping from node_id to row index in embeddings.
    """
    nodes = list(G.nodes())

    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Compute embeddings using selected method
    if method == "TransE":
        node_embeddings_dict = TransE_embedding(G, embedding_dim=embedding_dim, epochs=epochs, device=device)
    elif method == "DistMult":
        node_embeddings_dict = DistMult_embedding(G, embedding_dim=embedding_dim, epochs=epochs, device=device)
    elif method == "ComplEx":
        node_embeddings_dict = ComplEx_embedding(G, embedding_dim=embedding_dim, epochs=epochs, device=device)
    elif method == "RGCN":
        node_embeddings_dict, _ = RGCN_embedding(G, embedding_dim=embedding_dim, epochs=epochs, device=device)
    elif method == "CompGCN":
        node_embeddings_dict, _ = CompGCN_embedding(G, embedding_dim=embedding_dim, epochs=epochs, device=device)
    else:
        raise ValueError(f"Invalid embedding method: {method}")

    # Ensure embedding_dim consistency
    embedding_dim_actual = len(next(iter(node_embeddings_dict.values())))
    if embedding_dim_actual != embedding_dim:
        print(f"[WARNING] embedding_dim={embedding_dim} but actual={embedding_dim_actual}, adjusting...")
        embedding_dim = embedding_dim_actual

    # Build embedding matrix
    embeddings = np.array([
        node_embeddings_dict.get(node, np.zeros(embedding_dim, dtype=np.float32))
        for node in nodes
    ], dtype=np.float32)

    node_id_to_index = {node: i for i, node in enumerate(nodes)}
    return embeddings, node_id_to_index


if __name__ == "__main__":
    import argparse
    import os
    import json

    parser = argparse.ArgumentParser(description="Compute KG embeddings for a GraphML file")
    parser.add_argument("--graphml_path", type=str, default="Result/mix/rkg_graph/graph_storage/graph_storage_nx_data.graphml")
    parser.add_argument("--output_dir", type=str, default="Result/mix/rkg_graph/kg_embeddings")
    parser.add_argument("--embedding_method", type=str, default="TransE", choices=["TransE", "DistMult", "ComplEx", "RGCN", "CompGCN"])
    parser.add_argument("--embedding_dim", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # Load graph
    G = nx.read_graphml(args.graphml_path)

    # Compute embeddings
    embeddings, node_id_to_index = compute_kg_embeddings(
        G,
        method=args.embedding_method,
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        device=args.device
    )

    # Save embeddings and node mapping
    os.makedirs(args.output_dir, exist_ok=True)
    embeddings_path = os.path.join(args.output_dir, f"{args.embedding_method}_kg_embeddings.npy")
    np.save(embeddings_path, embeddings)

    node_mapping_path = os.path.join(args.output_dir, f"{args.embedding_method}_kg_embeddings_node_mapping.json")
    with open(node_mapping_path, "w", encoding="utf-8") as f:
        json.dump(node_id_to_index, f, indent=2)

    print(f"[INFO] Saved embeddings to {embeddings_path}")
    print(f"[INFO] Saved node mapping to {node_mapping_path}")
