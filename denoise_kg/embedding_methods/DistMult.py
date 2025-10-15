import os
import numpy as np
import torch
import networkx as nx
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline

def DistMult_embedding(
    G: nx.Graph,
    epochs: int = 100,
    embedding_dim: int = 100,
    device: str = "auto",
):
    """
    Train DistMult embeddings on a NetworkX Graph.
    Returns: dict[node -> embedding]
    """
    # Handle device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DistMult] Running on device: {device}")

    # Convert Graph to triples
    triples = []
    for u, v, data in G.edges(data=True):
        rel = data.get("relation", "related_to")
        triples.append((str(u), str(rel), str(v)))

    entities = list(G.nodes())
    relations = list(set(nx.get_edge_attributes(G, "relation").values()))

    print(f"[DistMult] Extracted {len(triples)} triples")

    # Train/test split
    n_train = int(len(triples) * 0.8)
    train_triples = triples[:n_train]
    test_triples = triples[n_train:]

    # TriplesFactory
    train_tf = TriplesFactory.from_labeled_triples(
        np.array([tuple(map(str, t)) for t in train_triples], dtype=str)
    )
    test_tf = TriplesFactory.from_labeled_triples(
        np.array([tuple(map(str, t)) for t in test_triples], dtype=str)
    )

    # Train model
    result = pipeline(
        model="DistMult",
        training=train_tf,
        testing=test_tf,
        model_kwargs=dict(embedding_dim=embedding_dim),
        training_kwargs=dict(num_epochs=epochs),
        device=device,
    )
    model = result.model

    # Collect entity embeddings
    entity_embeddings = {}
    for entity in entities:
        if entity not in train_tf.entity_to_id:
            entity_embeddings[entity] = [0.0] * embedding_dim
        else:
            idx = train_tf.entity_to_id[entity]
            vec = (
                model.entity_representations[0](torch.as_tensor([idx], device=model.device))
                .detach()
                .cpu()
                .numpy()
                .flatten()
                .tolist()
            )
            entity_embeddings[entity] = vec

    return entity_embeddings
