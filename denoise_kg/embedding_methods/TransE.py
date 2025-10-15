import torch
import networkx as nx
import numpy as np
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline


def TransE_embedding(G, embedding_dim=100, epochs=100, device="auto"):
    """
    Train TransE embeddings on a NetworkX graph and return
    entity and relation embeddings as dictionaries.
    """
    # Convert edges to triples
    triples = []
    for u, v, data in G.edges(data=True):
        rel = data.get("relation") or data.get("relation_name") or "related_to"
        triples.append((str(u), str(rel), str(v)))

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Train/test split
    n_train = int(len(triples) * 0.8)
    train_triples = triples[:n_train]
    test_triples = triples[n_train:]

    train_array = np.array([tuple(map(str, t)) for t in train_triples], dtype=str)
    test_array = np.array([tuple(map(str, t)) for t in test_triples], dtype=str)

    train_tf = TriplesFactory.from_labeled_triples(train_array)
    test_tf = TriplesFactory.from_labeled_triples(test_array)

    # Train TransE
    result = pipeline(
        model="TransE",
        training=train_tf,
        testing=test_tf,
        model_kwargs=dict(embedding_dim=embedding_dim),
        training_kwargs=dict(num_epochs=epochs),
        device=device,
    )

    model = result.model

    # Extract entity embeddings
    entity_embeddings = {}
    for entity, idx in train_tf.entity_to_id.items():
        vec = model.entity_representations[0](torch.as_tensor([idx], device=model.device))
        if isinstance(vec, torch.Tensor):
            vec = vec.detach().cpu().numpy().flatten()
        entity_embeddings[entity] = vec.tolist()
    return entity_embeddings
