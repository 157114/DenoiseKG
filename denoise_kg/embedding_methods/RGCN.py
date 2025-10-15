import numpy as np
import torch
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from sklearn.model_selection import train_test_split
import networkx as nx
import warnings

# Ignore specific PyKEEN warning
warnings.filterwarnings("ignore", message=".*has parameters, but no reset_parameters.*")


def load_triples_from_graph(G):
    """Convert networkx Graph into (head, relation, tail) triples."""
    triples = []
    for u, v, data in G.edges(data=True):
        rel = data.get("relation", "related_to")
        triples.append((str(u), str(rel), str(v)))
    return triples


def safe_train_test_split(triples, test_size=0.2, random_state=42):
    """Split triples so all entities in test appear in train."""
    train_triples, test_triples = train_test_split(
        triples, test_size=test_size, random_state=random_state
    )
    train_entities = {h for h, r, t in train_triples} | {t for h, r, t in train_triples}
    moved = []

    for triple in test_triples[:]:
        h, r, t = triple
        if h not in train_entities or t not in train_entities:
            train_triples.append(triple)
            test_triples.remove(triple)
            moved.append(triple)

    return train_triples, test_triples


def RGCN_embedding(
    G,
    embedding_dim=100,
    epochs=100,
    batch_size=128,
    device="auto",
):
    """Train R-GCN embeddings using PyKEEN and return entity embeddings as a dict."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    triples = load_triples_from_graph(G)
    train_triples, test_triples = safe_train_test_split(triples)

    # Build PyKEEN triples factories
    train_tf = TriplesFactory.from_labeled_triples(
        np.array(train_triples, dtype=str),
        create_inverse_triples=False,
    )
    test_tf = TriplesFactory.from_labeled_triples(
        np.array(test_triples, dtype=str),
        entity_to_id=train_tf.entity_to_id,
        relation_to_id=train_tf.relation_to_id,
        create_inverse_triples=False,
    )

    # Train R-GCN
    result = pipeline(
        model="RGCN",
        training=train_tf,
        testing=test_tf,
        training_loop="sLCWA",
        model_kwargs=dict(embedding_dim=embedding_dim),
        training_kwargs=dict(
            num_epochs=epochs,
            batch_size=batch_size,
            drop_last=False,
            sampler="schlichtkrull",
        ),
        device=device,
    )

    model = result.model

    # Collect entity embeddings
    entity_embeddings = {}
    for entity in train_tf.entity_to_id:
        if not entity:  # skip empty string
            continue
        idx = train_tf.entity_to_id[entity]
        vec = model.entity_representations[0](torch.as_tensor([idx], device=model.device))
        if isinstance(vec, torch.Tensor):
            vec = vec.detach().cpu().numpy()
        vec = vec.flatten()
        if len(vec) != embedding_dim:
            vec = np.zeros(embedding_dim, dtype=np.float32)
        entity_embeddings[entity] = vec.tolist() 

    # Collect relation embeddings
    relation_embeddings = {}
    for relation in train_tf.relation_to_id:
        if not relation:  # skip empty string
            continue
        idx = train_tf.relation_to_id[relation]
        vec = model.relation_representations[0](torch.as_tensor([idx], device=model.device))
        if isinstance(vec, torch.Tensor):
            vec = vec.detach().cpu().numpy()
        vec = vec.flatten()
        if len(vec) != embedding_dim:
            vec = np.zeros(embedding_dim, dtype=np.float32)
        relation_embeddings[relation] = vec.tolist()

    return entity_embeddings, relation_embeddings
