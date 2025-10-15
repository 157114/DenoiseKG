import os
import numpy as np
import torch
import networkx as nx
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
import random

def split_and_fix_triples(triples, train_ratio=0.8):
    """Shuffle triples, split into train/test, move unseen test triples into train."""
    random.shuffle(triples)
    n_train = int(len(triples) * train_ratio)
    train_triples = triples[:n_train]
    test_triples = triples[n_train:]


    train_entities = set([t[0] for t in train_triples] + [t[2] for t in train_triples])
    train_relations = set([t[1] for t in train_triples])

    extra_to_train = [t for t in test_triples
                      if t[0] not in train_entities
                      or t[2] not in train_entities
                      or t[1] not in train_relations]

    for t in extra_to_train:
        test_triples.remove(t)
        train_triples.append(t)

    return train_triples, test_triples


def CompGCN_embedding(G, epochs=100, embedding_dim=100, batch_size=128, device="auto"):
    """Train CompGCN embeddings on a networkx Graph G using PyKEEN."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Convert graph to triples
    triples = []
    for u, v, data in G.edges(data=True):
        rel = data.get("relation_name")
        if not rel or rel.strip() == "":
            src_id = data.get("source_id", str(u))
            tgt_id = data.get("target_id", str(v))
            rel = f"{src_id}->{tgt_id}"
        triples.append((str(u), str(rel), str(v)))

    # Safe train/test split
    train_triples, test_triples = split_and_fix_triples(triples)

    # Build triples factories with inverse triples
    train_tf = TriplesFactory.from_labeled_triples(
        np.array(train_triples, dtype=str),
        create_inverse_triples=True
    )
    test_tf = TriplesFactory.from_labeled_triples(
        np.array(test_triples, dtype=str),
        entity_to_id=train_tf.entity_to_id,
        relation_to_id=train_tf.relation_to_id,
        create_inverse_triples=True
    )

    # Training
    print("Training CompGCN...")
    result = pipeline(
        model="CompGCN",
        training=train_tf,
        testing=test_tf,
        model_kwargs=dict(embedding_dim=embedding_dim),
        training_kwargs=dict(num_epochs=epochs, batch_size=batch_size, drop_last=False),
        device=device
    )
    model = result.model

    # Collect entity embeddings
    entity_embeddings = {}
    for entity in train_tf.entity_to_id:
        idx = train_tf.entity_to_id[entity]
        vec = model.entity_representations[0](
            torch.as_tensor([idx], device=model.device)
        ).detach().cpu().numpy().flatten()
        if len(vec) != embedding_dim:
            vec = np.zeros(embedding_dim, dtype=np.float32)
        entity_embeddings[entity] = vec.tolist()

    # Collect relation embeddings
    relation_embeddings = {}
    for relation in train_tf.relation_to_id:
        idx = train_tf.relation_to_id[relation]
        vec = model.relation_representations[0](
            torch.as_tensor([idx], device=model.device)
        ).detach().cpu().numpy().flatten()
        if len(vec) != embedding_dim:
            vec = np.zeros(embedding_dim, dtype=np.float32)
        relation_embeddings[relation] = vec.tolist()

    return entity_embeddings, relation_embeddings
