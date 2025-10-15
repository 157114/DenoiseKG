import numpy as np
import torch
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
import random

def graph_to_triples(G):
    """Convert networkx GraphML to triples list (h, r, t)."""
    triples = []
    for u, v, data in G.edges(data=True):
        rel = data.get("relation_name")
        if not rel or rel.strip() == "":
            src_id = data.get("source_id", str(u))
            tgt_id = data.get("target_id", str(v))
            rel = f"{src_id}->{tgt_id}"
        triples.append((str(u), str(rel), str(v)))
    return triples

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

def ComplEx_embedding(G, embedding_dim=100, epochs=100, device="auto"):
    """Train ComplEx embeddings using PyKEEN and return node embeddings dict."""
    triples = graph_to_triples(G)
    train_triples, test_triples = split_and_fix_triples(triples)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build triples factories
    train_tf = TriplesFactory.from_labeled_triples(np.array(train_triples, dtype=str))
    test_tf = TriplesFactory.from_labeled_triples(
        np.array(test_triples, dtype=str),
        entity_to_id=train_tf.entity_to_id,
        relation_to_id=train_tf.relation_to_id
    )

    result = pipeline(
        model="ComplEx",
        training=train_tf,
        testing=test_tf,
        model_kwargs=dict(embedding_dim=embedding_dim),
        training_kwargs=dict(num_epochs=epochs),
        device=device,
    )

    model = result.model
    node_embeddings_dict = {}
    for entity, idx in train_tf.entity_to_id.items():
        vec = model.entity_representations[0](torch.as_tensor([idx], device=model.device)).detach().cpu().numpy()
        node_embeddings_dict[entity] = np.concatenate([vec.real, vec.imag], axis=-1).squeeze()

    return node_embeddings_dict
