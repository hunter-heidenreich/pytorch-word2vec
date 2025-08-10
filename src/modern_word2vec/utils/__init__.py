"""Utility functions for Word2Vec."""

import json
import os
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from modern_word2vec.config import DeviceType


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer: DeviceType = None) -> torch.device:
    """Get the best available device with fallback hierarchy.

    Args:
        prefer: Preferred device type ("cuda", "mps", "cpu")

    Returns:
        PyTorch device
    """
    device_checks = [
        (prefer, lambda: prefer in ["cuda", "mps", "cpu"]),
        ("cuda", torch.cuda.is_available),
        (
            "mps",
            lambda: hasattr(torch.backends, "mps") and torch.backends.mps.is_available,
        ),
        ("cpu", lambda: True),  # CPU is always available
    ]

    for device_type, check_fn in device_checks:
        if device_type and check_fn():
            if device_type == "cuda" and torch.cuda.is_available():
                return torch.device("cuda")
            elif (
                device_type == "mps"
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                return torch.device("mps")
            elif device_type == "cpu":
                return torch.device("cpu")

    return torch.device("cpu")


def setup_device_optimizations(device: torch.device) -> None:
    """Configure device-specific optimizations.

    Args:
        device: PyTorch device to optimize for
    """
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    try:
        torch.set_float32_matmul_precision("medium")
    except Exception:
        pass


def export_embeddings(model: nn.Module, out_dir: str, dataset) -> None:
    """Export model embeddings and vocabulary.

    Args:
        model: Trained Word2Vec model
        out_dir: Output directory
        dataset: Dataset with vocabulary mappings
    """
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        embeddings = model.in_embeddings.weight.detach().cpu().numpy()

    # Save embeddings and vocabulary
    np.save(os.path.join(out_dir, "embeddings.npy"), embeddings)

    vocab_data = {
        "idx_to_word": dataset.idx_to_word,
        "word_to_idx": dataset.word_to_idx,
    }

    with open(os.path.join(out_dir, "vocab.json"), "w") as f:
        json.dump(vocab_data, f, indent=2)


def load_vocab_and_embeddings(run_dir: str) -> Tuple[dict, dict, np.ndarray]:
    """Load vocabulary and embeddings from saved files.

    Args:
        run_dir: Directory containing saved files

    Returns:
        Tuple of (idx_to_word, word_to_idx, embeddings)

    Raises:
        FileNotFoundError: If required files are not found
        ValueError: If vocabulary format is invalid
    """
    vocab_path = os.path.join(run_dir, "vocab.json")
    embeddings_path = os.path.join(run_dir, "embeddings.npy")

    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    with open(vocab_path, "r") as f:
        vocab_data = json.load(f)

    embeddings = np.load(embeddings_path)

    # Ensure proper mapping types
    idx_to_word = {int(k): v for k, v in vocab_data["idx_to_word"].items()}
    word_to_idx = vocab_data["word_to_idx"]

    return idx_to_word, word_to_idx, embeddings


def compute_cosine_similarities(
    query_embedding: np.ndarray, all_embeddings: np.ndarray
) -> np.ndarray:
    """Compute cosine similarities between query and all embeddings.

    Args:
        query_embedding: Single embedding vector
        all_embeddings: Matrix of all embeddings

    Returns:
        Array of cosine similarities
    """
    query_norm = np.linalg.norm(query_embedding)
    all_norms = np.linalg.norm(all_embeddings, axis=1)

    # Prevent division by zero
    denominator = all_norms * query_norm + 1e-9
    similarities = np.dot(all_embeddings, query_embedding) / denominator

    return similarities


def find_similar(
    word: str, model: nn.Module, dataset, top_n: int = 5
) -> List[Tuple[str, float]]:
    """Find words similar to a given word.

    Args:
        word: Query word
        model: Trained Word2Vec model
        dataset: Dataset with vocabulary mappings
        top_n: Number of similar words to return

    Returns:
        List of (word, similarity_score) tuples
    """
    word_idx = dataset.word_to_idx.get(word)
    if word_idx is None:
        print(f"Word '{word}' not in vocabulary.")
        return []

    with torch.no_grad():
        query_embedding = model.in_embeddings.weight[word_idx].detach().cpu().numpy()
        all_embeddings = model.in_embeddings.weight.detach().cpu().numpy()

    similarities = compute_cosine_similarities(query_embedding, all_embeddings)

    # Get top similar words (excluding the query word itself)
    top_indices = np.argsort(-similarities)[1 : top_n + 1]
    result = [(dataset.idx_to_word[i], float(similarities[i])) for i in top_indices]

    print(f"Words similar to '{word}':")
    for word_sim, score in result:
        print(f" - {word_sim} (Similarity: {score:.3f})")

    return result
