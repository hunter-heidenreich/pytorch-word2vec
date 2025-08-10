"""Query CLI for Word2Vec embeddings."""

import json
from typing import List, Optional

import numpy as np

from modern_word2vec.cli_args import parse_query_args
from modern_word2vec.utils import load_vocab_and_embeddings, compute_cosine_similarities


def find_similar_words(
    word: str,
    idx_to_word: dict,
    word_to_idx: dict,
    embeddings: np.ndarray,
    top_n: int = 10,
) -> dict:
    """Find words similar to a query word.

    Args:
        word: Query word
        idx_to_word: Index to word mapping
        word_to_idx: Word to index mapping
        embeddings: Word embeddings matrix
        top_n: Number of similar words to return

    Returns:
        Dictionary with query results or error information
    """
    if word not in word_to_idx:
        return {"error": f"'{word}' not in vocabulary"}

    query_idx = word_to_idx[word]
    query_embedding = embeddings[query_idx]

    # Compute similarities
    similarities = compute_cosine_similarities(query_embedding, embeddings)

    # Get top similar words (excluding the query word itself)
    top_indices = np.argsort(-similarities)
    result = []

    for idx in top_indices:
        idx = int(idx)
        if idx == query_idx:
            continue  # Skip the query word itself

        word_text = idx_to_word[idx]
        similarity = float(similarities[idx])
        result.append((word_text, similarity))

        if len(result) >= top_n:
            break

    return {"word": word, "neighbors": result}


def main(argv: Optional[List[str]] = None) -> None:
    """Main query function.

    Args:
        argv: Optional command line arguments
    """
    args = parse_query_args(argv)

    try:
        idx_to_word, word_to_idx, embeddings = load_vocab_and_embeddings(args.run_dir)
    except (FileNotFoundError, ValueError) as e:
        print(json.dumps({"error": str(e)}))
        return

    result = find_similar_words(
        args.word, idx_to_word, word_to_idx, embeddings, args.topn
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
