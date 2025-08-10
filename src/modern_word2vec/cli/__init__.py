"""CLI modules for Word2Vec."""

from modern_word2vec.cli.train import main as train_main
from modern_word2vec.cli.query import main as query_main

__all__ = ["train_main", "query_main"]
