"""Hierarchical Softmax implementation for Word2Vec."""

import heapq
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


class HuffmanTree:
    """Huffman tree for building binary hierarchy of vocabulary words.
    
    This implementation follows the original Word2Vec approach where:
    - Frequent words get shorter binary codes (closer to root)
    - Less frequent words get longer binary codes (farther from root)
    - Each internal node has an associated parameter vector
    """

    def __init__(self, word_counts: Dict[str, int], word_to_idx: Dict[str, int]):
        """Initialize Huffman tree from word frequencies.

        Args:
            word_counts: Dictionary mapping words to their frequency counts
            word_to_idx: Dictionary mapping words to vocabulary indices
        """
        self.word_to_idx = word_to_idx
        self.idx_to_word = {idx: word for word, idx in word_to_idx.items()}
        
        # Build the Huffman tree
        self.word_codes, self.word_paths, self.num_inner_nodes = self._build_tree(word_counts)
        
    def _build_tree(self, word_counts: Dict[str, int]) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], int]:
        """Build Huffman tree and return binary codes and paths for each word.

        Args:
            word_counts: Word frequency counts

        Returns:
            Tuple of (word_codes, word_paths, num_inner_nodes) where:
            - word_codes: Maps word index to binary code (list of 0s and 1s)
            - word_paths: Maps word index to path of internal node indices
            - num_inner_nodes: Total number of internal nodes in the tree
        """
        # Create initial leaf nodes for all words using a min-heap
        vocab_size = len(self.word_to_idx)
        heap = []
        
        # Add all words as leaf nodes with their frequencies
        # Use negative node_id as tiebreaker to ensure deterministic ordering
        for word, word_idx in self.word_to_idx.items():
            count = word_counts.get(word, 1)  # Default count of 1 if not found
            # heapq is a min-heap, so we store (count, tiebreaker, node_data)
            heapq.heappush(heap, (count, -word_idx, (word_idx, None, None)))
        
        # Build Huffman tree bottom-up using heapq for O(N log N) performance
        next_inner_node_id = vocab_size  # Internal nodes get IDs starting from vocab_size
        
        while len(heap) > 1:
            # Extract two nodes with smallest frequencies
            count1, _, left_data = heapq.heappop(heap)
            count2, _, right_data = heapq.heappop(heap)
            
            # Create new internal node
            merged_count = count1 + count2
            # Use negative node_id as tiebreaker for deterministic behavior
            heapq.heappush(heap, (
                merged_count, 
                -next_inner_node_id,
                (next_inner_node_id, left_data, right_data)
            ))
            next_inner_node_id += 1
        
        # Extract root
        if heap:
            _, _, root_data = heap[0]
        else:
            root_data = None
            
        num_inner_nodes = next_inner_node_id - vocab_size
        
        # Generate codes and paths for each word
        word_codes = {}
        word_paths = {}
        
        if root_data is not None:
            self._generate_codes(root_data, [], [], word_codes, word_paths, vocab_size)
        
        return word_codes, word_paths, num_inner_nodes
    
    def _generate_codes(self, node_data, current_code: List[int], current_path: List[int], 
                       word_codes: Dict[int, List[int]], word_paths: Dict[int, List[int]], vocab_size: int):
        """Recursively generate binary codes and paths for words.

        Args:
            node_data: Current tree node data (node_id, left_data, right_data)
            current_code: Binary code path to current node
            current_path: Internal node indices on path to current node
            word_codes: Dictionary to store word codes
            word_paths: Dictionary to store word paths
            vocab_size: Size of vocabulary (to distinguish leaf from internal nodes)
        """
        node_id, left_data, right_data = node_data
        
        # If this is a leaf node (word), store its code and path
        if node_id < vocab_size:
            word_codes[node_id] = current_code.copy()
            word_paths[node_id] = current_path.copy()
        else:
            # Internal node - recurse to children
            if left_data is not None:
                self._generate_codes(left_data, current_code + [0], current_path + [node_id], 
                                   word_codes, word_paths, vocab_size)
            if right_data is not None:
                self._generate_codes(right_data, current_code + [1], current_path + [node_id], 
                                   word_codes, word_paths, vocab_size)


class HierarchicalSoftmax(nn.Module):
    """Hierarchical Softmax layer for efficient Word2Vec training.
    
    This replaces the standard softmax output layer with a binary tree structure
    where each word is represented as a leaf node. The probability of a word is
    computed as the product of probabilities along the path from root to leaf.
    
    This reduces computational complexity from O(V) to O(log V) where V is vocabulary size.
    """

    def __init__(self, embedding_dim: int, vocab_size: int, word_counts: Dict[str, int], 
                 word_to_idx: Dict[str, int]):
        """Initialize hierarchical softmax layer.

        Args:
            embedding_dim: Dimension of input embeddings
            vocab_size: Size of vocabulary
            word_counts: Word frequency counts for building Huffman tree
            word_to_idx: Mapping from words to vocabulary indices
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        
        # Build Huffman tree
        self.huffman_tree = HuffmanTree(word_counts, word_to_idx)
        
        # Create parameter matrix for internal nodes
        # Each internal node has a parameter vector of size embedding_dim
        self.inner_node_embeddings = nn.Embedding(
            self.huffman_tree.num_inner_nodes, 
            embedding_dim
        )
        
        # Initialize weights (similar to original word2vec initialization)
        self._init_weights()
        
        # Pre-compute tensors for efficient computation
        self._precompute_paths()
    
    def _init_weights(self):
        """Initialize internal node embeddings."""
        with torch.no_grad():
            # Initialize to zero as in original word2vec
            self.inner_node_embeddings.weight.zero_()
    
    def _precompute_paths(self):
        """Precompute paths and codes for all words as tensors for efficient batching."""
        max_path_length = max(len(path) for path in self.huffman_tree.word_paths.values()) if self.huffman_tree.word_paths else 0
        
        # Create padded tensors for all word paths and codes
        self.word_path_indices = torch.full((self.vocab_size, max_path_length), -1, dtype=torch.long)
        self.word_codes_tensor = torch.full((self.vocab_size, max_path_length), -1, dtype=torch.long)
        self.path_lengths = torch.zeros(self.vocab_size, dtype=torch.long)
        
        for word_idx in range(self.vocab_size):
            if word_idx in self.huffman_tree.word_paths:
                path = self.huffman_tree.word_paths[word_idx]
                code = self.huffman_tree.word_codes[word_idx]
                
                path_len = len(path)
                self.path_lengths[word_idx] = path_len
                
                if path_len > 0:
                    # Convert internal node IDs to indices in the embedding matrix
                    path_indices = [node_id - self.vocab_size for node_id in path]
                    self.word_path_indices[word_idx, :path_len] = torch.tensor(path_indices)
                    self.word_codes_tensor[word_idx, :path_len] = torch.tensor(code)
    
    def forward(self, input_embeddings: torch.Tensor, target_words: torch.Tensor) -> torch.Tensor:
        """Compute hierarchical softmax loss with full batching for GPU efficiency.

        Args:
            input_embeddings: Input embeddings of shape (batch_size, embedding_dim)
            target_words: Target word indices of shape (batch_size,)

        Returns:
            Loss tensor for the batch
        """
        batch_size = input_embeddings.shape[0]
        device = input_embeddings.device
        
        # Move precomputed tensors to the same device
        if self.word_path_indices.device != device:
            self.word_path_indices = self.word_path_indices.to(device)
            self.word_codes_tensor = self.word_codes_tensor.to(device)
            self.path_lengths = self.path_lengths.to(device)
        
        # Get maximum path length for this batch
        batch_path_lengths = self.path_lengths[target_words]  # (batch_size,)
        max_path_length = batch_path_lengths.max().item()
        
        if max_path_length == 0:
            # Handle degenerate case
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Extract paths and codes for the entire batch
        batch_paths = self.word_path_indices[target_words, :max_path_length]  # (batch_size, max_path_len)
        batch_codes = self.word_codes_tensor[target_words, :max_path_length]  # (batch_size, max_path_len)
        
        # Create mask for valid path positions
        path_mask = torch.arange(max_path_length, device=device).unsqueeze(0) < batch_path_lengths.unsqueeze(1)
        # (batch_size, max_path_len)
        
        # Get inner node embeddings for all paths in the batch
        # Flatten to get all unique inner node indices, then reshape
        valid_paths = batch_paths[path_mask]  # (total_valid_positions,)
        
        # Handle case where some paths might have invalid indices (-1)
        valid_mask = valid_paths >= 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Replace invalid indices with 0 temporarily (will be masked out anyway)
        safe_paths = torch.where(valid_paths >= 0, valid_paths, 0)
        inner_embeds_flat = self.inner_node_embeddings(safe_paths)  # (total_valid_positions, embedding_dim)
        
        # Reshape back to batch format
        inner_embeds = torch.zeros(batch_size, max_path_length, self.embedding_dim, 
                                 device=device, dtype=inner_embeds_flat.dtype)
        inner_embeds[path_mask] = inner_embeds_flat
        
        # Compute dot products for the entire batch
        # input_embeddings: (batch_size, embedding_dim)
        # inner_embeds: (batch_size, max_path_len, embedding_dim)
        dots = torch.sum(input_embeddings.unsqueeze(1) * inner_embeds, dim=2)  # (batch_size, max_path_len)
        
        # Apply sigmoid with correct signs based on codes
        # For code=1, we want sigmoid(dot), for code=0, we want sigmoid(-dot)
        adjusted_dots = dots * (2 * batch_codes.float() - 1)  # (batch_size, max_path_len)
        
        # Apply mask to ignore invalid positions
        adjusted_dots = torch.where(path_mask, adjusted_dots, torch.zeros_like(adjusted_dots))
        
        # Compute log probabilities
        log_probs = torch.nn.functional.logsigmoid(adjusted_dots)  # (batch_size, max_path_len)
        
        # Sum log probabilities along path dimension, respecting the mask
        masked_log_probs = torch.where(path_mask, log_probs, torch.zeros_like(log_probs))
        word_log_probs = masked_log_probs.sum(dim=1)  # (batch_size,)
        
        # Return negative log likelihood (average over batch)
        return -word_log_probs.mean()
    
    def predict_probabilities(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute probabilities for all words given input embeddings.
        
        Note: This is computationally expensive O(V log V) and mainly for evaluation.
        For large vocabularies, consider using approximate methods.

        Args:
            input_embeddings: Input embeddings of shape (batch_size, embedding_dim)

        Returns:
            Probability tensor of shape (batch_size, vocab_size)
        """
        batch_size = input_embeddings.shape[0]
        device = input_embeddings.device
        
        # Move precomputed tensors to device
        if self.word_path_indices.device != device:
            self.word_path_indices = self.word_path_indices.to(device)
            self.word_codes_tensor = self.word_codes_tensor.to(device)
            self.path_lengths = self.path_lengths.to(device)
        
        probs = torch.zeros(batch_size, self.vocab_size, device=device)
        
        for word_idx in range(self.vocab_size):
            path_length = self.path_lengths[word_idx].item()
            
            if path_length == 0:
                # Uniform probability for words with no path
                probs[:, word_idx] = 1.0 / self.vocab_size
                continue
            
            # Get path and codes for this word
            path_indices = self.word_path_indices[word_idx, :path_length]
            codes = self.word_codes_tensor[word_idx, :path_length]
            
            # Get internal node embeddings
            inner_embeds = self.inner_node_embeddings(path_indices)  # (path_length, embedding_dim)
            
            # Compute probabilities for all samples in batch
            for batch_idx in range(batch_size):
                input_embed = input_embeddings[batch_idx]
                
                # Compute dot products
                dots = torch.sum(input_embed * inner_embeds, dim=1)
                
                # Compute path probability
                adjusted_dots = dots * (2 * codes.float() - 1)
                path_probs = torch.sigmoid(adjusted_dots)
                word_prob = torch.prod(path_probs)
                
                probs[batch_idx, word_idx] = word_prob
        
        return probs


def build_word_counts_from_dataset(dataset) -> Dict[str, int]:
    """Build word frequency counts from a Word2Vec dataset.
    
    Args:
        dataset: Word2VecDataset instance
        
    Returns:
        Dictionary mapping words to their frequency counts
    """
    word_counts = {}
    
    # Get word counts from the dataset's vocabulary builder if available
    if hasattr(dataset, 'vocab_builder') and hasattr(dataset.vocab_builder, 'word_counts'):
        # The vocab builder stores a Counter object with word strings as keys
        vocab_counts = dataset.vocab_builder.word_counts
        for word in dataset.word_to_idx.keys():
            word_counts[word] = vocab_counts.get(word, 1)
    else:
        # Fallback: assign uniform counts
        for word in dataset.word_to_idx.keys():
            word_counts[word] = 1
    
    return word_counts
