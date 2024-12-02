"""
n-gram Language Model for Geographic Data

This version is adapted to work with geographic data including coordinates,
addresses, and location information. Uses sparse matrices for efficiency.
"""

import os
import itertools
import numpy as np
from collections import defaultdict
import re

# -----------------------------------------------------------------------------
# data processing and tokenization

def normalize_coordinate(coord_str):
    """Round coordinates to reduce vocabulary size."""
    try:
        return f"{float(coord_str):.3f}"  # Reduce precision to 3 decimal places
    except ValueError:
        return coord_str

def tokenize(text):
    """
    Tokenize text into individual tokens.
    Handles numbers, letters, spaces, and special characters.
    """
    tokens = []
    for line in text.split('\n'):
        if not line.strip():
            continue
        # Split on pipe and process each part
        parts = line.split('|')
        for i, part in enumerate(parts):
            part = part.strip()
            if i == 0:  # First part is coordinates
                # Split and normalize coordinates
                try:
                    lat, lon = part.split(',')
                    tokens.extend([normalize_coordinate(lat), normalize_coordinate(lon)])
                except ValueError:
                    # If coordinates can't be split, add as is
                    tokens.append(part)
            else:
                # For other parts, split on space
                tokens.extend(part.split())
        tokens.append('\n')  # Add newline token between entries
    return tokens

def detokenize(tokens):
    """Convert tokens back into text."""
    return ' '.join(tokens)

# -----------------------------------------------------------------------------
# random number generation

class RNG:
    def __init__(self, seed):
        self.state = seed

    def random_u32(self):
        self.state ^= (self.state >> 12) & 0xFFFFFFFFFFFFFFFF
        self.state ^= (self.state << 25) & 0xFFFFFFFFFFFFFFFF
        self.state ^= (self.state >> 27) & 0xFFFFFFFFFFFFFFFF
        return ((self.state * 0x2545F4914F6CDD1D) >> 32) & 0xFFFFFFFF

    def random(self):
        return (self.random_u32() >> 8) / 16777216.0

# -----------------------------------------------------------------------------
# sampling from the model

def sample_discrete(probs, coinf):
    cdf = 0.0
    for i, prob in enumerate(probs):
        cdf += prob
        if coinf < cdf:
            return i
    return len(probs) - 1

# -----------------------------------------------------------------------------
# n-gram model with sparse storage

class NgramModel:
    def __init__(self, vocab_size, seq_len, smoothing=0.1):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        # Use dictionary for sparse storage
        self.counts = defaultdict(int)
        self.context_totals = defaultdict(int)
        self.uniform = np.ones(self.vocab_size, dtype=np.float32) / self.vocab_size

    def train(self, tape):
        assert isinstance(tape, list)
        assert len(tape) == self.seq_len
        context = tuple(tape[:-1])
        target = tape[-1]
        self.counts[(context, target)] += 1
        self.context_totals[context] += 1

    def get_counts(self, context):
        context = tuple(context)
        counts = np.zeros(self.vocab_size, dtype=np.float32)
        for i in range(self.vocab_size):
            counts[i] = self.counts.get((context, i), 0)
        return counts

    def __call__(self, context):
        context = tuple(context)
        counts = self.get_counts(context)
        counts += self.smoothing
        counts_sum = counts.sum()
        probs = counts / counts_sum if counts_sum > 0 else self.uniform
        return probs

# -----------------------------------------------------------------------------
# data iteration and evaluation

def dataloader(tokens, window_size):
    for i in range(len(tokens) - window_size + 1):
        yield tokens[i:i+window_size]

def eval_split(model, tokens):
    sum_loss = 0.0
    count = 0
    for tape in dataloader(tokens, model.seq_len):
        x = tape[:-1]
        y = tape[-1]
        probs = model(x)
        prob = probs[y]
        sum_loss += -np.log(prob)
        count += 1
    mean_loss = sum_loss / count if count > 0 else 0.0
    return mean_loss

# -----------------------------------------------------------------------------
# main training and generation

def main():
    # Read all data files first
    train_text = open('data/train.txt', 'r').read()
    val_text = open('data/val.txt', 'r').read()
    test_text = open('data/test.txt', 'r').read()
    
    # Tokenize all data to build vocabulary
    all_tokens = []
    for text in [train_text, val_text, test_text]:
        all_tokens.extend(tokenize(text))
    
    # Create vocabulary from all tokens
    unique_tokens = sorted(list(set(all_tokens)))
    vocab_size = len(unique_tokens)
    token_to_idx = {token: i for i, token in enumerate(unique_tokens)}
    idx_to_token = {i: token for i, token in enumerate(unique_tokens)}
    
    # Convert text to indices
    train_tokens = [token_to_idx[t] for t in tokenize(train_text)]
    val_tokens = [token_to_idx[t] for t in tokenize(val_text)]
    test_tokens = [token_to_idx[t] for t in tokenize(test_text)]
    
    print(f"Vocabulary size: {vocab_size}")
    print("Sample tokens:", unique_tokens[:10])
    
    # Hyperparameter search
    seq_lens = [3, 4]  # Reduced sequence lengths
    smoothings = [0.01, 0.1, 0.3, 1.0]
    best_loss = float('inf')
    best_kwargs = {}
    
    for seq_len, smoothing in itertools.product(seq_lens, smoothings):
        model = NgramModel(vocab_size, seq_len, smoothing)
        for tape in dataloader(train_tokens, seq_len):
            model.train(tape)
        
        train_loss = eval_split(model, train_tokens)
        val_loss = eval_split(model, val_tokens)
        print(f"seq_len {seq_len} | smoothing {smoothing:.2f} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_kwargs = {'seq_len': seq_len, 'smoothing': smoothing}
    
    # Train final model with best parameters
    print("Best hyperparameters:", best_kwargs)
    model = NgramModel(vocab_size, **best_kwargs)
    for tape in dataloader(train_tokens, model.seq_len):
        model.train(tape)
    
    # Generate samples
    sample_rng = RNG(1337)
    n_samples = 5
    print("\nGenerated samples:")
    for _ in range(n_samples):
        tape = [token_to_idx['\n']] * (model.seq_len - 1)  # Start with newline token
        generated_tokens = []
        
        while True:
            probs = model(tape)
            coinf = sample_rng.random()
            next_token = sample_discrete(probs.tolist(), coinf)
            next_token_str = idx_to_token[next_token]
            
            generated_tokens.append(next_token_str)
            if next_token_str == '\n' or len(generated_tokens) > 100:  # Prevent infinite sequences
                break
            
            tape.append(next_token)
            if len(tape) > model.seq_len - 1:
                tape = tape[1:]
        
        print(' '.join(generated_tokens))
    
    # Evaluate final model
    test_loss = eval_split(model, test_tokens)
    test_perplexity = np.exp(test_loss)
    print(f"\nTest loss: {test_loss:.4f}, Test perplexity: {test_perplexity:.4f}")

if __name__ == '__main__':
    main()
