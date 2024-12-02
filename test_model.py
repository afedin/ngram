"""
Test script for the n-gram model.
Tests various aspects of the model including:
1. Data preprocessing
2. Model generation
3. Statistical metrics
4. Geographic validation
"""

import re
import os
from ngram import tokenize, NgramModel, normalize_coordinate, RNG, sample_discrete
import numpy as np

def test_coordinate_format(text):
    """Test if coordinates are in the correct format."""
    coord_pattern = r'-?\d+\.\d{3}'  # Matches numbers with exactly 3 decimal places
    coords = re.findall(coord_pattern, text)
    return len(coords) > 0, len(coords)

def test_address_structure(text):
    """Test if the generated text follows basic address structure."""
    # Check for common address elements
    has_number = bool(re.search(r'\d+', text))
    has_word = bool(re.search(r'[a-zA-Z]+', text))
    has_postcode = bool(re.search(r'[A-Z0-9]{2,4}\s?[0-9][A-Z]{2}', text))
    
    return {
        'has_number': has_number,
        'has_word': has_word,
        'has_postcode': has_postcode
    }

def test_tokenization():
    """Test the tokenization function."""
    test_cases = [
        "55.853729,-4.254518|40 Carlton Pl|Glasgow|Lanarkshire|G5 9TS",
        "55.950042,-4.24762|East Blairskeith|Glasgow|Lanarkshire|G64 4AX"
    ]
    
    for text in test_cases:
        tokens = tokenize(text)
        print(f"\nInput: {text}")
        print(f"Tokens: {tokens}")
        
        # Test coordinate normalization
        coords = text.split('|')[0].split(',')
        norm_coords = [normalize_coordinate(c) for c in coords]
        print(f"Normalized coordinates: {norm_coords}")

def test_model_generation(model, token_to_idx, idx_to_token, n_samples=5):
    """Test the model's generation capabilities."""
    sample_rng = RNG(1337)
    
    print("\nGenerated samples with analysis:")
    for i in range(n_samples):
        tape = [token_to_idx['\n']] * (model.seq_len - 1)
        generated_tokens = []
        
        while True:
            probs = model(tape)
            coinf = sample_rng.random()
            next_token = sample_discrete(probs.tolist(), coinf)
            next_token_str = idx_to_token[next_token]
            
            generated_tokens.append(next_token_str)
            if next_token_str == '\n' or len(generated_tokens) > 100:
                break
            
            tape.append(next_token)
            if len(tape) > model.seq_len - 1:
                tape = tape[1:]
        
        generated_text = ' '.join(generated_tokens)
        print(f"\nSample {i+1}:")
        print(generated_text)
        
        # Analyze the generated text
        has_coords, coord_count = test_coordinate_format(generated_text)
        structure = test_address_structure(generated_text)
        
        print(f"Analysis:")
        print(f"- Contains coordinates: {has_coords} (count: {coord_count})")
        print(f"- Structure check: {structure}")

def train_and_test_model():
    """Train a model and test its performance."""
    # Read training data
    train_text = open('data/train.txt', 'r').read()
    val_text = open('data/val.txt', 'r').read()
    test_text = open('data/test.txt', 'r').read()
    
    # Tokenize all data
    all_tokens = []
    for text in [train_text, val_text, test_text]:
        all_tokens.extend(tokenize(text))
    
    # Create vocabulary
    unique_tokens = sorted(list(set(all_tokens)))
    vocab_size = len(unique_tokens)
    token_to_idx = {token: i for i, token in enumerate(unique_tokens)}
    idx_to_token = {i: token for i, token in enumerate(unique_tokens)}
    
    print(f"\nVocabulary statistics:")
    print(f"- Total unique tokens: {vocab_size}")
    print(f"- Sample tokens: {unique_tokens[:10]}")
    
    # Train a small model for testing
    model = NgramModel(vocab_size, seq_len=3, smoothing=0.01)
    train_tokens = [token_to_idx[t] for t in tokenize(train_text)]
    
    print("\nTraining model...")
    for i in range(0, len(train_tokens) - 3 + 1):
        model.train(train_tokens[i:i+3])
    
    # Test generation
    test_model_generation(model, token_to_idx, idx_to_token, n_samples=3)

def main():
    print("Testing tokenization:")
    test_tokenization()
    
    print("\nTesting model training and generation:")
    train_and_test_model()
    
    print("\nTest complete!")

if __name__ == '__main__':
    main() 