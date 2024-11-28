"""
Preprocesses geographic data for n-gram language model training.
Handles addresses, coordinates, and location information.
"""

import csv
import random
import os
from typing import List, Tuple
import re

def clean_text(text: str) -> str:
    """
    Clean and normalize text:
    - Convert to lowercase
    - Replace special characters with spaces
    - Remove extra whitespace
    """
    text = text.lower()
    # Replace special characters with space
    text = re.sub(r'[^a-z0-9\s.-]', ' ', text)
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def format_coordinates(lat: str, lon: str) -> str:
    """Format coordinates to consistent decimal places."""
    try:
        lat_float = float(lat)
        lon_float = float(lon)
        return f"{lat_float:.6f} {lon_float:.6f}"
    except ValueError:
        return ""

def create_training_example(row: dict) -> str:
    """
    Create a formatted string for training from a data row.
    Format: coordinates | address | locality | region | postcode
    """
    parts = []
    
    # Add coordinates
    coords = format_coordinates(row['latitude'], row['longitude'])
    if coords:
        parts.append(coords)
    
    # Add other fields with cleaning
    fields = ['address', 'locality', 'region', 'postcode']
    for field in fields:
        if row.get(field):
            cleaned = clean_text(row[field])
            if cleaned:
                parts.append(cleaned)
    
    # Join all parts with special separator
    return ' | '.join(parts)

def split_data(examples: List[str], train_ratio=0.8, val_ratio=0.1) -> Tuple[List[str], List[str], List[str]]:
    """Split data into train/val/test sets."""
    random.seed(42)  # For reproducibility
    random.shuffle(examples)
    
    n = len(examples)
    train_idx = int(n * train_ratio)
    val_idx = int(n * (train_ratio + val_ratio))
    
    train_data = examples[:train_idx]
    val_data = examples[train_idx:val_idx]
    test_data = examples[val_idx:]
    
    return train_data, val_data, test_data

def write_to_file(examples: List[str], filename: str):
    """Write examples to file, one per line."""
    with open(filename, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(example + '\n')

def main():
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Read and process the CSV file
    examples = []
    with open('ngram/data/places.txt', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            example = create_training_example(row)
            if example:  # Only add non-empty examples
                examples.append(example)
    
    # Split the data
    train_data, val_data, test_data = split_data(examples)
    
    # Write to files
    write_to_file(train_data, 'train.txt')
    write_to_file(val_data, 'val.txt')
    write_to_file(test_data, 'test.txt')
    
    # Print statistics
    print(f"Total examples: {len(examples)}")
    print(f"Train set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")
    print(f"Test set size: {len(test_data)}")
    
    # Print sample examples
    print("\nSample training examples:")
    for example in train_data[:3]:
        print(example)

if __name__ == '__main__':
    main() 