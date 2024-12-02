# Geographic Address Matching System

A Python-based system for analyzing, matching, and finding similar geographic addresses. The system uses advanced text similarity algorithms and geographic distance calculations to find the most relevant matches for any given address query.

## Features

- **Flexible Address Input**:
  - Full address with coordinates: `latitude,longitude|street_address|locality|region|postcode`
  - Street address only: `street_number street_name`
  - Partial address with components: `street_address|locality|region|postcode`
  - Support for comma-separated addresses: `street_number street_name, locality`

- **Smart Address Parsing**:
  - Automatic extraction of address components
  - Support for unit/apartment numbers
  - Coordinate detection and normalization
  - Handling of various address formats

- **Intelligent Matching**:
  - Geographic distance-based matching using coordinates
  - Text similarity using combined Jaccard and sequence matching
  - Component-wise comparison with weighted scoring
  - Duplicate removal in search results

- **Address Components**:
  - Coordinates (latitude, longitude)
  - House number
  - Unit/apartment number
  - Street name
  - Locality
  - Region
  - Postcode

## Installation

1. Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Unix/macOS
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

Run the interactive CLI:

```bash
python3 address_cli.py
```

Available commands:
- `:help` - Show help message
- `:quit` - Exit the program
- `:ex` - Show example addresses

### Example Queries

1. Full address with coordinates:

```
55.853729,-4.254518|40 Carlton Pl|Glasgow|Lanarkshire|G5 9TS
```

2. Street address only:

```
40 Carlton Pl
```

3. Address with locality:

```
40 Carlton Pl, Glasgow
```

4. Address with unit:

```
Unit 7, 23 Westminster Terrace
```

### Data Format

The system expects training data in the following format:

```
latitude,longitude|street_address|locality|region|postcode
```

Training data should be stored in:
- `data/train.txt` - Training dataset
- `data/val.txt` - Validation dataset
- `data/test.txt` - Test dataset

## Similarity Scoring

The system uses a weighted scoring system for address matching:
- Coordinates: 15%
- House number: 20%
- Unit/apartment: 10%
- Street name: 30%
- Locality: 15%
- Region: 5%
- Postcode: 5%

Distance calculations for coordinates use the Haversine formula to account for Earth's curvature.

## Project Structure

```
ngram/
├── data/
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
├── address_analyzer.py  # Core address parsing and matching logic
├── address_cli.py      # Interactive command-line interface
├── requirements.txt    # Project dependencies
└── README.md          # This file
```

## Dependencies

- Python 3.6+
- NumPy - For numerical operations and distance calculations
- Regular expressions (built-in) - For address parsing
- difflib (built-in) - For text similarity calculations

## Notes

- Coordinates are normalized to 3 decimal places for consistency
- Addresses are case-insensitive during matching
- The system removes duplicates from search results based on core address components
- Geographic distance is normalized using a 5km scale factor
- Text similarity uses a combination of word-based (Jaccard) and character-based (sequence) matching
