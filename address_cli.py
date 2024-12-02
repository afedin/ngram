"""
Command-line interface for address analysis and similarity search.
"""

import sys
from address_analyzer import Address, AddressMatcher

def print_help():
    print("""
Address Analysis Tool

Usage:
1. Find similar addresses by entering either:
   a) Full address with coordinates:
      latitude,longitude|street_address|locality|region|postcode
      Example: 55.853729,-4.254518|40 Carlton Pl|Glasgow|Lanarkshire|G5 9TS
   
   b) Street address only:
      street_number street_name
      Example: 40 Carlton Pl
   
   c) Partial address with components:
      street_address|locality|region|postcode
      Example: 40 Carlton Pl|Glasgow|Lanarkshire|G5 9TS

2. Commands:
   :help  - Show this help message
   :quit  - Exit the program
   :ex    - Show example addresses

Note: Press Ctrl+C at any time to exit
""")

def print_examples():
    print("""
Example addresses:
1. Full address with coordinates:
   55.853729,-4.254518|40 Carlton Pl|Glasgow|Lanarkshire|G5 9TS

2. Street address only:
   40 Carlton Pl

3. Partial address:
   40 Carlton Pl|Glasgow|Lanarkshire|G5 9TS

4. Just street name:
   Carlton Pl
""")

def format_similarity_results(similar_addresses):
    """Format similarity results for display."""
    if not similar_addresses:
        return "No similar addresses found."
    
    results = []
    for addr, score in similar_addresses:
        results.append(f"\nSimilarity score: {score:.3f}")
        
        # Format components nicely
        components = []
        if addr.coordinates:
            components.append(f"Coordinates: {addr.coordinates[0]:.6f}, {addr.coordinates[1]:.6f}")
        
        # Group other components by type
        for comp in addr.components:
            if comp.type != "coordinates":
                components.append(f"{comp.type}: {comp.text}")
        
        results.extend(components)
    
    return "\n".join(results)

def main():
    # Load training data
    print("Loading address database...")
    try:
        with open('data/train.txt', 'r') as f:
            training_data = [line.strip() for line in f if line.strip()]
        matcher = AddressMatcher(training_data)
        print(f"Loaded {len(training_data)} addresses")
    except FileNotFoundError:
        print("Error: Could not find training data (data/train.txt)")
        return
    
    print_help()
    
    # Main interaction loop
    while True:
        try:
            query = input("\nEnter address (or command): ").strip()
            
            # Handle commands
            if query.lower() == ':quit':
                break
            elif query.lower() == ':help':
                print_help()
                continue
            elif query.lower() == ':ex':
                print_examples()
                continue
            elif not query:
                continue
            
            # Process address query
            try:
                print("\nAnalyzing query address:")
                query_addr = Address(query)
                print(query_addr)
                
                print("\nFinding similar addresses...")
                similar = matcher.find_similar(query, n=5)
                print(format_similarity_results(similar))
                
            except Exception as e:
                print(f"Error processing address: {str(e)}")
                print("Please check the address format and try again.")
                print("Use :help to see the correct format.")
        
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {str(e)}")
    
    print("Goodbye!")

if __name__ == '__main__':
    main() 