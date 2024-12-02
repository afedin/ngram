"""
Address analyzer and similarity search module with improved parsing and matching.
"""

import re
from typing import Dict, List, Tuple, Optional
from difflib import SequenceMatcher
import numpy as np

class AddressComponent:
    def __init__(self, text: str, component_type: str):
        self.text = text.lower()  # normalize to lowercase
        self.type = component_type
        
    def __str__(self):
        return f"{self.type}: {self.text}"

class Address:
    def __init__(self, raw_text: str):
        self.raw_text = raw_text
        self.components = self._parse_components()
        self.coordinates = self._extract_coordinates()
        
    @staticmethod
    def _extract_coords_from_text(text: str) -> Optional[Tuple[str, str]]:
        """Extract coordinates from text if present."""
        coord_match = re.match(r'^(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*(.*)$', text)
        if coord_match:
            lat, lon, remaining = coord_match.groups()
            return (lat, lon), remaining.strip()
        return None, text
        
    def _parse_components(self) -> List[AddressComponent]:
        """Parse address into components."""
        components = []
        
        # Handle different input formats
        if '|' in self.raw_text:
            parts = [p.strip() for p in self.raw_text.split('|')]
        else:
            # If no pipes, treat as street address
            parts = [None, self.raw_text]
        
        # Process each part
        for i, part in enumerate(parts):
            if not part:
                continue
            
            # First check for explicit coordinates (lat,lon format)
            coord_match = re.match(r'^(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)$', part)
            if coord_match:
                lat, lon = coord_match.groups()
                components.append(AddressComponent(f"{lat},{lon}", "coordinates"))
                continue
            
            # Process street address
            if i == 0 or i == 1:
                # Remove any trailing commas and clean up
                clean_part = re.sub(r',\s*$', '', part).strip()
                
                # Check for coordinates at the start of street field
                coords, remaining_text = self._extract_coords_from_text(clean_part)
                if coords:
                    lat, lon = coords
                    components.append(AddressComponent(f"{lat},{lon}", "coordinates"))
                    clean_part = remaining_text
                
                # Extract unit/apartment number if present
                unit_match = re.search(r'\b(unit|flat|apt\.?|apartment|suite)\s*[-#]?\s*(\d+[a-z]?)\b', clean_part, re.IGNORECASE)
                if unit_match:
                    unit_type, unit_num = unit_match.groups()
                    components.append(AddressComponent(f"{unit_type} {unit_num}", "unit"))
                    clean_part = clean_part.replace(unit_match.group(0), '').strip()
                
                # Split remaining address into number and street
                # Handle cases with comma-separated parts
                address_parts = [p.strip() for p in clean_part.split(',')]
                main_address = address_parts[0]
                
                # Split into house number and street name
                match = re.match(r'^(\d+[a-zA-Z-]*)\s+(.+)$', main_address)
                if match:
                    number, street = match.groups()
                    components.append(AddressComponent(number, "house_number"))
                    components.append(AddressComponent(street, "street"))
                else:
                    components.append(AddressComponent(main_address, "street"))
                
                # Add additional address parts as locality if present
                if len(address_parts) > 1:
                    components.append(AddressComponent(address_parts[1], "locality"))
                
            elif i == 2:  # Locality
                components.append(AddressComponent(part, "locality"))
            elif i == 3:  # Region
                components.append(AddressComponent(part, "region"))
            elif i == 4:  # Postcode
                components.append(AddressComponent(part, "postcode"))
        
        return components
    
    def _extract_coordinates(self) -> Optional[Tuple[float, float]]:
        """Extract coordinates from address."""
        for comp in self.components:
            if comp.type == "coordinates":
                try:
                    lat, lon = map(float, comp.text.split(','))
                    return (lat, lon)
                except ValueError:
                    pass
        return None
    
    def __str__(self):
        result = []
        if self.coordinates:
            result.append(f"Coordinates: {self.coordinates[0]:.6f}, {self.coordinates[1]:.6f}")
        
        # Group components by type for better display
        grouped = {
            "house_number": [],
            "unit": [],
            "street": [],
            "locality": [],
            "region": [],
            "postcode": []
        }
        
        for comp in self.components:
            if comp.type != "coordinates":
                grouped[comp.type].append(comp.text)
        
        # Format address components
        address_parts = []
        if grouped["house_number"]:
            address_parts.append(f"house_number: {grouped['house_number'][0]}")
        if grouped["unit"]:
            address_parts.append(f"unit: {grouped['unit'][0]}")
        if grouped["street"]:
            address_parts.append(f"street: {grouped['street'][0]}")
        if grouped["locality"]:
            address_parts.append(f"locality: {grouped['locality'][0]}")
        if grouped["region"]:
            address_parts.append(f"region: {grouped['region'][0]}")
        if grouped["postcode"]:
            address_parts.append(f"postcode: {grouped['postcode'][0]}")
        
        result.extend(address_parts)
        return "\n".join(result)

class AddressMatcher:
    def __init__(self, training_data: List[str]):
        self.addresses = []
        for addr_str in training_data:
            # Pre-process training data to handle space-separated coordinates
            if '|' in addr_str:
                parts = addr_str.split('|')
                coords_match = re.match(r'^(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*(.*)$', parts[1])
                if coords_match:
                    lat, lon, remaining = coords_match.groups()
                    parts[1] = remaining.strip()
                    parts.insert(0, f"{lat},{lon}")
                    addr_str = '|'.join(parts)
            self.addresses.append(Address(addr_str))
        
    def find_similar(self, query: str, n: int = 5) -> List[Tuple[Address, float]]:
        """Find n most similar addresses to the query."""
        query_addr = Address(query)
        scores = []
        
        for addr in self.addresses:
            score = self._calculate_similarity(query_addr, addr)
            scores.append((addr, score))
        
        # Sort by similarity score in descending order
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Remove duplicates based on core address components
        unique_results = []
        seen_addresses = set()
        
        for addr, score in scores:
            # Create a key from core components
            key_parts = []
            for comp in addr.components:
                if comp.type in ['house_number', 'street', 'postcode']:
                    key_parts.append(f"{comp.type}:{comp.text}")
            key = '|'.join(key_parts)
            
            if key not in seen_addresses:
                seen_addresses.add(key)
                unique_results.append((addr, score))
                if len(unique_results) == n:
                    break
        
        return unique_results
    
    def _calculate_similarity(self, addr1: Address, addr2: Address) -> float:
        """Calculate similarity score between two addresses."""
        score = 0.0
        weights = {
            "coordinates": 0.15,
            "house_number": 0.20,
            "unit": 0.10,
            "street": 0.30,
            "locality": 0.15,
            "region": 0.05,
            "postcode": 0.05
        }
        
        # Get components by type for both addresses
        components1 = {comp.type: comp.text for comp in addr1.components}
        components2 = {comp.type: comp.text for comp in addr2.components}
        
        # Compare each component type
        for comp_type, weight in weights.items():
            text1 = components1.get(comp_type)
            text2 = components2.get(comp_type)
            
            if text1 and text2:
                if comp_type == "coordinates" and addr1.coordinates and addr2.coordinates:
                    # Calculate geographic distance
                    dist = self._haversine_distance(addr1.coordinates, addr2.coordinates)
                    # Convert distance to similarity score (closer = higher score)
                    sim = 1.0 / (1.0 + (dist / 5.0))  # Normalize distance impact
                else:
                    # Text similarity with improved matching
                    sim = self._text_similarity(text1, text2)
                score += weight * sim
            elif comp_type in ["house_number", "street", "postcode"] and bool(text1) != bool(text2):
                # Penalize missing core components
                score -= weight * 0.2
        
        return max(0, score)  # Ensure non-negative score
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity with improved matching."""
        # Direct match
        if text1 == text2:
            return 1.0
            
        # Split into words and find best matches
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Calculate Jaccard similarity for word sets
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        if union == 0:
            return 0.0
            
        # Combine Jaccard similarity with sequence matching
        jaccard = intersection / union
        sequence = SequenceMatcher(None, text1, text2).ratio()
        
        return 0.7 * jaccard + 0.3 * sequence
    
    def _haversine_distance(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """Calculate haversine distance between two coordinates in kilometers."""
        lat1, lon1 = map(np.radians, coord1)
        lat2, lon2 = map(np.radians, coord2)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371  # Earth's radius in kilometers
        
        return c * r

def main():
    # Read training data
    with open('data/train.txt', 'r') as f:
        training_data = [line.strip() for line in f if line.strip()]
    
    # Create matcher
    matcher = AddressMatcher(training_data)
    
    # Example queries
    example_queries = [
        "55.853729,-4.254518|40 Carlton Pl|Glasgow|Lanarkshire|G5 9TS",
        "55.950042,-4.24762|East Blairskeith|Glasgow|Lanarkshire|G64 4AX"
    ]
    
    # Test similarity search
    for query in example_queries:
        print(f"\nQuery address:")
        print(Address(query))
        print("\nSimilar addresses:")
        for addr, score in matcher.find_similar(query):
            print(f"\nSimilarity score: {score:.3f}")
            print(addr)

if __name__ == '__main__':
    main() 