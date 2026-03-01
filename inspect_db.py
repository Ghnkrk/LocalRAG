"""
Inspect Database Script
=======================
Check the stored points and their metadata in Qdrant.
"""

from qdrant_client import QdrantClient
from pathlib import Path

def inspect_collection(collection_name: str, qdrant_path: str = "./qdrant_data"):
    client = QdrantClient(path=qdrant_path)
    
    # Get collection info
    info = client.get_collection(collection_name)
    print(f"Collection: {collection_name}")
    print(f"Points count: {info.points_count}")
    print("-" * 50)
    
    # Scroll through some points
    points, _ = client.scroll(
        collection_name=collection_name,
        limit=2,
        with_payload=True,
        with_vectors=False
    )
    
    for i, point in enumerate(points):
        print(f"\nPoint {i+1} (ID: {point.id})")
        print("Payload/Metadata:")
        for key, value in point.payload.items():
            if key == "text":
                print(f"  {key}: {value[:100]}...")
            else:
                print(f"  {key}: {value}")

if __name__ == "__main__":
    import sys
    collection = sys.argv[1] if len(sys.argv) > 1 else "test_docs"
    inspect_collection(collection)
