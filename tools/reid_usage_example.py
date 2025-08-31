#!/usr/bin/env python3
"""
Simple Usage Example: RT-DETRv3 Re-ID Embeddings
================================================

This example shows how to quickly generate and use Re-ID embeddings
from RT-DETRv3 for object tracking applications.
"""

import numpy as np
from reid_embeddings import ReIDEmbeddingGenerator

# Example 1: Basic usage
def basic_example():
    print("🚀 Basic Re-ID Embedding Example")

    # Initialize the generator
    generator = ReIDEmbeddingGenerator(
        model_path="output/rtdetrv3_r18vd_6x_backbone.onnx",
        debug=True
    )

    # Process an image
    embeddings = generator.process_image(
        image_path="demo/demo.jpg",
        conf_threshold=0.5,
        output_dir="output/reid"
    )

    print(f"✅ Generated {len(embeddings)} Re-ID embeddings")

    # Use embeddings for similarity comparison
    if len(embeddings) >= 2:
        det1, emb1 = embeddings[0]
        det2, emb2 = embeddings[1]

        # Cosine similarity
        similarity = np.dot(emb1, emb2)
        distance = 1.0 - similarity

        print(f"Similarity between detection 1 and 2: {similarity:.3f}")
        print(f"Distance: {distance:.3f}")

# Example 2: BoT-SORT integration
def botsort_example():
    print("\n🔗 BoT-SORT Integration Example")

    # Simulate BoT-SORT tracking workflow
    class SimpleTracker:
        def __init__(self, distance_threshold=0.3):
            self.tracks = []
            self.next_id = 1
            self.distance_threshold = distance_threshold

        def update(self, detections_with_embeddings):
            """Update tracks with new detections"""
            new_tracks = []

            for detection_info, embedding in detections_with_embeddings:
                best_track = None
                best_distance = float('inf')

                # Find best matching track
                for track in self.tracks:
                    distance = 1.0 - np.dot(track['embedding'], embedding)
                    if distance < self.distance_threshold and distance < best_distance:
                        best_distance = distance
                        best_track = track

                if best_track:
                    # Update existing track
                    best_track['embedding'] = embedding  # Update embedding
                    best_track['bbox'] = detection_info['original_bbox']
                    best_track['confidence'] = detection_info['confidence']
                    new_tracks.append(best_track)
                else:
                    # Create new track
                    new_track = {
                        'id': self.next_id,
                        'bbox': detection_info['original_bbox'],
                        'confidence': detection_info['confidence'],
                        'class_id': detection_info['class_id'],
                        'embedding': embedding
                    }
                    new_tracks.append(new_track)
                    self.next_id += 1

            self.tracks = new_tracks
            return self.tracks

    # Initialize tracker and generator
    tracker = SimpleTracker(distance_threshold=0.3)
    generator = ReIDEmbeddingGenerator(
        model_path="output/rtdetrv3_r18vd_6x_backbone.onnx",
        debug=False
    )

    # Process frame
    embeddings = generator.process_image(
        image_path="demo/demo.jpg",
        conf_threshold=0.5,
        output_dir="output/reid"
    )

    # Update tracker
    tracks = tracker.update(embeddings)

    print(f"Active tracks: {len(tracks)}")
    for track in tracks:
        class_name = "person" if track['class_id'] == 0 else f"class_{track['class_id']}"
        print(f"  Track {track['id']}: {class_name} (conf={track['confidence']:.3f})")

if __name__ == "__main__":
    # Run examples
    basic_example()
    botsort_example()

    print("\n✅ Examples completed!")
    print("\n📝 Next steps:")
    print("   1. Integrate with your tracking system")
    print("   2. Adjust distance thresholds based on your data")
    print("   3. Consider temporal smoothing for video sequences")
    print("   4. Optimize for your deployment environment")
