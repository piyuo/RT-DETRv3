# onnx/validate_pedestrian.py
#!/usr/bin/env python3
"""
Pedestrian-Specific Re-ID Validation
===================================
Focuses on metrics relevant for pedestrian counting applications.
"""

import numpy as np
from reid_embeddings import RobustReIDEmbeddingGenerator

def validate_pedestrian_reid(model_path: str, image_path: str, feature_map_name: str = None):
    """Validate Re-ID pipeline specifically for pedestrian counting."""

    print("🚶 Validating Re-ID pipeline for pedestrian counting...")

    generator = RobustReIDEmbeddingGenerator(
        model_path,
        feature_map_name=feature_map_name,
        debug=True
    )

    embeddings = generator.process_image(image_path, conf_threshold=0.3)

    # Filter only people (class_id = 0)
    people_embeddings = [
        (info, emb) for info, emb in embeddings
        if info['class_id'] == 0
    ]

    print(f"\n📊 Pedestrian Counting Analysis:")
    print(f"   Total detections: {len(embeddings)}")
    print(f"   People detected: {len(people_embeddings)}")
    print(f"   Other objects: {len(embeddings) - len(people_embeddings)}")

    if len(people_embeddings) < 2:
        print("⚠️  Need at least 2 people for separability analysis")
        return

    # Analyze person-to-person separability
    people_vectors = np.array([emb for _, emb in people_embeddings])
    similarity_matrix = np.dot(people_vectors, people_vectors.T)

    # Get all pairwise similarities between different people
    person_similarities = []
    for i in range(len(people_embeddings)):
        for j in range(i + 1, len(people_embeddings)):
            sim = similarity_matrix[i, j]
            person_similarities.append(sim)

    if person_similarities:
        mean_similarity = np.mean(person_similarities)
        std_similarity = np.std(person_similarities)
        min_similarity = np.min(person_similarities)
        max_similarity = np.max(person_similarities)

        print(f"\n🔍 Person-to-Person Similarity Analysis:")
        print(f"   Mean similarity: {mean_similarity:.3f}")
        print(f"   Std deviation: {std_similarity:.3f}")
        print(f"   Min similarity: {min_similarity:.3f}")
        print(f"   Max similarity: {max_similarity:.3f}")

        # For pedestrian counting, we want:
        # - Not too high similarity (can distinguish different people)
        # - Not too low similarity (stable embeddings for same person)

        if mean_similarity > 0.8:
            print("⚠️  Warning: Very high person-to-person similarity")
            print("    This may make it hard to distinguish different people")
        elif mean_similarity < 0.3:
            print("✅ Good: Low person-to-person similarity")
            print("    Different people are well distinguished")
        else:
            print("✅ Reasonable: Moderate person-to-person similarity")

        # Check for very similar people (potential duplicates)
        very_similar_pairs = [(i, j, sim) for idx, sim in enumerate(person_similarities)
                             if sim > 0.9 for i, j in [(idx//len(people_embeddings), idx%len(people_embeddings))]]

        if very_similar_pairs:
            print(f"⚠️  Found {len(very_similar_pairs)} very similar person pairs (sim > 0.9)")
            print("    These might be the same person detected multiple times")

    # Analyze ROI quality specifically for people
    people_roi_areas = [info['roi_shape'][1] * info['roi_shape'][2]
                       for info, _ in people_embeddings]

    small_people_rois = sum(1 for area in people_roi_areas if area <= 9)  # 3x3 or smaller

    print(f"\n📏 People ROI Quality:")
    print(f"   Small people ROIs: {small_people_rois}/{len(people_embeddings)}")

    if small_people_rois > len(people_embeddings) * 0.5:
        print("⚠️  Warning: Many people have small ROIs")
        print("    Consider using C3 (stride 8) for better small person detection")
    else:
        print("✅ Good: Most people have adequate ROI size")

    # Overall assessment for pedestrian counting
    print(f"\n🎯 Pedestrian Counting Assessment:")
    if len(people_embeddings) >= 5:
        print("✅ Good detection count for analysis")
    else:
        print("⚠️  Low detection count - test with busier scenes")

    if mean_similarity < 0.7 and small_people_rois < len(people_embeddings) * 0.6:
        print("✅ Pipeline suitable for pedestrian counting")
    else:
        print("⚠️  Pipeline may have challenges with pedestrian counting")

    return {
        'total_people': len(people_embeddings),
        'person_similarity_stats': {
            'mean': mean_similarity,
            'std': std_similarity,
            'min': min_similarity,
            'max': max_similarity
        },
        'small_rois_percentage': small_people_rois / len(people_embeddings) if people_embeddings else 0
    }

if __name__ == "__main__":
    # Test with your current setup
    model_path = "onnx/backbone/rtdetrv3_r18vd_6x.onnx"
    image_path = "onnx/demo/demo.jpg"
    feature_map_name = "Concat.5"  # C3 level

    validate_pedestrian_reid(model_path, image_path, feature_map_name)