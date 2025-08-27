# tools/reduce_coco_val.py
# Utility to create a smaller COCO-style validation set for faster evaluation during model optimization.

# example usage:
# python3 tools/reduce_coco_val.py \
# --input dataset/coco/annotations/instances_val2017.json \
# --output dataset/coco/annotations/simple_val2017.json \
# --num-images 250 --strategy random --seed 42

import json
import argparse
import random
from pathlib import Path

def load_coco(path):
    with open(path, "r") as f:
        return json.load(f)

def save_coco(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)
    print(f"Saved {path} (images={len(obj.get('images', []))}, annotations={len(obj.get('annotations', []))})")

def reduce_coco(input_json, output_json, num_images, seed=0, strategy="first", keep_unused_categories=True):
    data = load_coco(input_json)

    images = data.get("images", [])
    anns = data.get("annotations", [])
    cats = data.get("categories", [])

    if num_images <= 0 or num_images >= len(images):
        print("num_images covers / exceeds dataset size; copying original.")
        save_coco(data, output_json)
        return

    if strategy == "first":
        chosen_images = images[:num_images]
    elif strategy == "random":
        random.seed(seed)
        chosen_images = random.sample(images, num_images)
    else:
        raise ValueError("strategy must be 'first' or 'random'")

    chosen_ids = {im["id"] for im in chosen_images}
    filtered_anns = [a for a in anns if a.get("image_id") in chosen_ids]

    if keep_unused_categories:
        filtered_cats = cats
    else:
        used_cat_ids = {a["category_id"] for a in filtered_anns}
        filtered_cats = [c for c in cats if c["id"] in used_cat_ids]

    out = {
        k: data[k] for k in data.keys() if k not in ["images", "annotations", "categories"]
    }
    out["images"] = chosen_images
    out["annotations"] = filtered_anns
    out["categories"] = filtered_cats

    save_coco(out, output_json)
    print(f"Selected strategy={strategy}, num_images={num_images}, seed={seed}")
    print(f"Original images={len(images)} -> {len(chosen_images)}")
    print(f"Original annotations={len(anns)} -> {len(filtered_anns)}")
    if not keep_unused_categories:
        print(f"Original categories={len(cats)} -> {len(filtered_cats)}")

def main():
    ap = argparse.ArgumentParser(description="Reduce COCO annotation file to a smaller subset.")
    ap.add_argument("--input", required=True, help="Path to original instances_val2017.json")
    ap.add_argument("--output", required=True, help="Path to write reduced JSON (e.g. simple_val2017.json)")
    ap.add_argument("--num-images", type=int, default=200, help="Number of images to keep")
    ap.add_argument("--seed", type=int, default=0, help="Random seed (used if strategy=random)")
    ap.add_argument("--strategy", choices=["first", "random"], default="first", help="Subset selection strategy")
    ap.add_argument("--drop-unused-categories", action="store_true", help="Remove categories not present in subset")
    args = ap.parse_args()

    reduce_coco(
        Path(args.input),
        Path(args.output),
        args.num_images,
        seed=args.seed,
        strategy=args.strategy,
        keep_unused_categories=not args.drop_unused_categories
    )

if __name__ == "__main__":
    main()