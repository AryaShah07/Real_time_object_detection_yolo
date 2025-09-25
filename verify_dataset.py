import os
import yaml
from glob import glob

def verify_yolo_dataset(dataset_path):
    errors = []
    all_class_ids = set()

    print(f"\nVerifying YOLO dataset at: {dataset_path}\n")

    # --- Step 1: Load data.yaml ---
    print("Step 1: Checking data.yaml ...")
    yaml_path = os.path.join(dataset_path, "data.yaml")
    if not os.path.exists(yaml_path):
        print("  data.yaml not found!")
        return

    try:
        with open(yaml_path, "r") as f:
            yaml_data = yaml.safe_load(f)
        print("  data.yaml loaded successfully")
    except Exception as e:
        print(f"  Failed to parse data.yaml: {e}")
        return

    # Fix: resolve paths relative to the current script directory
    dataset_dirs = {}
    for key in ["train", "valid"]:
        if key in yaml_data:
            # If path is relative, resolve from dataset_path
            dir_path = os.path.normpath(os.path.join(dataset_path, os.path.relpath(yaml_data[key], "..")))
            dataset_dirs[key] = dir_path
            if os.path.exists(dir_path):
                print(f"   ✔ {key} path found: {dir_path}")
            else:
                errors.append(f"data.yaml: {key} path '{yaml_data[key]}' not found (resolved to {dir_path})")

    if "nc" not in yaml_data or "names" not in yaml_data:
        errors.append("data.yaml must define both 'nc' and 'names'")
    else:
        nc = yaml_data["nc"]
        names = yaml_data["names"]
        print(f"   → nc = {nc}, names = {names}")
        if len(names) != nc:
            errors.append(f"data.yaml: length of names ({len(names)}) != nc ({nc})")

    # --- Step 2: Verify splits ---
    print("\n Step 2: Checking dataset splits ...")
    for split, img_dir in dataset_dirs.items():
        label_dir = os.path.normpath(os.path.join(os.path.dirname(img_dir), "labels"))

        if not os.path.exists(img_dir):
            errors.append(f"{split}: images directory not found: {img_dir}")
            continue
        if not os.path.exists(label_dir):
            errors.append(f"{split}: labels directory not found: {label_dir}")
            continue

        img_files = sorted(glob(os.path.join(img_dir, "*.*")))
        label_files = sorted(glob(os.path.join(label_dir, "*.txt")))

        print(f"   🔹 {split}: {len(img_files)} images, {len(label_files)} labels")

        if not img_files:
            errors.append(f"{split}: no images found in {img_dir}")
        if not label_files:
            errors.append(f"{split}: no labels found in {label_dir}")

        # --- Check that each image has a matching label ---
        img_basenames = {os.path.splitext(os.path.basename(f))[0] for f in img_files}
        label_basenames = {os.path.splitext(os.path.basename(f))[0] for f in label_files}

        missing_labels = img_basenames - label_basenames
        missing_images = label_basenames - img_basenames

        if not missing_labels and not missing_images:
            print(f"       Image ↔ Label mapping check passed for {split}")
        else:
            print(f"       Image ↔ Label mapping check failed for {split}")
            for m in missing_labels:
                errors.append(f"{split}: missing label file for image '{m}'")
            for m in missing_images:
                errors.append(f"{split}: missing image file for label '{m}'")

    # --- Step 3: Validate labels ---
    print("\n Step 3: Validating label file contents ...")
    label_files_found = False
    total_labels_checked = 0
    total_lines_checked = 0
    for split, img_dir in dataset_dirs.items():
        label_dir = os.path.normpath(os.path.join(os.path.dirname(img_dir), "labels"))
        label_files = sorted(glob(os.path.join(label_dir, "*.txt")))
        if label_files:
            label_files_found = True

        for idx, lf in enumerate(label_files, 1):
            if idx % 500 == 0:
                print(f"     Checked {idx} label files in {split}...")
            with open(lf, "r") as f:
                for i, line in enumerate(f, 1):
                    total_lines_checked += 1
                    parts = line.strip().split()
                    if len(parts) < 5:
                        errors.append(f"{lf} line {i}: Invalid format (need >=5 values)")
                        continue

                    try:
                        nums = [float(x) for x in parts]
                    except ValueError:
                        errors.append(f"{lf} line {i}: Non-numeric values")
                        continue

                    class_id = int(nums[0])
                    all_class_ids.add(class_id)

                    if len(nums) == 5:
                        # YOLO Detection
                        _, x, y, w, h = nums
                        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                            errors.append(f"{lf} line {i}: Detection coords out of range [0,1]")
                    elif len(nums) > 5:
                        # YOLO Segmentation (polygon x,y pairs)
                        polygon = nums[1:]
                        if len(polygon) % 2 != 0:
                            errors.append(f"{lf} line {i}: Polygon has odd number of coords")
                        for val in polygon:
                            if not (0 <= val <= 1):
                                errors.append(f"{lf} line {i}: Segmentation coord {val} out of range [0,1]")
            total_labels_checked += 1

        if label_files:
            print(f"     Finished checking {len(label_files)} label files in {split}.")

    print(f"   → Found class IDs in labels: {sorted(all_class_ids)}")
    print(f"   → Checked {total_labels_checked} label files, {total_lines_checked} label lines.")

    # --- Step 4: Cross-check class IDs with data.yaml ---
    print("\n Step 4: Cross-checking classes with data.yaml ...")
    if not label_files_found:
        print("   No label files found, skipping class ID cross-check.")
    elif all_class_ids:
        max_id = max(all_class_ids)
        print(f"   → Max class_id = {max_id}")
        if "nc" in yaml_data and (max_id + 1) != yaml_data["nc"]:
            errors.append(f"data.yaml: nc={yaml_data['nc']} but found class_id up to {max_id} (should be {max_id+1})")
        if "names" in yaml_data and len(yaml_data["names"]) <= max_id:
            errors.append(f"data.yaml: names has {len(yaml_data['names'])} entries but found class_id {max_id}")
    else:
        print("   No class IDs found in label files.")

    # --- Final Report ---
    print("\n ---Final Report---")
    if errors:
        print("  Dataset Verification Found Issues:")
        for e in errors:
            print("   -", e)
    else:
        print(" ✅ Dataset structure, annotations, and data.yaml are valid!")
        print(f"   Classes found in labels: {sorted(all_class_ids)}")

if __name__ == "__main__":
    dataset_root = os.path.dirname(os.path.abspath(__file__))  
    verify_yolo_dataset(dataset_root)
