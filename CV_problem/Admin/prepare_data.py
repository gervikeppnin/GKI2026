import pathlib
import json
import shutil
import random
import cv2
import yaml
from tqdm import tqdm

def setup_directories(base_path):
    """Creates the necessary directory structure for YOLO."""
    dirs = [
        base_path / "datasets/images/train",
        base_path / "datasets/images/val",
        base_path / "datasets/labels/train",
        base_path / "datasets/labels/val",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    return base_path / "datasets"

def get_image_dimensions(image_path):
    """Gets image width and height."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    h, w, _ = img.shape
    return w, h

def convert_to_yolo_format(points, img_w, img_h, box_size=50):
    """Converts a list of points to YOLO format bounding boxes.
    
    Args:
        points: List of dicts {'x': ..., 'y': ...}
        img_w: Image width
        img_h: Image height
        box_size: Size of the bounding box in pixels (square)
    
    Returns:
        List of strings: "0 x_center y_center width height"
    """
    yolo_lines = []
    
    # Normalize box size
    dw = 1.0 / img_w
    dh = 1.0 / img_h
    
    w_norm = box_size * dw
    h_norm = box_size * dh
    
    for pt in points:
        x = pt['x']
        y = pt['y']
        
        # YOLO format uses normalized center coordinates
        # The point is assumed to be the center of the bird
        x_norm = x * dw
        y_norm = y * dh
        
        # Clamp values to be within [0, 1] (though YOLO handles slight overshoots, best to be safe)
        x_norm = max(0.0, min(1.0, x_norm))
        y_norm = max(0.0, min(1.0, y_norm))
        
        # Format: class x_center y_center width height
        # Class is always 0 (merged)
        line = f"0 {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}"
        yolo_lines.append(line)
        
    return yolo_lines

def main():
    # Configuration
    SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
    # Looking for Súla in CV_problem/Súla. 
    # Since this script is in CV_problem/Admin, we go ../Súla
    SOURCE_DIR = (SCRIPT_DIR / "../Súla").resolve()
    
    # Create datasets folder in the current directory (CV_problem)
    DATASET_ROOT = setup_directories(SCRIPT_DIR)
    
    print(f"Source Directory: {SOURCE_DIR}")
    print(f"Dataset Root: {DATASET_ROOT}")
    
    # Find all file pairs with smarter matching
    all_files = list(SOURCE_DIR.glob("**/*"))
    
    jpgs = {}
    for f in all_files:
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            # Normalize stem: remove '_edited' if present
            stem = f.stem.replace("_edited", "")
            jpgs[stem] = f
            
    pnts = {}
    for f in all_files:
        if f.suffix.lower() == '.pnt':
            pnts[f.stem] = f
    
    common_stems = sorted(list(set(jpgs.keys()) & set(pnts.keys())))
    print(f"Found {len(common_stems)} matched image/annotation pairs.")
    
    if not common_stems:
        print("No matches found. Exiting.")
        return

    # Shuffle and split
    random.seed(42)
    random.shuffle(common_stems)
    
    split_idx = int(len(common_stems) * 0.8)
    train_stems = common_stems[:split_idx]
    val_stems = common_stems[split_idx:]
    
    print(f"Training set: {len(train_stems)} images")
    print(f"Validation set: {len(val_stems)} images")
    
    def process_split(stems, split_name):
        print(f"Processing {split_name} set...")
        for stem in tqdm(stems):
            img_path = jpgs[stem]
            pnt_path = pnts[stem]
            
            # Read Image Dims
            dims = get_image_dimensions(img_path)
            if dims is None:
                print(f"Warning: Could not read image {img_path}. Skipping.")
                continue
            w, h = dims
            
            # Read Labels
            try:
                with open(pnt_path, 'r') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error reading {pnt_path}: {e}")
                continue
            
            # Extract Points
            # Structure: data['points'][filename_key][class_name] -> list of points
            points_map = data.get("points", {})
            
            # Try to find the correct key for this image
            # The pnt file structure has the filename as a key
            # We look for a key that contains our stem or just take the first one if it's the only one
            image_key = None
            for k in points_map.keys():
                if stem in k: # loose match
                    image_key = k
                    break
            
            if not image_key and len(points_map) == 1:
                image_key = list(points_map.keys())[0]
            
            if not image_key:
                print(f"Warning: Could not find key for {stem} in {pnt_path}. Keys: {list(points_map.keys())}")
                continue
                
            image_points = points_map[image_key]
            
            # Collect ALL points regardless of class (Merge Logic)
            all_bird_points = []
            for class_name, pts in image_points.items():
                all_bird_points.extend(pts)
            
            # Convert to YOLO
            yolo_labels = convert_to_yolo_format(all_bird_points, w, h, box_size=50)
            
            # Copy Image
            dest_img_path = DATASET_ROOT / "images" / split_name / f"{stem}.jpg"
            shutil.copy2(img_path, dest_img_path)
            
            # Write Label File
            dest_label_path = DATASET_ROOT / "labels" / split_name / f"{stem}.txt"
            with open(dest_label_path, 'w') as f:
                f.write("\n".join(yolo_labels))
                
    process_split(train_stems, "train")
    process_split(val_stems, "val")
    
    # Create data.yaml
    data_yaml_content = f"""
path: {(DATASET_ROOT).resolve()} # dataset root dir
train: images/train # train images (relative to 'path')
val: images/val # val images (relative to 'path')

names:
  0: bird
"""
    with open(DATASET_ROOT / "data.yaml", 'w') as f:
        f.write(data_yaml_content)
        
    print(f"Dataset preparation complete. Saved to {DATASET_ROOT}")
    print(f"data.yaml created at {DATASET_ROOT / 'data.yaml'}")

if __name__ == "__main__":
    main()
