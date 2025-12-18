import pathlib
import json
import shutil
import random
import cv2
import yaml
from tqdm import tqdm
import math

def setup_directories(base_path):
    """Creates the necessary directory structure for YOLO."""
    dirs = [
        base_path / "datasets/images/train",
        base_path / "datasets/images/val",
        base_path / "datasets/labels/train",
        base_path / "datasets/labels/val",
    ]
    for d in dirs:
        if d.exists():
            shutil.rmtree(d) # Clean up previous runs to avoid mixed data
        d.mkdir(parents=True, exist_ok=True)
    return base_path / "datasets"

def get_image_dimensions(image_path):
    """Gets image width and height."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    h, w, _ = img.shape
    return w, h, img

def convert_to_yolo_format(points, img_w, img_h, box_size=50):
    """Converts a list of points to YOLO format bounding boxes."""
    yolo_lines = []
    
    dw = 1.0 / img_w
    dh = 1.0 / img_h
    
    w_norm = box_size * dw
    h_norm = box_size * dh
    
    for pt in points:
        x = pt['x']
        y = pt['y']
        
        x_norm = x * dw
        y_norm = y * dh
        
        x_norm = max(0.0, min(1.0, x_norm))
        y_norm = max(0.0, min(1.0, y_norm))
        
        line = f"0 {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}"
        yolo_lines.append(line)
        
    return yolo_lines

def create_tiles(img, points, stem, tile_size=1024, overlap=0.2):
    """Generates tiles from the image and filters points."""
    h, w, _ = img.shape
    stride = int(tile_size * (1 - overlap))
    
    tiles = []
    
    y = 0
    while y < h:
        y_start = y
        y_end = y + tile_size
        
        # If we go past the end, shift back to fit the last tile exactly
        if y_end > h:
            y_start = max(0, h - tile_size)
            y_end = h
            
        x = 0
        while x < w:
            x_start = x
            x_end = x + tile_size
            
            # If we go past the end, shift back
            if x_end > w:
                x_start = max(0, w - tile_size)
                x_end = w
            
            # Crop
            tile_img = img[y_start:y_end, x_start:x_end]
            real_h, real_w = tile_img.shape[:2]
            
            # Filter Points for this tile
            tile_points = []
            for pt in points:
                px, py = pt['x'], pt['y']
                # Check if point is within this tile window
                if x_start <= px < x_end and y_start <= py < y_end:
                    # Adjust coordinates to be relative to the tile
                    new_pt = {'x': px - x_start, 'y': py - y_start}
                    tile_points.append(new_pt)
            
            # Only keep tiles that have birds? 
            # Ideally we want some negatives too, but usually random tiles will have empty spots. 
            # For now keep all tiles from the image to cover the area.
            
            # Generate Labels
            yolo_labels = convert_to_yolo_format(tile_points, real_w, real_h, box_size=50)
            
            tile_name = f"{stem}_{y_start}_{x_start}"
            tiles.append((tile_img, yolo_labels, tile_name))
            
            if x_end == w:
                break
            x += stride
            
        if y_end == h:
            break
        y += stride
        
    return tiles

def main():
    # Configuration
    SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
    SOURCE_DIR = (SCRIPT_DIR / "../Súla").resolve()
    DATASET_ROOT = setup_directories(SCRIPT_DIR)
    
    print(f"Source Directory: {SOURCE_DIR}")
    print(f"Dataset Root: {DATASET_ROOT}")
    
    all_files = list(SOURCE_DIR.glob("**/*"))
    jpgs = {f.stem.replace("_edited", ""): f for f in all_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']}
    pnts = {f.stem: f for f in all_files if f.suffix.lower() == '.pnt'}
    
    common_stems = sorted(list(set(jpgs.keys()) & set(pnts.keys())))
    print(f"Found {len(common_stems)} matched image/annotation pairs.")
    
    if not common_stems:
        print("No matches found. Exiting.")
        return

    random.seed(42)
    random.shuffle(common_stems)
    
    split_idx = int(len(common_stems) * 0.8)
    train_stems = common_stems[:split_idx]
    val_stems = common_stems[split_idx:]
    
    print(f"Training set: {len(train_stems)} large images")
    print(f"Validation set: {len(val_stems)} large images")
    
    def process_split(stems, split_name):
        print(f"Processing {split_name} set with tiling...")
        total_tiles = 0
        
        for stem in tqdm(stems):
            img_path = jpgs[stem]
            pnt_path = pnts[stem]
            
            # Load Image
            w, h, img = get_image_dimensions(img_path)
            if img is None:
                continue
            
            # Load Labels
            try:
                with open(pnt_path, 'r') as f:
                    data = json.load(f)
            except Exception:
                continue
            
            points_map = data.get("points", {})
            image_key = None
            for k in points_map.keys():
                if stem in k:
                    image_key = k
                    break
            if not image_key and len(points_map) == 1:
                image_key = list(points_map.keys())[0]
            
            if not image_key:
                continue
                
            image_points_dict = points_map[image_key]
            all_bird_points = []
            for pts in image_points_dict.values():
                all_bird_points.extend(pts)
            
            # Generate Tiles
            # For validation, we might ideally want full images for metrics, 
            # but YOLO val expects same format. We can tile val as well.
            tiles = create_tiles(img, all_bird_points, stem, tile_size=1024, overlap=0.2)
            
            for tile_img, yolo_labels, tile_name in tiles:
                # Save Image
                cv2.imwrite(str(DATASET_ROOT / "images" / split_name / f"{tile_name}.jpg"), tile_img)
                
                # Save Label
                with open(DATASET_ROOT / "labels" / split_name / f"{tile_name}.txt", 'w') as f:
                    f.write("\n".join(yolo_labels))
                
                total_tiles += 1
                
        print(f"Generated {total_tiles} tiles for {split_name}.")

    process_split(train_stems, "train")
    process_split(val_stems, "val")
    
    # Create data.yaml
    data_yaml_content = f"""
path: {(DATASET_ROOT).resolve()}
train: images/train
val: images/val

names:
  0: bird
"""
    with open(DATASET_ROOT / "data.yaml", 'w') as f:
        f.write(data_yaml_content)
        
    print(f"Dataset preparation complete.")

if __name__ == "__main__":
    main()
