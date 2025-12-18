from ultralytics import YOLO
import pathlib
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import shutil

def get_ground_truth_boxes(label_path, img_w, img_h):
    """Parses YOLO label file and returns boxes in [x1, y1, x2, y2] format."""
    boxes = []
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # class x_c y_c w h
                    x_c = float(parts[1])
                    y_c = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    
                    x1 = int((x_c - w/2) * img_w)
                    y1 = int((y_c - h/2) * img_h)
                    x2 = int((x_c + w/2) * img_w)
                    y2 = int((y_c + h/2) * img_h)
                    
                    boxes.append([x1, y1, x2, y2])
    except FileNotFoundError:
        pass
    return boxes

def draw_boxes(img, boxes, color=(0, 255, 0), thickness=2, label="Bird"):
    """Draws boxes on the image."""
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img

def main():
    SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
    
    # Paths
    MODEL_PATH = SCRIPT_DIR / "runs" / "bird_count_exp" / "weights" / "best.pt"
    VAL_IMAGES_DIR = SCRIPT_DIR / "datasets" / "images" / "val"
    VAL_LABELS_DIR = SCRIPT_DIR / "datasets" / "labels" / "val"
    OUTPUT_DIR = SCRIPT_DIR / "evaluation_results"
    
    # Setup Output
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir()
    
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    print("Loading model...")
    model = YOLO(str(MODEL_PATH))
    
    image_paths = list(VAL_IMAGES_DIR.glob("*.jpg"))
    print(f"Visualizing {len(image_paths)} images...")
    
    true_counts = []
    pred_counts = []
    
    for img_path in tqdm(image_paths):
        # Read Image
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        h, w, _ = img_bgr.shape
        
        # 1. Ground Truth
        label_path = VAL_LABELS_DIR / f"{img_path.stem}.txt"
        gt_boxes = get_ground_truth_boxes(label_path, w, h)
        true_counts.append(len(gt_boxes))
        
        # Draw GT on a copy
        img_gt = img_bgr.copy()
        draw_boxes(img_gt, gt_boxes, color=(0, 255, 0), thickness=3) # Green for GT
        cv2.putText(img_gt, f"Ground Truth: {len(gt_boxes)}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        # 2. Prediction
        results = model.predict(str(img_path), conf=0.25, verbose=False)
        pred_boxes = []
        for box in results[0].boxes:
            coords = box.xyxy[0].cpu().numpy().astype(int) # x1, y1, x2, y2
            pred_boxes.append(coords)
            
        pred_counts.append(len(pred_boxes))
        
        # Draw Pred on a copy
        img_pred = img_bgr.copy()
        draw_boxes(img_pred, pred_boxes, color=(0, 0, 255), thickness=3) # Red for Pred
        cv2.putText(img_pred, f"Prediction: {len(pred_boxes)}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        
        # 3. Combine Side-by-Side
        # Resize for display if huge
        scale = 0.5
        dim = (int(w * scale), int(h * scale))
        img_gt_small = cv2.resize(img_gt, dim)
        img_pred_small = cv2.resize(img_pred, dim)
        
        combined = np.hstack((img_gt_small, img_pred_small))
        
        # Save
        out_path = OUTPUT_DIR / f"vis_{img_path.name}"
        cv2.imwrite(str(out_path), combined)
        
    # Scatter Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(true_counts, pred_counts, alpha=0.7)
    
    # Perfect prediction line
    max_val = max(max(true_counts, default=0), max(pred_counts, default=0))
    plt.plot([0, max_val], [0, max_val], 'r--', label='Ideal')
    
    plt.xlabel("True Count")
    plt.ylabel("Predicted Count")
    plt.title("Bird Counting Performance")
    plt.legend()
    plt.grid(True)
    
    plot_path = OUTPUT_DIR / "count_scatter.png"
    plt.savefig(plot_path)
    print(f"Comparison images and plot saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
