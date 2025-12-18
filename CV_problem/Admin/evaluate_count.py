from ultralytics import YOLO
import pathlib
import json
import glob
from tqdm import tqdm
import numpy as np

def count_ground_truth(label_path):
    """Counts objects in a YOLO label file."""
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
            return len(lines)
    except FileNotFoundError:
        return 0

def main():
    SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
    
    # Load trained model
    # Assuming standard path from train_yolo.py
    MODEL_PATH = SCRIPT_DIR / "runs" / "bird_count_exp" / "weights" / "best.pt"
    
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}. Run training first.")
        # Fallback to yolo11n.pt just to test script logic if needed, but better to fail.
        return

    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    
    # Validation Images
    VAL_IMAGES_DIR = SCRIPT_DIR / "datasets" / "images" / "val"
    VAL_LABELS_DIR = SCRIPT_DIR / "datasets" / "labels" / "val"
    
    image_paths = list(VAL_IMAGES_DIR.glob("*.jpg"))
    print(f"Evaluating on {len(image_paths)} images...")
    
    true_counts = []
    pred_counts = []
    errors = []
    
    for img_path in tqdm(image_paths):
        # 1. Get Ground Truth Count
        label_path = VAL_LABELS_DIR / f"{img_path.stem}.txt"
        true_count = count_ground_truth(label_path)
        
        # 2. Run Inference
        # conf=0.25 is standard, but for small objects sometimes lower is better or higher if many false positives.
        results = model.predict(str(img_path), conf=0.25, verbose=False)
        
        # 3. Get Predicted Count
        # results[0].boxes is the Boxes object
        pred_count = len(results[0].boxes)
        
        # 4. Record
        true_counts.append(true_count)
        pred_counts.append(pred_count)
        errors.append(pred_count - true_count) # Positive means overcount, Negative means undercount
        
    # Statistics
    true_counts = np.array(true_counts)
    pred_counts = np.array(pred_counts)
    errors = np.array(errors)
    abs_errors = np.abs(errors)
    
    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(errors**2))
    bias = np.mean(errors)
    
    print("\n=== Evaluation Results ===")
    print(f"Total Images: {len(image_paths)}")
    print(f"Total Birds (Truth): {np.sum(true_counts)}")
    print(f"Total Birds (Pred):  {np.sum(pred_counts)}")
    print("--------------------------")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Sq Error (RMSE): {rmse:.2f}")
    print(f"Bias (Mean Error):         {bias:.2f}")
    print("--------------------------")
    
    # Show worst offenders
    # Sort by absolute error
    sorted_indices = np.argsort(abs_errors)[::-1]
    print("\nTop 5 Worst Predictions:")
    for i in sorted_indices[:5]:
        img_name = image_paths[i].name
        print(f"{img_name}: True={true_counts[i]}, Pred={pred_counts[i]}, Diff={errors[i]}")

if __name__ == "__main__":
    main()
