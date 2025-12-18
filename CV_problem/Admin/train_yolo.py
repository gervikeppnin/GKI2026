from ultralytics import YOLO
import pathlib

def main():
    # Setup paths
    SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
    # The data.yaml should be in datasets/data.yaml relative to this script or consistent with prepare_data.py
    DATA_YAML = (SCRIPT_DIR / "datasets" / "data.yaml").resolve()
    
    print(f"Training using config: {DATA_YAML}")
    
    # Load a model
    model = YOLO("yolo11n.pt")  # load a pretrained model (nano version)

    # Train the model
    # imgsz=640 is standard. Rectangular training can be useful if aspect ratios vary wildly, but 640 is safe.
    results = model.train(
        data=str(DATA_YAML), 
        epochs=10, 
        project=str(SCRIPT_DIR / "runs"), 
        name="bird_count_exp",
        exist_ok=True, # overwrite existing experiment
        device='mps',
        batch=1,
        workers=2,
    )
    
    # Export or just leave it. The best model will be in runs/bird_count_exp/weights/best.pt
    print("Training complete.")

if __name__ == "__main__":
    main()
