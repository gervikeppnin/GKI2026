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
    # Train the model
    # imgsz=1024 matches our tile size, ensuring no downscaling occurs
    results = model.train(
        data=str(DATA_YAML), 
        epochs=4, 
        project=str(SCRIPT_DIR / "runs"), 
        name="bird_count_exp",
        exist_ok=True, 
        device='mps', # Switch to CPU to debug MPS error
        batch=-1,      
        imgsz=1024,   
        workers=8,    # Disable workers for stability
        cache=True,
    )
    
    # Export or just leave it. The best model will be in runs/bird_count_exp/weights/best.pt
    print("Training complete.")

if __name__ == "__main__":
    main()
