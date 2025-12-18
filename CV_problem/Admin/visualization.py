import pathlib
import json
import matplotlib.pyplot as plt

def get_data_folder():
    """Resolves the data folder relative to this script."""
    script_dir = pathlib.Path(__file__).parent.resolve()
    return (script_dir / "../Súla").resolve()

def load_data(folder_path):
    """Finds all jpg and pnt files in the folder."""
    all_files = list(folder_path.glob("**/*"))
    jpgs = {f.stem: f for f in all_files if f.suffix.lower() in ['.jpg', '.jpeg']}
    pnts = {f.stem: f for f in all_files if f.suffix.lower() == '.pnt'}
    return jpgs, pnts

def visualize_sample(image_path, pnt_path):
    """Visualizes the image and overlays the points from the pnt file."""
    print(f"Visualizing: {image_path.name}")
    
    # Load Image
    try:
        image = plt.imread(image_path)
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return

    # Load Annotation
    try:
        with open(pnt_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading annotation {pnt_path}: {e}")
        return

    # Setup Plot
    # Setup Plot
    height, width = image.shape[:2]
    dpi = 100
    figsize = (width / dpi, height / dpi)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(image)
    
    # Extract and Plot Points
    # Structure: data['points'][filename_key][class_name] = [{'x':..., 'y':...}, ...]
    points_map = data.get("points", {})
    
    # The key in 'points' is usually the filename (often with extension)
    # We try to match loosely or just take the first key if unique logic isn't strictly defined
    # For this specific case, we look for the key corresponding to our image name.
    
    image_key = image_path.name
    if image_key not in points_map:
        # Fallback: sometimes encoding differs or name mismatch. 
        # If there's only one key, we might use it, but valid lookup is best.
        print(f"Warning: Key '{image_key}' not found in pnt file keys: {list(points_map.keys())}")
        # Attempt to find a partial match or just skip
        # For the sample file provided: "501 S\u00f6lvan\u00f6f Rau\u00f0an\u00fapi_3309.jpg"
        # Let's try to just use the one key if it exists
        if len(points_map) == 1:
            image_key = list(points_map.keys())[0]
            print(f"Using available key: {image_key}")
        else:
            return

    image_points = points_map[image_key]
    colors = ['r', 'b', 'g', 'c', 'm', 'y'] # Simple cycle
    
    for idx, (class_name, points) in enumerate(image_points.items()):
        color = colors[idx % len(colors)]
        xs = [p['x'] for p in points]
        ys = [p['y'] for p in points]
        
        ax.scatter(xs, ys, c=color, s=20, label=class_name)
        print(f"  - Plotted {len(points)} points for class '{class_name}'")

    ax.legend()
    ax.set_title(image_path.name)
    plt.show() # In non-interactive environments this might just block or do nothing visible, but code is correct.

def main(target=None):
    data_folder = get_data_folder()
    print(f"Data folder: {data_folder}")
    
    if not data_folder.exists():
        print("Error: Data folder not found!")
        return

    jpgs, pnts = load_data(data_folder)
    
    # Find common stems
    common_stems = set(jpgs.keys()) & set(pnts.keys())

    # Visualize one example (or specifically the '501' one requested previously)
    if target:
        for stem in common_stems:
            if target in stem:
                visualize_sample(jpgs[stem], pnts[stem])
    else:
        for stem in common_stems:
            visualize_sample(jpgs[stem], pnts[stem])

if __name__ == "__main__":
    main(target=None)





