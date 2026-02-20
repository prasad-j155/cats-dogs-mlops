import os
from PIL import Image
from sklearn.model_selection import train_test_split
import yaml

# Load parameters
with open("params.yaml") as f:
    params = yaml.safe_load(f)
# Define our static paths and variables
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
IMG_SIZE = params["prepare"]["img_size"]
#IMG_SIZE = 128  # Neural networks need uniform input sizes

def process_and_split():
    categories = ["Cat", "Dog"]
    
    # Locate the folders (handles different download methods)
    search_paths = [RAW_DIR, os.path.join(RAW_DIR, "PetImages")]
    base_path = next((path for path in search_paths if os.path.exists(os.path.join(path, "Cat"))), None)
            
    if not base_path:
        print("Error: Could not find 'Cat' and 'Dog' folders in data/raw.")
        return

    print(f"Found image folders in: {base_path}")

    # 1. Create the output folder structure: data/processed/train/Cat, etc.
    for split in ['train', 'val']:
        for category in categories:
            os.makedirs(os.path.join(PROCESSED_DIR, split, category), exist_ok=True)

    # 2. Process the images
    for category in categories:
        folder_path = os.path.join(base_path, category)
        # Grab only image files
        images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Split 80% for training, 20% for validation
        train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)
        
        def process_images(image_list, split_name):
            valid_count = 0
            for img_name in image_list:
                try:
                    img_path = os.path.join(folder_path, img_name)
                    # Open, convert to standard RGB, and resize
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((IMG_SIZE, IMG_SIZE))
                    
                    out_path = os.path.join(PROCESSED_DIR, split_name, category, img_name)
                    img.save(out_path)
                    valid_count += 1
                except Exception:
                    # Silently skip corrupted files
                    pass
            print(f"Saved {valid_count} valid {category} images to {split_name} set.")

        process_images(train_imgs, 'train')
        process_images(val_imgs, 'val')

if __name__ == "__main__":
    print("Starting data preparation...")
    process_and_split()
    print("Data preparation complete!")