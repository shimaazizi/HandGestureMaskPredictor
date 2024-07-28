import os
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split



def load_image_and_mask(image_path, mask_path, target_size=(128, 128)):
    image = Image.open(image_path)
    mask = Image.open(mask_path)

    # Correct orientation using EXIF data
    image = ImageOps.exif_transpose(image)
    mask = ImageOps.exif_transpose(mask)

    # Verify dimensions
    assert image.size == mask.size, f"Image and mask dimensions do not match: {image.size} vs {mask.size}"

    # Resize images and masks
    image = image.resize(target_size)
    mask = mask.resize(target_size)

    # Convert to numpy arrays
    image_array = np.array(image) / 255.0
    mask_array = np.array(mask)

    # Handle different mask formats
    if len(mask_array.shape) == 3:  # RGB mask
        mask_array = np.dot(mask_array[..., :3], [0.2989, 0.5870, 0.1140])

    mask_indices = (mask_array > 128).astype(np.int64)

    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float()
    mask_tensor = torch.from_numpy(mask_indices).long()

    return image_tensor, mask_tensor

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, target_size=(128, 128)):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.target_size = target_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        image, mask = load_image_and_mask(image_path, mask_path, self.target_size)
        mask = mask.squeeze()
        return image, mask

def dataset_split(image_paths, mask_paths, test_split=0.2, val_split=0.2):
    class_labels = [os.path.basename(os.path.dirname(file)) for file in image_paths]
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        image_paths, mask_paths,
        test_size=test_split,
        random_state=42,
        stratify=class_labels
    )
    train_val_class_labels = [os.path.basename(os.path.dirname(file)) for file in x_train_val]
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val,
        test_size=val_split,
        random_state=42,
        stratify=train_val_class_labels
    )
    print(f'Train samples: {len(x_train)}')
    print(f'Validation samples: {len(x_val)}')
    print(f'Test samples: {len(x_test)}')
    return x_train, x_val, x_test, y_train, y_val, y_test


def create_dataloaders(dataset_path, mask_path, batch_size=32):
    #dataset_path = '/content/drive/MyDrive/Dataset'
    #mask_path = '/content/drive/MyDrive/New_Mask'

    image_paths = []
    mask_paths = []

    for class_name in ['Fist', 'OpenPalm', 'PeaceSign', 'ThumbsUp']:
        class_images = sorted([os.path.join(dataset_path, class_name, f) for f in os.listdir(os.path.join(dataset_path, class_name)) if f.endswith('.jpg')])
        class_masks = sorted([os.path.join(mask_path, f"{class_name}_Mask", f) for f in os.listdir(os.path.join(mask_path, f"{class_name}_Mask")) if f.endswith('.jpg')])

        # Ensure we only use images that have corresponding masks
        common_files = set([os.path.basename(f) for f in class_images]) & set([os.path.basename(f) for f in class_masks])

        image_paths.extend([f for f in class_images if os.path.basename(f) in common_files])
        mask_paths.extend([f for f in class_masks if os.path.basename(f) in common_files])
    
        # Split the dataset
    x_train, x_val, x_test, y_train, y_val, y_test = dataset_split(image_paths, mask_paths)

    # Create datasets for each split
    train_dataset = SegmentationDataset(x_train, y_train)
    val_dataset = SegmentationDataset(x_val, y_val)
    test_dataset = SegmentationDataset(x_test, y_test)
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)


    return train_loader, val_loader, test_loader
