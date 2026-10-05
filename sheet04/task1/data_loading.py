import random
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image

DATA_ROOT    = ""        # Pointing to the directory containing "celeba" subdir
OUTPUT_PATH  = "./sample_images.png"
NUM_SAMPLES  = 5                 # number of pictures to plot
IMG_SIZE     = 64                # resize images


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Converts a (C, H, W) tensor in [0, 1] to a (H, W, C) uint8 array."""
    return (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


class CustomCelebA(torch.utils.data.Dataset):
   
    def __init__(self, root, split="test", transform=None):
        self.root = root
        self.transform = transform
        
        # Map split string to partition integer
        split_map = {"train": 0, "valid": 1, "test": 2}
        split_val = split_map.get(split, 2)
        
        # Verify image directory
        img_dir = os.path.join(root, "celeba", "img_align_celeba")
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
            
        existing_imgs = set(os.listdir(img_dir))
        
        # Read evaluation partition info
        partition_file = os.path.join(root, "celeba", "list_eval_partition.txt")
        split_filenames = []
        with open(partition_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    fn, part = parts
                    if int(part) == split_val and fn in existing_imgs:
                        split_filenames.append(fn)
                        
        self.filenames = split_filenames
        if len(self.filenames) == 0:
            raise ValueError(f"No images found for split '{split}' in {img_dir} matching list_eval_partition.txt")
            
        # Read attribute file
        attr_file = os.path.join(root, "celeba", "list_attr_celeba.txt")
        fn_set = set(self.filenames)
        attr_dict = {}
        
        with open(attr_file, "r") as f:
            _ = int(f.readline().strip())  # Number of images
            self.attr_names = f.readline().strip().split()
            for line in f:
                parts = line.strip().split()
                if len(parts) > 0:
                    fn = parts[0]
                    if fn in fn_set:
                        attrs = [1 if int(x) > 0 else 0 for x in parts[1:]]
                        attr_dict[fn] = torch.tensor(attrs, dtype=torch.long)
                        
        # Align attributes with the filenames list
        self.attr = [attr_dict[fn] for fn in self.filenames]
        
    def __len__(self):
        return len(self.filenames)
        
    def __getitem__(self, idx):
        fn = self.filenames[idx]
        img_path = os.path.join(self.root, "celeba", "img_align_celeba", fn)
        img = Image.open(img_path).convert("RGB")
        
        if self.transform is not None:
            img = self.transform(img)
            
        attr = self.attr[idx]
        return img, attr


class MaskedCelebADataset(torch.utils.data.Dataset):
    """Custom CelebA dataset that resizes images to 64x64 and applies on-the-fly random masking."""
    def __init__(self, root, split="test", transform=None):
        self.celeba = CustomCelebA(
            root=root,
            split=split,
            transform=transform,
        )
        self.img_size = IMG_SIZE

    def __len__(self):
        return len(self.celeba)

    def __getitem__(self, idx):
        # Load image and attribute from the custom CelebA loader
        image, _ = self.celeba[idx]
        
        # Calculate random mask area covering 10-30% of the image area
        total_area = self.img_size * self.img_size
        min_area = int(total_area * 0.10)
        max_area = int(total_area * 0.30)
        
        # Rejection sampling for mask height and width
        while True:
            h = random.randint(1, self.img_size)
            w = random.randint(1, self.img_size)
            if min_area <= h * w <= max_area:
                break
                
        # Random top-left corner coordinates
        top = random.randint(0, self.img_size - h)
        left = random.randint(0, self.img_size - w)
        
        # Create the binary mask tensor (1.0 for masked region, 0.0 otherwise)
        mask = torch.zeros((1, self.img_size, self.img_size), dtype=torch.float32)
        mask[:, top:top+h, left:left+w] = 1.0
        
        # Apply the mask to the image on-the-fly (setting masked region to 0)
        masked_image = image.clone()
        masked_image = masked_image * (1.0 - mask)
        
        return masked_image, mask, image


# Image transform: resize to 64x64 and convert to tensor
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

if __name__ == "__main__":
    print("Loading Masked CelebA Dataset...")
    dataset = MaskedCelebADataset(
        root=DATA_ROOT,
        split="test",
        transform=transform,
    )

    # Initialize DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=NUM_SAMPLES,
        shuffle=True,
    )

    # Fetch a single batch
    print("Fetching a batch of samples from DataLoader...")
    masked_images, masks, original_images = next(iter(dataloader))

    print("Plotting samples...")
    fig, axes = plt.subplots(
        3, NUM_SAMPLES,
        figsize=(NUM_SAMPLES * 2.2, 3 * 2.2),
        gridspec_kw={"hspace": 0.15, "wspace": 0.05},
    )

    for col in range(NUM_SAMPLES):
        # Plot original image
        ax_orig = axes[0, col]
        orig_np = to_numpy(original_images[col])
        ax_orig.imshow(orig_np, interpolation="bilinear")
        ax_orig.set_xticks([])
        ax_orig.set_yticks([])
        if col == 0:
            ax_orig.set_ylabel("Original", fontsize=12)
            
        # Plot mask
        ax_mask = axes[1, col]
        mask_np = masks[col, 0].numpy()
        ax_mask.imshow(mask_np, cmap="gray", interpolation="nearest")
        ax_mask.set_xticks([])
        ax_mask.set_yticks([])
        if col == 0:
            ax_mask.set_ylabel("Mask", fontsize=12)
            
        # Plot masked image
        ax_masked = axes[2, col]
        masked_np = to_numpy(masked_images[col])
        ax_masked.imshow(masked_np, interpolation="bilinear")
        ax_masked.set_xticks([])
        ax_masked.set_yticks([])
        if col == 0:
            ax_masked.set_ylabel("Masked", fontsize=12)

    plt.savefig(OUTPUT_PATH, bbox_inches="tight", dpi=200)
    print(f"Saved to {OUTPUT_PATH}")