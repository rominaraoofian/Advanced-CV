import os
import math
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Conditional Attributes 
SELECTED_ATTRS = ["Smiling", "Eyeglasses", "Male", "Blond_Hair", "Young"]

def load_full_attributes(root, attr_file="list_attr_celeba.txt",
                          partition_file="list_eval_partition.txt", split=None):
    attr_path = os.path.join(root, attr_file)
    if not os.path.exists(attr_path):
        raise FileNotFoundError(f"Attribute file not found at: {attr_path}")
        
    df = pd.read_csv(attr_path, sep=r"\s+", skiprows=1, header=0)
    df = df.replace(-1, 0)
    df.index.name = "filename"
    df = df.reset_index()

    if split is not None:
        part_path = os.path.join(root, partition_file)
        if os.path.exists(part_path):
            part = pd.read_csv(part_path, sep=r"\s+", header=None, names=["filename", "split"])
            split_id = {"train": 0, "val": 1, "test": 2}[split]
            keep = set(part[part.split == split_id].filename)
            df = df[df.filename.isin(keep)].reset_index(drop=True)
        else:
            print(f"Warning: Partition file not found at {part_path}. Using all samples.")
    return df


def plot_attribute_distributions(df, attrs, out_path, ncols=5):


    n = len(attrs)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.6, nrows * 2.4))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, a in enumerate(attrs):
        counts = df[a].value_counts().reindex([0, 1], fill_value=0)
        pct_pos = 100.0 * df[a].mean()
        axes[i].bar(["0 (no)", "1 (yes)"], counts.values, color=["#B0413E", "#3E7CB1"])
        axes[i].set_title(f"{a}\n({pct_pos:.1f}% positive)", fontsize=8)
        axes[i].tick_params(axis="x", labelsize=7)
        axes[i].tick_params(axis="y", labelsize=7)

    for j in range(len(attrs), len(axes)):
        axes[j].axis("off")

    fig.suptitle("CelebA attribute distributions", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved attribute distribution grid -> {out_path}")


def plot_correlation_matrix(df, attrs, out_path):

    corr = df[attrs].corr()
    size = max(6, 0.32 * len(attrs))
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(attrs)))
    ax.set_xticklabels(attrs, rotation=90, fontsize=6)
    ax.set_yticks(range(len(attrs)))
    ax.set_yticklabels(attrs, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    ax.set_title("CelebA attribute correlation matrix", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved correlation heatmap -> {out_path}")
    return corr


def run_eda(root, out_dir="./eda_outputs", attrs=None, split=None):
    os.makedirs(out_dir, exist_ok=True)
    df = load_full_attributes(root, split=split)
    if attrs is None:
        attrs = [c for c in df.columns if c != "filename"]

    plot_attribute_distributions(df, attrs, os.path.join(out_dir, "attribute_distributions.png"))
    corr = plot_correlation_matrix(df, attrs, os.path.join(out_dir, "attribute_correlation.png"))
    corr.to_csv(os.path.join(out_dir, "attribute_correlation.csv"))

    pos_rates = df[attrs].mean().sort_values(ascending=False)
    pos_rates.to_csv(os.path.join(out_dir, "attribute_positive_rates.csv"), header=["positive_rate"])
    print(f"Saved correlation matrix CSV and positive-rate CSV to {out_dir}")

    return {"positive_rates": pos_rates, "correlation": corr}


class CelebADataSet(Dataset):
    def __init__(self, root, split="train", image_size=64, attrs=SELECTED_ATTRS,
                 attr_file="list_attr_celeba.txt", partition_file="list_eval_partition.txt",
                 img_dir="img_align_celeba"):
        self.root = root
        self.img_dir = os.path.join(root, img_dir)
        self.attrs = attrs

        df = load_full_attributes(root, attr_file=attr_file, partition_file=partition_file, split=split)

        self.filenames = df["filename"].values
        self.labels = df[attrs].values.astype("float32")

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),  
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.filenames[idx])
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        label = torch.from_numpy(self.labels[idx])  
        return img, label

    def attribute_balance(self):
        return {a: float(self.labels[:, i].mean()) for i, a in enumerate(self.attrs)}
