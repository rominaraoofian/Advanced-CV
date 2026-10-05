import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchmetrics.image.fid import FrechetInceptionDistance
import matplotlib.pyplot as plt
import numpy as np
try:
    from task2_dataloader import CelebADataSet, SELECTED_ATTRS
except ImportError:
    from task2_dataloader import CelebADataSet, SELECTED_ATTRS
try:
    from task2_model1 import Generator, Z_DIM
except ImportError:
    from task2_model1 import Generator, Z_DIM
def to_pm1(x):
    return x * 2.0 - 1.0
def ensure_pm1(imgs):
    if imgs.min() >= -0.1: 
        return to_pm1(imgs)
    return imgs
class SimpleAttrClassifier(nn.Module):
    def __init__(self, n_attr):
        super().__init__()
        ch = 64
        self.features = nn.Sequential(
            nn.Conv2d(3, ch, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, ch * 2, 4, 2, 1),
            nn.BatchNorm2d(ch * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch * 2, ch * 4, 4, 2, 1),
            nn.BatchNorm2d(ch * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch * 4, ch * 8, 4, 2, 1),
            nn.BatchNorm2d(ch * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.fc = nn.Linear(ch * 8, n_attr)
    def forward(self, x):
        return self.fc(self.features(x))
def train_external_classifier(data_root, device, epochs=5, batch_size=128):
    dataset = CelebADataSet(root=data_root, split="train", image_size=64)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    clf = SimpleAttrClassifier(len(SELECTED_ATTRS)).to(device)
    optimizer = torch.optim.Adam(clf.parameters(), lr=2e-4, betas=(0.5, 0.999))
    clf.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for imgs, labels in loader:
            imgs = ensure_pm1(imgs).to(device)
            labels = labels.to(device)
            logits = clf(imgs)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"[Classifier] Epoch {epoch+1}/{epochs} | Loss: {running_loss / len(loader):.4f}")
    return clf

@torch.no_grad()
def conditioning_metrics(G, clf, device, n_attr, n_samples=5000, batch_size=200):
    G.eval()
    clf.eval()
    correct = torch.zeros(n_attr, device=device)
    tp = torch.zeros(n_attr, device=device)
    fp = torch.zeros(n_attr, device=device)
    fn = torch.zeros(n_attr, device=device)
    num_batches = n_samples // batch_size
    for _ in range(num_batches):
        z = torch.randn(batch_size, Z_DIM, device=device)
        labels = (torch.rand(batch_size, n_attr, device=device) > 0.5).float()
        fake = G(z, labels)
        pred = (torch.sigmoid(clf(fake)) > 0.5).float()
        correct += (pred == labels).sum(dim=0)
        tp += ((pred == 1) & (labels == 1)).sum(dim=0)
        fp += ((pred == 1) & (labels == 0)).sum(dim=0)
        fn += ((pred == 0) & (labels == 1)).sum(dim=0)
    acc = correct / n_samples
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = (2 * precision * recall) / (precision + recall + 1e-8)
    results = {}
    for i, attr in enumerate(SELECTED_ATTRS):
        results[attr] = {
            "accuracy": acc[i].item(),
            "f1": f1[i].item()
        }
    return results

@torch.no_grad()
def compute_fid(G, data_root, device, n_attr, num_samples=5000, batch_size=64):
    G.eval()
    fid = FrechetInceptionDistance(feature=2048).to(device)
    real_ds = CelebADataSet(root=data_root, split="test", image_size=64)
    real_loader = DataLoader(real_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    all_labels = []
    seen = 0
    for imgs, labels in real_loader:
        bs = imgs.size(0)
        if seen + bs > num_samples:
            bs = num_samples - seen
            imgs = imgs[:bs]
            labels = labels[:bs]
        # Standardize real images from [-1, 1] to [0, 255] uint8 for torchmetrics FID
        imgs = ((imgs + 1.0) / 2.0).clamp(0, 1)
        imgs = (imgs * 255).to(torch.uint8).to(device)
        fid.update(imgs, real=True)
        all_labels.append(labels)
        seen += bs
        if seen >= num_samples:
            break
    all_labels = torch.cat(all_labels, dim=0).to(device)
    generated = 0
    while generated < num_samples:
        bs = min(batch_size, num_samples - generated)
        z = torch.randn(bs, Z_DIM, device=device)
        labels = all_labels[generated:generated + bs]
        fake = G(z, labels)
        fake = ((fake + 1.0) / 2.0).clamp(0, 1)
        fake = (fake * 255).to(torch.uint8)
        fid.update(fake, real=False)
        generated += bs
    return fid.compute().item()

@torch.no_grad()
def save_random_samples(G, device, n_attr, out_path, n=64):
    G.eval()
    z = torch.randn(n, Z_DIM, device=device)
    labels = (torch.rand(n, n_attr, device=device) > 0.5).float()
    fake = G(z, labels)
    fake = ((fake + 1.0) / 2.0).clamp(0, 1)
    save_image(fake, out_path, nrow=8)
    
@torch.no_grad()
def save_attribute_sweep(G, device, n_attr, out_path, n_identities=5):
    G.eval()
    z_id = torch.randn(n_identities, Z_DIM, device=device)
    base_labels = (torch.rand(n_identities, n_attr, device=device) > 0.5).float()

    fig, axes = plt.subplots(
        n_attr * 2,
        n_identities,
        figsize=(n_identities, n_attr * 2),
        gridspec_kw={"wspace": 0, "hspace": 0}
    )

    if n_attr == 1:
        axes = np.expand_dims(axes, axis=0)

    for a in range(n_attr):
        name = SELECTED_ATTRS[a]

        labels_off = base_labels.clone()
        labels_on = base_labels.clone()

        labels_off[:, a] = 0
        labels_on[:, a] = 1

        imgs_off = G(z_id, labels_off)
        imgs_on = G(z_id, labels_on)

        imgs_off = ((imgs_off + 1) / 2).clamp(0, 1).cpu().numpy()
        imgs_on = ((imgs_on + 1) / 2).clamp(0, 1).cpu().numpy()

        for i in range(n_identities):
            axes[a * 2, i].imshow(
                np.transpose(imgs_off[i], (1, 2, 0))
            )
            axes[a * 2, i].axis("off")

            axes[a * 2 + 1, i].imshow(
                np.transpose(imgs_on[i], (1, 2, 0))
            )
            axes[a * 2 + 1, i].axis("off")

        # axes[a * 2, 0].set_ylabel(f"{name} OFF", fontsize=8)
        # axes[a * 2 + 1, 0].set_ylabel(f"{name} ON", fontsize=8)

    # plt.subplots_adjust(
    #     left=1,
    #     right=1,
    #     top=1,
    #     bottom=1,
    #     wspace=1,
    #     hspace=1
    # )

    plt.savefig(
        out_path,
        bbox_inches="tight",
        pad_inches=0,
        dpi=300
    )
    plt.close()
@torch.no_grad()
def sample_fixed_grid(n_attr, device, n_samples_per_attr=8):
    """
    Rows:
      0: Smiling
      1: Eyeglasses
      2: Male
      3: Blond_Hair
      4: Young
      5: Smiling + Male
      6: Blond_Hair + Young
      7: Blond_Hair + Male
      8: Eyeglasses + Male + Young
    """
    z_identities = torch.randn(n_samples_per_attr, Z_DIM, device=device)
    mixed_combinations = [
        [0, 2],       # Smiling + Male
        [3, 4],       # Blond_Hair + Young
        [2, 3],       # Male + Blond_Hair
        [1, 2, 4],    # Eyeglasses + Male + Young
    ]
    total_rows = n_attr + len(mixed_combinations)
    total_samples = total_rows * n_samples_per_attr
    grid_z = z_identities.repeat(total_rows, 1)
    grid_labels = torch.zeros(total_samples, n_attr, device=device)
    for attr_idx in range(n_attr):
        start = attr_idx * n_samples_per_attr
        end = start + n_samples_per_attr
        grid_labels[start:end, attr_idx] = 1.0
    for mix_idx, combo in enumerate(mixed_combinations):
        row_offset = n_attr + mix_idx
        start = row_offset * n_samples_per_attr
        end = start + n_samples_per_attr
        for attr_idx in combo:
            grid_labels[start:end, attr_idx] = 1.0
    return grid_z, grid_labels, n_samples_per_attr

@torch.no_grad()
def save_fixed_tracking_grid(G, device, n_attr, out_path):
    G.eval()
    z, labels, nrow = sample_fixed_grid(
        n_attr=n_attr,
        device=device,
        n_samples_per_attr=8
    )
    imgs = G(z, labels)
    imgs = ((imgs + 1.0) / 2.0).clamp(0.0, 1.0)
    save_image(
        imgs,
        out_path,
        nrow=nrow
    )
    print(f"Saved fixed attribute tracking grid -> {out_path}")
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--clf_checkpoint", type=str, default="clf.pt")
    parser.add_argument("--out_dir", type=str, default="./outputs_evaluation_task2")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_attr = len(SELECTED_ATTRS)
    # ---------------- Load Generator ----------------
    print("Loading Generator...")
    G = Generator(n_attr=n_attr).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    G.load_state_dict(ckpt["G"])
    # ---------------- Classifier ----------------
    if os.path.exists(args.clf_checkpoint):
        print("Loading classifier...")
        clf = SimpleAttrClassifier(n_attr).to(device)
        clf.load_state_dict(torch.load(args.clf_checkpoint, map_location=device))
    else:
        print("Training classifier...")
        clf = train_external_classifier(args.data_root, device)
        torch.save(clf.state_dict(), args.clf_checkpoint)
    # ---------------- Metrics ----------------
    print("Computing conditioning metrics...")
    cond = conditioning_metrics(G, clf, device, n_attr)
    for k, v in cond.items():
        print(f"{k:15s} Acc={v['accuracy']:.4f} F1={v['f1']:.4f}")
    
    fid = 1000    
    # print("Computing FID...")
    # fid = compute_fid(G, args.data_root, device, n_attr)
    # print("FID:", fid)
    # ---------------- Outputs ----------------
    save_random_samples(G, device, n_attr, os.path.join(args.out_dir, "samples.png"))
    save_attribute_sweep(G, device, n_attr, os.path.join(args.out_dir, "sweep.png"))
    save_fixed_tracking_grid(
        G,
        device,
        n_attr,
        os.path.join(args.out_dir, "fixed_tracking_grid.png")
    )
    
    with open(os.path.join(args.out_dir, "report.txt"), "w") as f:
        f.write(f"FID: {fid}\n\n")
        for k, v in cond.items():
            f.write(f"{k}: Acc={v['accuracy']:.4f}, F1={v['f1']:.4f}\n")
    print("Done.")
if __name__ == "__main__":
    main()