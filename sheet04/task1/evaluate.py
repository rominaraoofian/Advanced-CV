"""
Evaluate a trained inpainting generator: qualitative grid + held-out quantitative check.

Usage:
    python evaluate.py --checkpoint best_generator.pth --num-samples 8
"""
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt

from data_loading import MaskedCelebADataset, DATA_ROOT
from models import InpaintingGenerator


def to_numpy_disp(tensor):
    """(C,H,W) tensor in [-1,1] -> (H,W,C) uint8 in [0,255] for display."""
    img = (tensor.clamp(-1, 1) + 1) / 2          # [-1,1] -> [0,1]
    return (img.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")


def psnr(pred, target, mask=None, eps=1e-8):
    """PSNR in dB"""
    if mask is not None:
        mse = torch.sum(((pred - target) ** 2) * mask) / (torch.sum(mask) * pred.shape[0] / mask.shape[0] + eps)
    else:
        mse = F.mse_loss(pred, target)
    # data range is 2.0 since images are in [-1, 1]
    return 20 * torch.log10(torch.tensor(2.0)) - 10 * torch.log10(mse + eps)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Same normalization used at training time
    eval_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # split="test" -> held-out images the model never trained on
    dataset = MaskedCelebADataset(root=DATA_ROOT, split="test", transform=eval_transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

    netG = InpaintingGenerator().to(device)
    netG.load_state_dict(torch.load(args.checkpoint, map_location=device))
    netG.eval()
    print(f"Loaded weights from {args.checkpoint}")

    # Quantitative check on the test set 
    total_l1_masked, total_mask_px, total_psnr, n_batches = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for masked, mask, original in loader:
            masked, mask, original = masked.to(device), mask.to(device), original.to(device)
            fake = netG(torch.cat([masked, mask], dim=1))

            total_l1_masked += torch.sum(torch.abs(fake - original) * mask).item()
            total_mask_px += torch.sum(mask).item()
            total_psnr += psnr(fake, original, mask).item()
            n_batches += 1

    mean_l1_masked = total_l1_masked / (total_mask_px + 1e-8)
    mean_psnr = total_psnr / n_batches
    print(f"\n=== Held-out TEST SET results ({len(dataset)} images) ===")
    print(f"Mean masked-region L1 : {mean_l1_masked:.4f}  (range [-1,1], so /2 for fraction of full range)")
    print(f"Mean masked-region PSNR: {mean_psnr:.2f} dB")

    # ---- 2. Qualitative grid on a handful of samples ----
    n = args.num_samples
    small_loader = torch.utils.data.DataLoader(dataset, batch_size=n, shuffle=True)
    masked, mask, original = next(iter(small_loader))
    masked, mask, original = masked.to(device), mask.to(device), original.to(device)
    with torch.no_grad():
        fake = netG(torch.cat([masked, mask], dim=1))
        composited = masked * (1 - mask) + fake * mask

    fig, axes = plt.subplots(4, n, figsize=(n * 2.2, 4 * 2.2),
                              gridspec_kw={"hspace": 0.15, "wspace": 0.05})
    row_labels = ["Original", "Masked input", "Raw G output", "Composited"]
    rows = [original, masked, fake, composited]

    for col in range(n):
        for row_idx, (label, tensor_batch) in enumerate(zip(row_labels, rows)):
            ax = axes[row_idx, col]
            ax.imshow(to_numpy_disp(tensor_batch[col]), interpolation="bilinear")
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(label, fontsize=11)

    plt.savefig("eval_qualitative_grid.png", bbox_inches="tight", dpi=200)
    print("\nSaved qualitative grid to eval_qualitative_grid.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="best_generator.pth")
    parser.add_argument("--num-samples", type=int, default=8)
    args = parser.parse_args()
    main(args)