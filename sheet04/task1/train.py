import argparse
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from data_loading import MaskedCelebADataset, DATA_ROOT
from models import InpaintingGenerator, InpaintingDiscriminator


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_transform():
    # Normalize images to [-1, 1] to match the generator's tanh output
    return transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def psnr(pred, target, mask=None, eps=1e-8):
    """PSNR in dB. If mask is given, computes it only over masked pixels."""
    if mask is not None:
        mse = torch.sum(((pred - target) ** 2) * mask) / (torch.sum(mask) + eps)
    else:
        mse = torch.mean((pred - target) ** 2)
    # data range is 2.0 since images are normalized to [-1, 1]
    return 20 * torch.log10(torch.tensor(2.0, device=pred.device)) - 10 * torch.log10(mse + eps)


@torch.no_grad()
def evaluate_on_split(netG, loader, device, max_batches=None):
    """Runs the generator in eval mode over (a subset of) a dataloader and
    returns the mean masked-region L1 and mean masked-region PSNR.
    """
    netG.eval()
    total_l1, total_mask_px, total_psnr, n_batches = 0.0, 0.0, 0.0, 0

    for i, (masked_images, masks, original_images) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        masked_images = masked_images.to(device)
        masks = masks.to(device)
        original_images = original_images.to(device)

        fake_images = netG(torch.cat([masked_images, masks], dim=1))

        total_l1 += torch.sum(torch.abs(fake_images - original_images) * masks).item()
        total_mask_px += torch.sum(masks).item()
        total_psnr += psnr(fake_images, original_images, masks).item()
        n_batches += 1

    netG.train()

    mean_l1 = total_l1 / (total_mask_px + 1e-8)
    mean_psnr = total_psnr / max(n_batches, 1)
    return mean_l1, mean_psnr


def save_checkpoint(netG, netD, optimizerG, optimizerD, epoch, step, val_l1, val_psnr, path):
    torch.save({
        "epoch": epoch,
        "step": step,
        "generator": netG.state_dict(),
        "discriminator": netD.state_dict(),
        "optimizerG": optimizerG.state_dict(),
        "optimizerD": optimizerD.state_dict(),
        "val_l1_masked": val_l1,
        "val_psnr": val_psnr,
    }, path)


def train(epochs, batch_size, lr, device, val_batch_size, val_max_batches,
          val_every_steps, seed, out_dir):
    print(f"Using device: {device}")
    set_seed(seed)

    transform = get_transform()

    # ---- Data ----
    print("Initializing datasets...")
    train_dataset = MaskedCelebADataset(root=DATA_ROOT, split="train", transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )


    val_dataset = MaskedCelebADataset(root=DATA_ROOT, split="valid", transform=transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=val_batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    print(f"Train images: {len(train_dataset)} | Validation images: {len(val_dataset)}")


    if val_every_steps is None:
        val_every_steps = len(train_loader)

    # ---- Models ----
    netG = InpaintingGenerator().to(device)
    netD = InpaintingDiscriminator().to(device)

    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()


    best_val_l1 = float("inf")
    global_step = 0

    print("Starting training loop...")
    for epoch in range(1, epochs + 1):
        for i, (masked_images, masks, original_images) in enumerate(train_loader):
            masked_images = masked_images.to(device)
            masks = masks.to(device)
            original_images = original_images.to(device)

            # ---- Train Discriminator ----
            netD.zero_grad()

            real_input = torch.cat([masked_images, masks, original_images], dim=1)
            pred_real = netD(real_input)
            loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))

            gen_input = torch.cat([masked_images, masks], dim=1)
            fake_images = netG(gen_input)

            fake_input = torch.cat([masked_images, masks, fake_images.detach()], dim=1)
            pred_fake = netD(fake_input)
            loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))

            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizerD.step()

            # ---- Train Generator ----
            netG.zero_grad()

            fake_input_for_G = torch.cat([masked_images, masks, fake_images], dim=1)
            pred_fake_for_G = netD(fake_input_for_G)

            loss_G_GAN = criterion_GAN(pred_fake_for_G, torch.ones_like(pred_fake_for_G))
            loss_G_L1_global = criterion_L1(fake_images, original_images)
            loss_G_L1_masked = torch.sum(torch.abs(fake_images - original_images) * masks) / (
                torch.sum(masks) + 1e-8
            )

            loss_G = loss_G_GAN + 10.0 * loss_G_L1_global + 100.0 * loss_G_L1_masked
            loss_G.backward()
            optimizerG.step()

            global_step += 1

            if torch.isnan(loss_D) or torch.isnan(loss_G):
                print("NaN loss encountered! Aborting training.")
                sys.exit(1)

            if i % 10 == 0:
                print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(train_loader)}] "
                      f"Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f} "
                      f"(GAN: {loss_G_GAN.item():.4f}, L1: {loss_G_L1_global.item():.4f}, "
                      f"L1_masked: {loss_G_L1_masked.item():.4f})")

            # ---- Validation-driven checkpointing ----
            if global_step % val_every_steps == 0:
                val_l1, val_psnr = evaluate_on_split(
                    netG, val_loader, device, max_batches=val_max_batches
                )
                print(f"  [Validation @ step {global_step}] "
                      f"masked L1: {val_l1:.4f} | masked PSNR: {val_psnr:.2f} dB")

                if val_l1 < best_val_l1:
                    best_val_l1 = val_l1
                    save_checkpoint(netG, netD, optimizerG, optimizerD, epoch, global_step,
                                     val_l1, val_psnr, f"{out_dir}/best_checkpoint.pth")
                    torch.save(netG.state_dict(), f"{out_dir}/best_generator.pth")
                    torch.save(netD.state_dict(), f"{out_dir}/best_discriminator.pth")
                    print(f"  New best validation masked L1 ({best_val_l1:.4f})! "
                          f"Saved best_generator.pth / best_discriminator.pth")


    print(f"Training finished. Best validation masked L1: {best_val_l1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GAN for Image Inpainting on CelebA")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate for Adam optimizer")
    parser.add_argument("--val-batch-size", type=int, default=256,
                         help="Batch size used when evaluating on the validation split")
    parser.add_argument("--val-max-batches", type=int, default=20,
                         help="Number of validation batches to average over at each check "
                              "(use -1 for the full validation set, slower but exact)")
    parser.add_argument("--val-every-steps", type=int, default=None,
                         help="Run validation every N training steps (default: once per epoch)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--out-dir", type=str, default=".", help="Directory to save checkpoints to")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_max_batches = None if args.val_max_batches == -1 else args.val_max_batches

    print("Proceeding straight to main training loop...")
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        val_batch_size=args.val_batch_size,
        val_max_batches=val_max_batches,
        val_every_steps=args.val_every_steps,
        seed=args.seed,
        out_dir=args.out_dir,
    )