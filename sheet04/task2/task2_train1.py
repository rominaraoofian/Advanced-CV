import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from task2_dataloader import CelebADataSet, SELECTED_ATTRS
from task2_model1 import Generator, Discriminator, Z_DIM
# --- Exponential Moving Average (EMA) for Generator ---
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()
    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data.copy_(self.backup[name])
        self.backup = {}
        
def diff_augment(x, policy="color,translation,cutout"):
    if not policy:
        return x
    for p in policy.split(","):
        if p == "color":
            brightness = torch.randn(x.size(0), 1, 1, 1, device=x.device) * 0.2
            contrast = torch.rand(x.size(0), 1, 1, 1, device=x.device) * 0.4 + 0.8
            x = (x + brightness) * contrast
        elif p == "translation":
            batch_size, _, h, w = x.size()
            shift_x = torch.randint(-8, 9, (batch_size,), device=x.device)
            shift_y = torch.randint(-8, 9, (batch_size,), device=x.device)
            
            grid_y, grid_x = torch.meshgrid(
                torch.arange(h, device=x.device), 
                torch.arange(w, device=x.device),
                indexing="ij"
            )
            grid_x = grid_x.unsqueeze(0).repeat(batch_size, 1, 1) + shift_x.view(-1, 1, 1)
            grid_y = grid_y.unsqueeze(0).repeat(batch_size, 1, 1) + shift_y.view(-1, 1, 1)
            
            grid_x = 2.0 * grid_x / (w - 1) - 1.0
            grid_y = 2.0 * grid_y / (h - 1) - 1.0
            grid = torch.stack((grid_x, grid_y), dim=-1)
            x = F.grid_sample(x, grid, padding_mode="zeros", align_corners=True)
        elif p == "cutout":
            batch_size, c, h, w = x.size()
            cut_size = 16
            offset_x = torch.randint(0, w - cut_size + 1, (batch_size,), device=x.device)
            offset_y = torch.randint(0, h - cut_size + 1, (batch_size,), device=x.device)
            
            mask = torch.ones(batch_size, 1, h, w, device=x.device)
            for i in range(batch_size):
                mask[i, :, offset_y[i] : offset_y[i] + cut_size, offset_x[i] : offset_x[i] + cut_size] = 0.0
            x = x * mask
    return x
def sample_fixed_grid(n_attr, device, n_samples_per_attr=8):
    z_identities = torch.randn(n_samples_per_attr, Z_DIM, device=device)
    mixed_combinations = [
        [0, 2],     # Mix 1: Smiling Male
        [3, 4],     # Mix 2: Blond Hair + Young
        [2, 3],     # Mix 3: Blond Hair  + Male
        [1, 2, 4]   # Mix 4: Young Male with Eyeglasses
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

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = CelebADataSet(root=args.data_root, split="train", image_size=64)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                     num_workers=args.num_workers, drop_last=True, pin_memory=True)
    n_attr = len(SELECTED_ATTRS)
    G = Generator(n_attr=n_attr).to(device)
    D = Discriminator(n_attr=n_attr).to(device)
    ema = EMA(G, decay=args.ema_decay)
    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.5, 0.999))
    # Initialize Mixed Precision scaler
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    # Binary cross entropy loss module for adversarial and attribute classification
    criterion = nn.BCEWithLogitsLoss()
    fixed_z, fixed_labels, grid_nrow = sample_fixed_grid(n_attr, device)
    step = 0
    for epoch in range(args.epochs):
        for real_imgs, real_labels in dl:
            real_imgs = real_imgs.to(device)
            real_labels = real_labels.to(device)
            bs = real_imgs.size(0)
            # Labels for real/fake classification
            real_target = torch.ones(bs, device=device)
            fake_target = torch.zeros(bs, device=device)
            # ---------------- Discriminator step ----------------
            # Generate fake samples
            z = torch.randn(bs, Z_DIM, device=device)
            gen_labels = (torch.rand(bs, n_attr, device=device) > 0.5).float()
            
            with torch.no_grad():
                with torch.amp.autocast(device_type=device.type, enabled=(scaler is not None)):
                    fake_imgs = G(z, gen_labels)
            # Apply optional DiffAugment
            real_aug = diff_augment(real_imgs, args.diff_augment_policy)
            fake_aug = diff_augment(fake_imgs, args.diff_augment_policy)
            # Forward passes under mixed precision
            with torch.amp.autocast(device_type=device.type, enabled=(scaler is not None)):
                real_adv_logits, real_attr_logits = D(real_aug)
                fake_adv_logits, _ = D(fake_aug)
                
                d_loss_real = criterion(real_adv_logits, real_target)
                d_loss_fake = criterion(fake_adv_logits, fake_target)
                d_adv = d_loss_real + d_loss_fake
                
                # Attribute classification loss on real images
                d_cls = criterion(real_attr_logits, real_labels)
                
                d_loss = d_adv + args.lambda_cls * d_cls
            opt_d.zero_grad()
            if scaler is not None:
                scaler.scale(d_loss).backward()
                scaler.step(opt_d)
            else:
                d_loss.backward()
                opt_d.step()
            # ---------------- Generator step ----------------
            z = torch.randn(bs, Z_DIM, device=device)
            gen_labels = (torch.rand(bs, n_attr, device=device) > 0.5).float()
            with torch.amp.autocast(device_type=device.type, enabled=(scaler is not None)):
                fake_imgs = G(z, gen_labels)
                fake_aug = diff_augment(fake_imgs, args.diff_augment_policy)
                fake_adv_logits, fake_attr_logits = D(fake_aug)
                # Generator wants the discriminator to think the fake images are real
                g_adv = criterion(fake_adv_logits, real_target)
                
                # Generator wants to correctly represent the conditional attributes
                g_cls = criterion(fake_attr_logits, gen_labels)
                
                g_loss = g_adv + args.lambda_cls * g_cls
            opt_g.zero_grad()
            if scaler is not None:
                scaler.scale(g_loss).backward()
                scaler.step(opt_g)
                scaler.update()
            else:
                g_loss.backward()
                opt_g.step()
            ema.update()
            # ---------------- Logging & Sampling ----------------
            if step % args.log_every == 0:
                print(f"epoch {epoch} step {step} | "
                      f"D {d_loss.item():.3f} (adv {d_adv.item():.3f} cls {d_cls.item():.3f}) | "
                      f"G {g_loss.item():.3f} (adv {g_adv.item():.3f} cls {g_cls.item():.3f})")
            if step % args.sample_every == 0:
                G.eval()
                ema.apply_shadow()
                with torch.no_grad():
                    with torch.amp.autocast(device_type=device.type, enabled=(scaler is not None)):
                        samples = G(fixed_z, fixed_labels)
                os.makedirs(args.out_dir, exist_ok=True)
                save_image(samples, os.path.join(args.out_dir, f"step_{step:06d}.png"),
                           nrow=grid_nrow, normalize=True, value_range=(-1, 1))
                ema.restore()
                G.train()
            step += 1
        ema.apply_shadow()
        torch.save({
            "G": G.state_dict(),
            "D": D.state_dict(),
            "epoch": epoch,
            "ema_shadow": ema.shadow
        }, os.path.join(args.out_dir, "checkpoint_last.pt"))
        ema.restore()
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True, help="Path to CelebA dataset root")
    p.add_argument("--out_dir", type=str, default="./outputs")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr_g", type=float, default=2e-4)
    p.add_argument("--lr_d", type=float, default=2e-4)
    p.add_argument("--lambda_cls", type=float, default=1.0, help="Auxiliary classification loss coefficient")
    p.add_argument("--ema_decay", type=float, default=0.999, help="Exponential Moving Average decay coefficient")
    p.add_argument("--diff_augment_policy", type=str, default="color,translation", 
                   help="DiffAugment policies separated by comma (e.g. 'color,translation,cutout' or '')")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--sample_every", type=int, default=500)
    args = p.parse_args()
    train(args)
