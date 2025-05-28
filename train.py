import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import _LRScheduler
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import lpips
import random
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.Mish()
        )
    def forward(self, x):
        return self.block(x)

class AdvancedSamplerNet_v2(nn.Module):
    def __init__(self, input_channels=12, base_channels=32):
        super().__init__()
        self.sigma_embed = nn.Sequential(nn.Linear(2, base_channels), nn.Mish(), nn.Linear(base_channels, base_channels))
        self.enc1 = ConvBlock(input_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.downsample = nn.MaxPool2d(2)
        self.mid = ConvBlock(base_channels * 2, base_channels * 4)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = ConvBlock(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec2 = ConvBlock(base_channels * 2 + base_channels, base_channels)
        self.output_conv = nn.Conv2d(base_channels, 4, kernel_size=1)

    def forward(self, x, denoised, sigma_current, sigma_next):
        diff = denoised - x
        net_input = torch.cat([x, denoised, diff], dim=1)
        sigmas_tensor = torch.stack([sigma_current, sigma_next], dim=1).view(-1, 2)
        sigma_embedding = self.sigma_embed(sigmas_tensor).unsqueeze(-1).unsqueeze(-1)
        e1 = self.enc1(net_input)
        e2 = self.enc2(self.downsample(e1))
        m = self.mid(self.downsample(e2))
        d1 = self.dec1(torch.cat([self.upsample(m), e2], dim=1))
        d2 = self.dec2(torch.cat([self.upsample(d1), e1], dim=1))
        output = self.output_conv(d2 + sigma_embedding)
        return x + output

# --- 指数移动平均 (EMA) 工具类 ---
class EMA:
    def __init__(self, model, decay):
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
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

# --- 数据集类 ---
class SamplerDataset(Dataset):
    def __init__(self, data_dir="sampler_data", noise_level=0.005):
        self.file_paths = glob.glob(os.path.join(data_dir, "run_*", "step_*.pt"))
        self.noise_level = noise_level
        if not self.file_paths:
            raise FileNotFoundError(f"在 '{data_dir}' 中没有找到任何数据！请先运行数据收集脚本。")
        logging.info(f"发现 {len(self.file_paths)} 个训练数据点。")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data_packet = torch.load(self.file_paths[idx])
        input_x = data_packet['input_x'].squeeze(0)
        predicted_denoised = data_packet['predicted_denoised'].squeeze(0)
        target_x = data_packet['target_x'].squeeze(0)
        if self.noise_level > 0:
            input_x += torch.randn_like(input_x) * self.noise_level
            predicted_denoised += torch.randn_like(predicted_denoised) * self.noise_level
        
        # 返回一个包含修正后张量的新字典
        return {
            'input_x': input_x,
            'predicted_denoised': predicted_denoised,
            'sigma_current': data_packet['sigma_current'],
            'sigma_next': data_packet['sigma_next'],
            'target_x': target_x
        }

# --- 训练主函数 ---
def train(
    data_dir="sampler_data",
    epochs=100,
    batch_size=16,
    lr=2e-4,
    val_split=0.05,
    lpips_weight=0.5,
    model_save_path="models/sampler/sampler_precision.pth"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")
    
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    dataset = SamplerDataset(data_dir)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, num_workers=4, pin_memory=True)

    model = AdvancedSamplerNet_v2().to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    num_training_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=1e-6)
    
    ema = EMA(model, decay=0.999)
    scaler = GradScaler()
    loss_fn_l1 = nn.L1Loss()
    loss_fn_lpips = lpips.LPIPS(net='vgg').to(device).eval()

    best_val_loss = float('inf')
    logging.info(f"🚀 训练开始！总共 {epochs} 个 Epochs, {len(train_loader)} 步/Epoch。")

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [训练]")
        
        for batch in pbar:
            input_x = batch['input_x'].to(device, non_blocking=True)
            denoised = batch['predicted_denoised'].to(device, non_blocking=True)
            sigma_curr = batch['sigma_current'].to(device, non_blocking=True)
            sigma_next = batch['sigma_next'].to(device, non_blocking=True)
            target_x = batch['target_x'].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(): # 开启自动混合精度
                predicted_x = model(input_x, denoised, sigma_curr, sigma_next)
                l1_loss = loss_fn_l1(predicted_x, target_x)
                lpips_loss = loss_fn_lpips(predicted_x[:, :3, :, :], target_x[:, :3, :, :]).mean()
                total_loss = l1_loss + lpips_weight * lpips_loss
            
            scaler.scale(total_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 梯度裁剪
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update()

            pbar.set_postfix(loss=f"{total_loss.item():.4f}", l1=f"{l1_loss.item():.4f}", lpips=f"{lpips_loss.item():.4f}")

        # --- 验证循环 ---
        model.eval()
        ema.apply_shadow() # 使用EMA权重进行验证
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_x = batch['input_x'].to(device, non_blocking=True)
                denoised = batch['predicted_denoised'].to(device, non_blocking=True)
                sigma_curr = batch['sigma_current'].to(device, non_blocking=True)
                sigma_next = batch['sigma_next'].to(device, non_blocking=True)
                target_x = batch['target_x'].to(device, non_blocking=True)
                
                with autocast():
                    predicted_x = model(input_x, denoised, sigma_curr, sigma_next)
                    val_loss += loss_fn_l1(predicted_x, target_x).item()
        
        ema.restore() # 恢复原始权重以继续训练
        
        avg_val_loss = val_loss / len(val_loader)
        logging.info(f"Epoch {epoch+1} | 平均验证L1损失: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ema.apply_shadow()
            torch.save(model.state_dict(), model_save_path)
            ema.restore()
            logging.info(f"🎉 新的最佳模型已保存到: {model_save_path} (验证损失: {best_val_loss:.6f})")

if __name__ == "__main__":
    train()
