import torch.nn as nn

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
        sigmas_tensor = torch.stack([sigma_current.view(-1), sigma_next.view(-1)], dim=1)
        sigma_embedding = self.sigma_embed(sigmas_tensor).unsqueeze(-1).unsqueeze(-1)
        e1 = self.enc1(net_input)
        e2 = self.enc2(self.downsample(e1))
        m = self.mid(self.downsample(e2))
        d1 = self.dec1(torch.cat([self.upsample(m), e2], dim=1))
        d2 = self.dec2(torch.cat([self.upsample(d1), e1], dim=1))
        output = self.output_conv(d2 + sigma_embedding)
        return x + output

CACHED_ML_SAMPLER = {}

@torch.no_grad()
def sample_ml_driven(
    model, 
    x, 
    sigmas, 
    extra_args=None, 
    callback=None, 
    disable=None,
    model_path='models/sampler/sampler_precision.pth'
):
    device = x.device
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]], device=device)

    ml_sampler_model = CACHED_ML_SAMPLER.get(model_path)
    if ml_sampler_model is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ML采样器模型未找到: {model_path}。请先运行训练脚本。")
        
        print(f"加载ML采样器: {model_path}")
        ml_sampler_model = AdvancedSamplerNet_v2()
        ml_sampler_model.load_state_dict(torch.load(model_path, map_location=device))
        ml_sampler_model.to(device)
        ml_sampler_model.eval()

        CACHED_ML_SAMPLER[model_path] = ml_sampler_model
    
    for i in trange(len(sigmas) - 1, disable=disable, desc="ML Sampler 推理中"):
        sigma_current = sigmas[i]
        sigma_next = sigmas[i+1]
        
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigma_current, 'denoised': x})
        denoised = model(x, sigma_current * s_in, **extra_args)
        
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigma_current, 'denoised': denoised})

        x = ml_sampler_model(x, denoised, sigma_current, sigma_next)
    
    return x
