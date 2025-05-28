from importlib import import_module
import torch
from tqdm.auto import trange
import os
import uuid

sampling = None
BACKEND = None
INITIALIZED = False

if not BACKEND:
    try:
        _ = import_module("modules.sd_samplers_kdiffusion")
        sampling = import_module("k_diffusion.sampling")
        BACKEND = "WebUI"
    except ImportError as _:
        pass

if not BACKEND:
    try:
        sampling = import_module("comfy.k_diffusion.sampling")
        BACKEND = "ComfyUI"
    except ImportError as _:
        pass

def default_noise_sampler(x):
    return lambda sigma, sigma_next: torch.randn_like(x)

def get_ancestral_step(sigma_from, sigma_to, eta=1.):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    if not eta:
        return sigma_to, 0.
    sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (None,) * dims_to_append]


def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)


@torch.no_grad()
def sample_euler_ancestral_with_data_collection(
    model, 
    x, 
    sigmas, 
    extra_args=None, 
    callback=None, 
    disable=None, 
    eta=1.0, 
    s_noise=1.0,
    data_dir="sampler_data"
):
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x)
    s_in = x.new_ones([x.shape[0]], device=x.device)

    run_uuid = str(uuid.uuid4())[:8]
    run_data_dir = os.path.join(data_dir, f"run_{run_uuid}")
    os.makedirs(run_data_dir, exist_ok=True)
    
    pbar_desc = f"数据收集中 (Run ID: {run_uuid})"
    for i in trange(len(sigmas) - 1, disable=disable, desc=pbar_desc):
        
        input_x = x.detach().clone()
        sigma_current = sigmas[i].detach().clone()
        sigma_next_original = sigmas[i + 1].detach().clone()
        
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'denoised': denoised})
        if sigma_down == 0:
            x = denoised
        else:
            d = to_d(x, sigmas[i], denoised)
            dt = sigma_down - sigmas[i]
            added_noise = noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
            x = x + d * dt + added_noise
        target_x = x.detach().clone()
        data_packet = {
            'input_x': input_x.cpu(),
            'predicted_denoised': denoised.detach().cpu(),
            'sigma_current': sigma_current.cpu(),
            'sigma_next': sigma_next_original.cpu(),
            'target_x': target_x.cpu()
        }
        torch.save(data_packet, os.path.join(run_data_dir, f"step_{i:03d}.pt"))
        
    print(f"✅ 运行 {run_uuid} 的数据已成功保存到 {run_data_dir}")
    return x