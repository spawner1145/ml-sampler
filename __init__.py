from . import gather_sampler
from .gather_sampler import sample_euler_ancestral_with_data_collection


if gather_sampler.BACKEND == "ComfyUI":
    if not gather_sampler.INITIALIZED:
        from comfy.k_diffusion import sampling as k_diffusion_sampling
        from comfy.samplers import SAMPLER_NAMES

        setattr(k_diffusion_sampling, "sample_euler_ancestral_with_data_collection", sample_euler_ancestral_with_data_collection)

        SAMPLER_NAMES.append("euler_ancestral_with_data_collection")

        gather_sampler.INITIALIZED = True

NODE_CLASS_MAPPINGS = {}
