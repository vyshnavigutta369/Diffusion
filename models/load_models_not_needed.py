from diffusers import LMSDiscreteScheduler, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPModel, CLIPTokenizer
import torch

auth_token = "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"

model_path_clip = "openai/clip-vit-large-patch14"
clip_tokenizer = CLIPTokenizer.from_pretrained(model_path_clip)
clip_model = CLIPModel.from_pretrained(model_path_clip, torch_dtype=torch.float16)
clip = clip_model.text_model
model_path_diffusion = "CompVis/stable-diffusion-v1-4"
unet = UNet2DConditionModel.from_pretrained(model_path_diffusion, subfolder="unet", use_auth_token=auth_token, revision="fp16", torch_dtype=torch.float16)
vae = AutoencoderKL.from_pretrained(model_path_diffusion, subfolder="vae", use_auth_token=auth_token, revision="fp16", torch_dtype=torch.float16)


# #Move to GPU
device = "cuda"
# clip.to(device)
# unet.to(device)
# vae.to(device)