import inspect
import warnings
from tqdm.auto import tqdm
from typing import List, Optional, Union

import torch
from diffusers import ModelMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import \
    StableDiffusionSafetyChecker
from diffusers.schedulers import (DDIMScheduler, LMSDiscreteScheduler,
                                  PNDMScheduler)
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer, CLIPModel
from src.utils import get_init_image
from difflib import SequenceMatcher
import PIL

import random
import os, requests
from PIL import Image
from io import BytesIO
import numpy as np

from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel
def get_init_image(url):
    response = requests.get(url)
    init_img = Image.open(BytesIO(response.content))
    return init_img


class StableDiffusionPipeline(DiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
    ):
        super().__init__()
        scheduler = scheduler.set_format("pt")
        self.register_modules(
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
    
        self.image_processor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda")     ## NOT SURE OF THE PRETRAINED MODEL
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")         ## NOT SURE OF THE PRETRAINED MODEL
        self.image_encoder =  StableDiffusionImageEmbedPipeline.from_pretrained("lambdalabs/sd-image-variations-diffusers", revision="273115e88df42350019ef4d628265b8c29ef4af5").image_encoder         ## NOT SURE OF THE PRETRAINED MODEL


        # print (hello)

    def enable_attention_slicing(self, slice_size: Optional[Union[str, int]] = "auto"):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.

        Args:
            slice_size (`str` or `int`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                a number is provided, uses as many slices as `attention_head_dim // slice_size`. In this case,
                `attention_head_dim` must be a multiple of `slice_size`.
        """
        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = self.unet.config.attention_head_dim // 2
        self.unet.set_attention_slice(slice_size)

    def disable_attention_slicing(self):
        r"""
        Disable sliced attention computation. If `enable_attention_slicing` was previously invoked, this method will go
        back to computing attention in one step.
        """
        # set slice_size = `None` to disable `attention slicing`
        self.enable_attention_slicing(None)

    @torch.no_grad()
    def __call__(
        self,
        input_image: Optional[Union[torch.FloatTensor, PIL.Image.Image, List[PIL.Image.Image]]] = None,
        prompt: Optional[Union[str, List[str]]] = None,
        t_start: Optional[int] = 0,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        text_embeddings: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        use_image_embeddings_only = False,
        **kwargs,
    ):
        width = width - width % 64
        height = height - height % 64

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)
        
        #If seed is None, randomly select seed from 0 to 2^32-1
        if generator is None:
            seed = random.randrange(2**32 - 1)
            generator = torch.cuda.manual_seed(seed)
        
        #Set inference timesteps to scheduler
        scheduler = self.scheduler
        scheduler.set_timesteps(num_inference_steps)
        
        #Preprocess image if it exists (img2img)
        if latents is None:
            init_latent = torch.zeros((1, self.unet.in_channels, height // 8, width // 8), device=device)
            t_start = 0
        else:
            init_latent = latents

        if use_image_embeddings_only:
            if isinstance(input_image, PIL.Image.Image):
                batch_size = 1
            elif isinstance(input_image, list):
                batch_size = len(input_image)
            else:
                raise ValueError(f"`input_image` has to be of type `str` or `list` but is {type(input_image)}")

            if not isinstance(input_image, torch.FloatTensor):
                print ('yes')
                input_image = self.image_processor(images=input_image, return_tensors="pt")

            image_embedding_conditional = self.image_encoder.get_image_features(**input_image).unsqueeze(1).to(self.device)
            init_latent = torch.zeros((1, self.unet.in_channels, height // 8, width // 8), device=device)
            t_start = 0

        text_embedding_conditional = text_embeddings    
        if text_embeddings is None:
            if isinstance(prompt, str):
                batch_size = 1
            elif isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                raise ValueError(
                    f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
                )

            if height % 8 != 0 or width % 8 != 0:
                raise ValueError(
                    "`height` and `width` have to be divisible by 8 but are"
                    f" {height} and {width}."
                )

            # get prompt text embeddings
            text_input = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_embedding_conditional = self.text_encoder(text_input.input_ids.to(self.device)).last_hidden_state     
        
        tokens_unconditional = self.tokenizer("", padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
        text_embedding_unconditional = self.text_encoder(tokens_unconditional.input_ids.to(device)).last_hidden_state
            
        # print (use_image_embeddings_only)
        if use_image_embeddings_only:
            embedding_conditional = image_embedding_conditional
            embedding_unconditional = text_embedding_unconditional
        else:
            embedding_conditional = text_embedding_conditional
            embedding_unconditional = text_embedding_unconditional

        #Generate random normal noise
        noise = torch.randn(init_latent.shape, generator=generator, device=device)
        latent = scheduler.add_noise(init_latent, noise, t_start).to(device)

        with torch.autocast(device):

            timesteps = scheduler.timesteps[t_start:]
            
            for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
                t_index = t_start + i

                sigma = scheduler.sigmas[t_index]
                latent_model_input = latent
                latent_model_input = (latent_model_input / ((sigma**2 + 1) ** 0.5)).to(self.unet.dtype)
                
                noise_pred_cond = self.unet(latent_model_input, t, encoder_hidden_states=embedding_conditional).sample
                noise_pred_uncond = self.unet(latent_model_input, t, encoder_hidden_states=embedding_unconditional).sample
                    
                #Perform guidance
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                latent = scheduler.step(noise_pred, t_index, latent).prev_sample

            #scale and decode the image latents with vae
            latent = latent / 0.18215
            image = self.vae.decode(latent.to(self.vae.dtype)).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image[0] * 255).round().astype("uint8")
        
        # if output_type == "pil":
        #     image = self.numpy_to_pil(image)

        # return {"sample": image}
        return [Image.fromarray(image)]

    def init_attention_weights(self, weight_tuples):
        tokens_length = self.tokenizer.model_max_length
        weights = torch.ones(tokens_length)

        for i, w in weight_tuples:
            if i < tokens_length and i >= 0:
                weights[i] = w

        for name, module in self.unet.named_modules():
            module_name = type(module).__name__
            if module_name == "CrossAttention" and "attn2" in name:
                module.last_attn_slice_weights = weights.to(self.device)
            if module_name == "CrossAttention" and "attn1" in name:
                module.last_attn_slice_weights = None


    def init_attention_edit(self, tokens, tokens_edit):
        tokens_length = self.tokenizer.model_max_length
        mask = torch.zeros(tokens_length)
        indices_target = torch.arange(tokens_length, dtype=torch.long)
        indices = torch.zeros(tokens_length, dtype=torch.long)

        tokens = tokens.input_ids.numpy()[0]
        tokens_edit = tokens_edit.input_ids.numpy()[0]

        for name, a0, a1, b0, b1 in SequenceMatcher(None, tokens, tokens_edit).get_opcodes():
            if b0 < tokens_length:
                # print ('op codes', name)
                if name == "equal" or (name == "replace" and a1-a0 == b1-b0):
                    mask[b0:b1] = 1
                    indices[b0:b1] = indices_target[a0:a1]

        for name, module in self.unet.named_modules():
            module_name = type(module).__name__
            if module_name == "CrossAttention" and "attn2" in name:
                module.last_attn_slice_mask = mask.to(self.device)
                module.last_attn_slice_indices = indices.to(self.device)
            if module_name == "CrossAttention" and "attn1" in name:
                module.last_attn_slice_mask = None
                module.last_attn_slice_indices = None


    def init_attention_edit_for_pca_walk(self):

        tokens_length = self.tokenizer.model_max_length
        mask = torch.zeros(tokens_length)
        indices_target = torch.arange(tokens_length, dtype=torch.long)
        indices = torch.zeros(tokens_length, dtype=torch.long)

        mask[:] =1
        indices[:]=indices_target

        for name, module in self.unet.named_modules():
            module_name = type(module).__name__
            if module_name == "CrossAttention" and "attn2" in name:
                # module.last_attn_slice_mask = mask.to(self.device)
                # module.last_attn_slice_indices = indices.to(self.device)
                module.last_attn_slice_mask = None
                module.last_attn_slice_indices = None
            if module_name == "CrossAttention" and "attn1" in name:
                module.last_attn_slice_mask = None
                module.last_attn_slice_indices = None


    def init_attention_func(self, walk_left_right=False):

        # print ('device 1', self.device)
        @torch.no_grad()
        def new_attention(self, query, key, value, sequence_length, dim):

            # print ('query shape', query.shape)
            # print ('device 2:', query.device)
            # print ('device 3:', key.device)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.to(device)

            batch_size_attention = query.shape[0]
            hidden_states = torch.zeros(
                (batch_size_attention, sequence_length, dim // self.heads), device=device, dtype=query.dtype
            )
            # self._slice_size = 8
            slice_size = self._slice_size if self._slice_size is not None else hidden_states.shape[0]
            # print ('slice_size: ', slice_size)
            # print ('self._slice size: ', self._slice_size)
            # print ('sequence length: ', sequence_length)
            for i in range(hidden_states.shape[0] // slice_size):
                #print (i)
                start_idx = i * slice_size
                end_idx = (i + 1) * slice_size
                
                attn_slice = (
                    torch.einsum("b i d, b j d -> b i j", query[start_idx:end_idx], key[start_idx:end_idx]) * self.scale
                )
                #print ('attn_slice shape1: ', attn_slice.shape)
                attn_slice = attn_slice.softmax(dim=-1)
                if walk_left_right:
                    # print ('yes')
                    attn_slice_to_roll = attn_slice
                    query_shape = int(np.sqrt(attn_slice.shape[1]))
                    attn_slice_to_roll = attn_slice_to_roll.reshape((attn_slice.shape[0],query_shape,query_shape,-1))
                    torch.roll(attn_slice_to_roll,20, dims=(1))
                    attn_slice = attn_slice_to_roll.reshape((attn_slice.shape[0],-1,attn_slice.shape[2]))
                #print ('attn_slice shape2: ', attn_slice.shape)

                if self.use_last_attn_slice:
                    if self.last_attn_slice_mask is not None:
                        new_attn_slice = torch.index_select(self.last_attn_slice, -1, self.last_attn_slice_indices)
                        attn_slice = attn_slice * (1 - self.last_attn_slice_mask) + new_attn_slice * self.last_attn_slice_mask
                    else:
                        attn_slice = self.last_attn_slice       
                    self.use_last_attn_slice = False
                

                if self.save_last_attn_slice:
                    self.last_attn_slice = attn_slice
                    self.save_last_attn_slice = False

                if self.use_last_attn_weights and self.last_attn_slice_weights is not None:
                    attn_slice = attn_slice * self.last_attn_slice_weights
                    self.use_last_attn_weights = False

                # print ('attn_shape', attn_slice.shape)
                attn_slice = torch.einsum("b i j, b j d -> b i d", attn_slice, value[start_idx:end_idx])
                    
                # print ('1:', hidden_states.shape)
                # print ('2:', attn_slice.shape)
                hidden_states[start_idx:end_idx] = attn_slice

            # reshape hidden_states
            hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
            return hidden_states

        for name, module in self.unet.named_modules():
            module_name = type(module).__name__
            if module_name == "CrossAttention":
                module.last_attn_slice = None
                module.use_last_attn_slice = False
                module.use_last_attn_weights = False
                module.save_last_attn_slice = False
                module._attention = new_attention.__get__(module, type(module))

    def use_last_tokens_attention(self, use=True):
        for name, module in self.unet.named_modules():
            module_name = type(module).__name__
            if module_name == "CrossAttention" and "attn2" in name:
                module.use_last_attn_slice = use

    def use_last_tokens_attention_weights(self, use=True):
        for name, module in self.unet.named_modules():
            module_name = type(module).__name__
            if module_name == "CrossAttention" and "attn2" in name:
                module.use_last_attn_weights = use

    def use_last_self_attention(self, use=True):
        for name, module in self.unet.named_modules():
            module_name = type(module).__name__
            if module_name == "CrossAttention" and "attn1" in name:
                module.use_last_attn_slice = use

    def save_last_tokens_attention(self, save=True):
        for name, module in self.unet.named_modules():
            module_name = type(module).__name__
            if module_name == "CrossAttention" and "attn2" in name:
                module.save_last_attn_slice = save

    def save_last_self_attention(self, save=True):
        for name, module in self.unet.named_modules():
            module_name = type(module).__name__
            if module_name == "CrossAttention" and "attn1" in name:
                module.save_last_attn_slice = save

    @torch.no_grad()
    def stablediffusion_p2p(self, prompt="", prompt_edit=None, prompt_edit_token_weights=[], prompt_edit_tokens_start=0.0, prompt_edit_tokens_end=1.0,
                        prompt_edit_spatial_start=0.0, prompt_edit_spatial_end=1.0, guidance_scale=7.5, steps=50, seed=None, width=512, height=512,
                        latents=None, init_image=None, init_image_strength=0.65, t_start=0, prompt_embedding = None, prompt_edit_embedding = None, walk_left_right= False):

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)
        #Change size to multiple of 64 to prevent size mismatches inside model
        width = width - width % 64
        height = height - height % 64

        #If seed is None, randomly select seed from 0 to 2^32-1
        if seed is None: seed = random.randrange(2**32 - 1)
        generator = torch.cuda.manual_seed(seed)

        #Set inference timesteps to scheduler
        scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
        scheduler.set_timesteps(steps)

        print (prompt)
        print (prompt_edit)

        #Preprocess image if it exists (img2img)
        # if init_image is not None:
        #     #Resize and transpose for numpy b h w c -> torch b c h w
        #     init_image = get_init_image(init_image)
        #     init_image = init_image.resize((width, height), resample=Image.LANCZOS)
        #     init_image = np.array(init_image).astype(np.float32) / 255.0 * 2.0 - 1.0
        #     init_image = torch.from_numpy(init_image[np.newaxis, ...].transpose(0, 3, 1, 2))

        #     #If there is alpha channel, composite alpha for white, as the diffusion model does not support alpha channel
        #     if init_image.shape[1] > 3:
        #         init_image = init_image[:, :3] * init_image[:, 3:] + (1 - init_image[:, 3:])

        #     #Move image to GPU
        #     init_image = init_image.to(device)

        #     #Encode image
        #     with torch.autocast(device):
        #         init_latent = self.vae.encode(init_image).latent_dist.sample(generator=generator) * 0.18215

        #     t_start = steps - int(steps * init_image_strength)

        # if latents is None:
        # # else:
        #     init_latent = torch.zeros((1, self.unet.in_channels, height // 8, width // 8), device=device)
        #     t_start = 0

        noise = torch.randn(latents.shape, generator=generator, device=device)
        latents = scheduler.add_noise(latents, noise, t_start).to(device)
        

        #Process clip
        with torch.autocast(device):

            # print ('clip tokeniser max length', clip_tokenizer.model_max_length)
            tokens_unconditional = self.tokenizer("", padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
            embedding_unconditional = self.text_encoder(tokens_unconditional.input_ids.to(device)).last_hidden_state

            # print ('tokens_conditional_edit_shape', tokens_conditional.shape)
            # print ('embeddings_conditional_edit_shape', embedding_conditional.shape )

            #Process prompt editing
            if prompt_edit is not None:
                tokens_conditional = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
                embedding_conditional = self.text_encoder(tokens_conditional.input_ids.to(device)).last_hidden_state
                tokens_conditional_edit = self.tokenizer(prompt_edit, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
                embedding_conditional_edit = self.text_encoder(tokens_conditional_edit.input_ids.to(device)).last_hidden_state
                self.init_attention_edit(tokens_conditional, tokens_conditional_edit)
            elif prompt_embedding is not None:
                embedding_conditional= prompt_embedding
            if prompt_edit_embedding is not None:
                embedding_conditional_edit = prompt_edit_embedding
                self.init_attention_edit_for_pca_walk()
                
            # print ('latent_model_input ', latent_model_input.shape)
            # print ('embedding_conditional_edit ', embedding_conditional_edit.shape)
            # print ('embedding_conditional ', embedding_conditional.shape)

            self.init_attention_func(walk_left_right)
            self.init_attention_weights(prompt_edit_token_weights)

            timesteps = scheduler.timesteps[t_start:]

            for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
                t_index = t_start + i

                sigma = scheduler.sigmas[t_index]
                latent_model_input = latents
                latent_model_input = (latent_model_input / ((sigma**2 + 1) ** 0.5)).to(self.unet.dtype)

                #Predict the unconditional noise residual
                noise_pred_uncond = self.unet(latent_model_input, t, encoder_hidden_states=embedding_unconditional).sample

                #Prepare the Cross-Attention layers
                if prompt_edit is not None or prompt_edit_embedding is not None:
                    self.save_last_tokens_attention()
                    self.save_last_self_attention()
                else:
                    #Use weights on non-edited prompt when edit is None
                    self.use_last_tokens_attention_weights()

                #Predict the conditional noise residual and save the cross-attention layer activations
                noise_pred_cond = self.unet(latent_model_input, t, encoder_hidden_states=embedding_conditional).sample

                #Edit the Cross-Attention layer activations
                if prompt_edit is not None or prompt_edit_embedding is not None:
                    t_scale = t / scheduler.num_train_timesteps
                    if t_scale >= prompt_edit_tokens_start and t_scale <= prompt_edit_tokens_end:
                        self.use_last_tokens_attention()
                    if t_scale >= prompt_edit_spatial_start and t_scale <= prompt_edit_spatial_end:
                        self.use_last_self_attention()

                    #Use weights on edited prompt
                    self.use_last_tokens_attention_weights()

                    #Predict the edited conditional noise residual using the cross-attention masks
                    noise_pred_cond = self.unet(latent_model_input, t, encoder_hidden_states=embedding_conditional_edit).sample

                #Perform guidance
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                latents = scheduler.step(noise_pred, t_index, latents).prev_sample

            #scale and decode the image latents with vae
            latents = latents / 0.18215
            image = self.vae.decode(latents.to(self.vae.dtype)).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image[0] * 255).round().astype("uint8")
        return [Image.fromarray(image)]

    def prompt_token(self, prompt, index):
        tokens = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True).input_ids[0]
        return self.tokenizer.decode(tokens[index:index+1])

    def show_token_indices(self, prompt):
        tokens = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True).input_ids[0]
        for index in range(len(tokens)):
            decoded_token = self.tokenizer.decode(tokens[index:index+1])
            print(f'{index}:', decoded_token)
            if decoded_token == "<|endoftext|>":
                break


    @torch.no_grad()
    def diffusion_not_working(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        t_start: Optional[int] = 0,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        text_embeddings: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        **kwargs,
    ):
        if "torch_device" in kwargs:
            device = kwargs.pop("torch_device")
            warnings.warn(
                "`torch_device` is deprecated as an input argument to `__call__` and"
                " will be removed in v0.3.0. Consider using `pipe.to(torch_device)`"
                " instead."
            )

            # Set device as before (to be removed in 0.3.0)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)

        if text_embeddings is None:
            if isinstance(prompt, str):
                batch_size = 1
            elif isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                raise ValueError(
                    f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
                )

            if height % 8 != 0 or width % 8 != 0:
                raise ValueError(
                    "`height` and `width` have to be divisible by 8 but are"
                    f" {height} and {width}."
                )

            # get prompt text embeddings
            text_input = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        else:
            batch_size = text_embeddings.shape[0]

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            # max_length = text_input.input_ids.shape[-1]
            max_length = 77  # self.tokenizer.model_max_length
            uncond_input = self.tokenizer(
                [""] * batch_size,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(self.device)
            )[0]

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # get the initial random noise unless the user supplied it
        latents_shape = (batch_size, self.unet.in_channels, height // 8, width // 8)
        if latents is None:
            latents = torch.randn(
                latents_shape,
                generator=generator,
                device=self.device,
            )
        else:
            if latents.shape != latents_shape:
                raise ValueError(
                    f"Unexpected latents shape, got {latents.shape}, expected"
                    f" {latents_shape}"
                )
            latents = latents.to(self.device)

        # set timesteps
        accepts_offset = "offset" in set(
            inspect.signature(self.scheduler.set_timesteps).parameters.keys()
        )
        extra_set_kwargs = {}
        if accepts_offset:
            extra_set_kwargs["offset"] = 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        # if we use LMSDiscreteScheduler, let's make sure latents are mulitplied by sigmas
        if t_start==0 and isinstance(self.scheduler, LMSDiscreteScheduler):
            latents = latents * self.scheduler.sigmas[0]

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps[t_start:])):
            t_index = t_start + i
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                sigma = self.scheduler.sigmas[t_index]
                # the model input needs to be scaled to match the continuous ODE formulation in K-LMS
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            )["sample"]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # compute the previous noisy sample x_t -> x_t-1
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                latents = self.scheduler.step(
                    noise_pred, t_index, latents, **extra_step_kwargs
                )["prev_sample"]
            else:
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                )["prev_sample"]

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        safety_cheker_input = self.feature_extractor(
            self.numpy_to_pil(image), return_tensors="pt"
        ).to(self.device)
        image, has_nsfw_concept = self.safety_checker(
            images=image, clip_input=safety_cheker_input.pixel_values
        )

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        return {"sample": image, "nsfw_content_detected": has_nsfw_concept}

    def embed_text(self, text):
        """Helper to embed some text"""
        with torch.autocast("cuda"):
            text_input = self.tokenizer(
                text,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                embed = self.text_encoder(text_input.input_ids.to(self.device))[0]
        return embed

class StableDiffusionImageEmbedPipeline(DiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        image_encoder: CLIPModel,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.image_processor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda")     ## NOT SURE OF THE PRETRAINED MODEL
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")         ## NOT SURE OF THE PRETRAINED MODEL

    
class NoCheck(ModelMixin):
    """Can be used in place of safety checker. Use responsibly and at your own risk."""
    def __init__(self):
        super().__init__()
        self.register_parameter(name='asdf', param=torch.nn.Parameter(torch.randn(3)))

    def forward(self, images=None, **kwargs):
        return images, [False]
