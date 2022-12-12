
import torch
torch.cuda.empty_cache()
import numpy as np
import PIL
import subprocess
from pathlib import Path
from diffusers import LMSDiscreteScheduler

from src.stable_diffusion_pipeline import StableDiffusionPipeline, StableDiffusionImageEmbedPipeline
from src.utils import fit_pca, get_centered_text_embeddings, get_directional_text_embeddings_with_pca, get_init_image, pil_to_latent, set_torch_gpu, get_text_conditioned_embeddings, get_principal_components_hspace, getActivation, get_init_image
import os

def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """helper function to spherically interpolate two arrays v1 v2"""

    if not isinstance(v0, np.ndarray):
        inputs_are_torch = True
        input_device = v0.device
        v0 = v0.detach().cpu().numpy()
        v1 = v1.detach().cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2


def make_video_ffmpeg(frame_dir, output_file_name='output.mp4', frame_filename="frame%06d.png", fps=30):
    frame_ref_path = str(frame_dir / frame_filename)
    video_path = str(frame_dir / output_file_name)
    subprocess.call(
        f"ffmpeg -r {fps} -i {frame_ref_path} -vcodec libx264 -crf 10 -pix_fmt yuv420p"
        f" {video_path}".split()
    )
    return video_path


def walk(args):

    if not bool(args.resume):

        prompts= args.prompts
        seeds= args.seeds
        init_image =  args.init_image
        init_image_strength =  args.init_image_strength
        use_centered_text_conditionals =  args.use_centered_text_conditionals
        epsilon =  args.epsilon
        use_directional_text_conditionals =  args.use_directional_text_conditionals
        use_directional_text_conditionals_h_space = args.use_directional_text_conditionals_h_space
        use_p2p=  args.use_p2p
        use_p2p_directional = args.use_p2p_directional
        component_ind =  args.component_ind
        alphas =  args.alphas
        num_steps=  args.num_steps
        output_dir=  args.output_dir
        name= str(seeds)
        height= args.height
        width= args.width
        guidance_scale= args.guidance_scale
        eta= args.eta
        num_inference_steps= args.num_inference_steps
        do_loop= args.do_loop
        make_video= args.make_video
        use_lerp_for_text= args.use_lerp_for_text
        scheduler= args.scheduler  # choices: default, ddim, klms
        disable_tqdm= args.disable_tqdm
        upsample= args.upsample
        resume= args.resume
        fps= args.fps
        less_vram= args.less_vram
        batch_size= args.batch_size
        frame_filename_ext= args.frame_filename_ext
        gpuid = args.gpuid
        walk_left_right= args.walk_left_right
        train_on= args.train_on
        use_image_embeddings_only = args.use_image_embeddings_only

    auth_token = "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"

    set_torch_gpu(gpuid)
    if use_image_embeddings_only:
      # pipeline = StableDiffusionImageEmbedPipeline.from_pretrained(
      pipeline = StableDiffusionPipeline.from_pretrained(
      "lambdalabs/sd-image-variations-diffusers",
      revision="273115e88df42350019ef4d628265b8c29ef4af5"
      ).to("cuda")
    else:
      pipeline= StableDiffusionPipeline.from_pretrained(
          "CompVis/stable-diffusion-v1-4",
          use_auth_token=auth_token,
          torch_dtype=torch.float16,
          revision="fp16",
      ).to("cuda")

    output_path = Path(output_dir) / name
    output_path.mkdir(exist_ok=True, parents=True)

    pipeline.scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    try:
      assert len(prompts) == len(seeds)
    except:
      seeds = [seeds[0] for _ in range(len(prompts))]
    first_seed, *seeds = seeds

    generator=torch.Generator(device=pipeline.device).manual_seed(first_seed)
    if not init_image:
      latents_a = torch.randn(
          (1, pipeline.unet.in_channels, height // 8, width // 8),
          device=pipeline.device,
          generator=generator,
      )
      t_start = 0   
      input_image = None
    else:
      input_image= get_init_image(init_image)
      # if use_image_embeddings_only:
      #   t_start = 0
      #   latents_a = torch.randn(
      #     (1, pipeline.unet.in_channels, height // 8, width // 8),
      #     device=pipeline.device,
      #     generator=generator,
      #   )
      # else:
      init_latent = pil_to_latent(pipeline, init_image, seed=first_seed)
      t_start = num_inference_steps - int(num_inference_steps * init_image_strength)
      noise = torch.randn(init_latent.shape, generator=generator, device=pipeline.device)
      latents_a = pipeline.scheduler.add_noise(init_latent, noise, t_start).to(pipeline.device)
      
    
    # print (prompts)
    first_prompt, *prompts = prompts
    embeds_a = pipeline.embed_text(first_prompt)
    # print (torch.norm(embeds_a))
    nx, ny, nz = embeds_a.shape
    if use_centered_text_conditionals:
      embeddings, _ = get_centered_text_embeddings(embeds_a, epsilon, nx=nx, ny=ny, nz=nz)
    elif use_directional_text_conditionals:
      # print (init_image_strength)
      text_prompts = np.load(train_on)
      text_embeddings = get_text_conditioned_embeddings(pipeline, text_prompts)
      pca_model = fit_pca(text_embeddings)
      embeddings, _ = get_directional_text_embeddings_with_pca(pca_model, nx, ny, nz, embeds_a, alphas, component_ind = component_ind)
    elif use_directional_text_conditionals_h_space:  
      pipeline.unet.mid_block.register_forward_hook(getActivation('mid_block_last_layer'))
      if ( init_image is None and not os.path.exists('eigen_vectors_h_space.npy')) or (init_image is not None and not os.path.exists('eigen_vectors_inits_h_space.npy')):
  
        eigen_vectors = get_principal_components_hspace(pipeline, generator, latents_a, t_start = t_start)
        if init_image is None: 
          np.save('eigen_vectors_h_space.npy', eigen_vectors)
        else:
          np.save('eigen_vectors_inits_h_space.npy', eigen_vectors)
        # print (eigen_vectors.shape)
      else:
        eigen_vectors = np.load('eigen_vectors_h_space.npy') if init_image is None else np.load('eigen_vectors_inits_h_space.npy')
      eigen_vectors = torch.Tensor(eigen_vectors).cuda()
      # print (eigen_vectors.shape)
      embeddings  = [embeds_a for i in range(len(alphas))]

    else:
      prompt_a = first_prompt
      embeddings = []
      if not prompts:
        prompts.append(first_prompt)
        seeds.append(first_seed)
      for prompt in prompts:
        embeddings.append(pipeline.embed_text(prompt))    

      # print (seeds)

    if do_loop:
        prompts.append(first_prompt)
        seeds.append(first_seed)

    frame_index = 0

    for j,embeds_b in enumerate(embeddings):

        # Latent Noise
        # if not use_centered_text_conditionals and not use_directional_text_conditionals and not use_p2p and not use_directional_text_conditionals_h_space:
        #   latents_b = torch.randn(
        #       (1, pipeline.unet.in_channels, height // 8, width // 8),
        #       device=pipeline.device,
        #       generator=torch.Generator(device=pipeline.device).manual_seed(seeds[j]),
        #   )
        
        latents_b = latents_a
        if use_directional_text_conditionals_h_space:
          # eigen_vector = alphas[j]*eigen_vectors[component_ind: component_ind+1]
          eigen_vector = 1*eigen_vectors[0]
          pipeline.unet.mid_block.register_forward_hook(getActivation('mid_block_last_layer', eigen_vector))
        

        latents_batch, embeds_batch = None, None
        for i, t in enumerate(np.linspace(0, 1, num_steps)):

            frame_filepath = output_path / (f"frame%06d{frame_filename_ext}" % frame_index)
            if resume and frame_filepath.is_file():
                frame_index += 1
                continue

            # print (embeds_a.shape)
            if use_lerp_for_text:
                embeds = torch.lerp(embeds_a, embeds_b, float(t))
            else:
                embeds = slerp(float(t), embeds_a, embeds_b)
            # print (embeds.shape)
            # print (latents_a.shape)
            latents = slerp(float(t), latents_a, latents_b)
            # print (latents.shape)

            embeds_batch = embeds if embeds_batch is None else torch.cat([embeds_batch, embeds])
            latents_batch = latents if latents_batch is None else torch.cat([latents_batch, latents])

            del embeds
            del latents
            torch.cuda.empty_cache()

            batch_is_ready = embeds_batch.shape[0] == batch_size or t == 1.0
            if not batch_is_ready:
                continue

            do_print_progress = (i == 0) or ((frame_index) % 20 == 0)
            if do_print_progress:
                print(f"COUNT: {frame_index}/{len(embeddings)*num_steps}")

            
            with torch.autocast("cuda"):
                
                if use_p2p:
                  # print ('yes')
                  outputs = pipeline.stablediffusion_p2p(
                    init_image = init_image,
                    latents=latents_batch, 
                    t_start = t_start, 
                    #prompt_embedding= embeds_a, 
                    seed=first_seed, 
                    #prompt_edit_embedding = embeds_batch,
                    walk_left_right = walk_left_right,
                    prompt = prompt_a,
                    prompt_edit = prompts[j]
                )
                elif use_p2p_directional:
                  outputs = pipeline.stablediffusion_p2p(
                    init_image = init_image,
                    latents=latents_batch, 
                    t_start = t_start, 
                    prompt_embedding= embeds_a, 
                    seed=first_seed, 
                    prompt_edit_embedding = embeds_batch,
                    walk_left_right = walk_left_right
                )
                # elif use_image_embeddings_only:
                #   outputs = pipeline(
                #       latents=latents_batch, 
                #       input_image=input_image,
                #       height=height,
                #       width=width,
                #       guidance_scale=guidance_scale,
                #       num_inference_steps=num_inference_steps,
                #       generator = generator,
                #       output_type='pil'

                #   )
                else:
                  outputs = pipeline(
                      latents=latents_batch,
                      t_start = t_start,
                      text_embeddings=embeds_batch,
                      height=height,
                      width=width,
                      guidance_scale=guidance_scale,
                      num_inference_steps=num_inference_steps,
                      generator = generator,
                      output_type='pil' if not upsample else 'numpy',
                      input_image=input_image,
                      use_image_embeddings_only=use_image_embeddings_only
                  )

                # print (activation['mid_block_last_layer'])
                del embeds_batch
                del latents_batch
                torch.cuda.empty_cache()
                latents_batch, embeds_batch = None, None

                images = outputs

            for image in images:
                frame_filepath = output_path / (f"frame%06d{frame_filename_ext}" % frame_index)
                image.save(frame_filepath)
                frame_index += 1

        embeds_a = embeds_b
        latents_a = latents_b
        prompt_a = prompt_a

    if make_video:
        vid = make_video_ffmpeg(output_path, f"{name}.mp4", fps=fps, frame_filename=f"frame%06d{frame_filename_ext}")
        output_path_files = os.listdir(output_path)

        # for item in output_path_files:
        #     if not item.endswith(".mp4"):
        #         os.remove(os.path.join(output_path, item))

        return vid
