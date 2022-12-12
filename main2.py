from src.walk import walk

seeds = [32, 44, 56]
# pca_model = fit_pca()

for i,seed in enumerate(seeds):
  
  walk(
      prompts=['a grey car on a road'],
      seeds=[seed],
      init_image = 'https://media.istockphoto.com/photos/kia-rio-picture-id1273404862?k=20&m=1273404862&s=612x612&w=0&h=P4BwD_I3qYIKCkgUPnG-sflREqygiP6K5Ne15W4XhRQ=', 
      # init_image= None, 
      # use_centered_text_conditionals = True, 
      use_p2p = True,
      component_ind = i,
      output_dir='walk_videos_centered_'+str(i),     # Where images/videos will be saved
      name=str(seed),     # Subdirectory of output_dir where images/videos will be saved
      guidance_scale=8.5,      # Higher adheres to prompt more, lower lets model take the wheel
      num_steps=1,             # Change to 60-200 for better results...3-5 for testing
      num_inference_steps=40, 
      scheduler='default',        # One of: "klms", "default", "ddim"
      disable_tqdm=False,      # Set to True to disable tqdm progress bar
      make_video=True,         # If false, just save images
      use_lerp_for_text=False,  # Use lerp for text embeddings instead of slerp
      do_loop=False           # Change to True if you want last prompt to loop back to first prompt
  )