import torch
import numpy as np
from sklearn.decomposition import PCA
from numpy import linalg, newaxis, random

import os, requests
from PIL import Image
from io import BytesIO

# from src.walk import pipeline

device = "cuda"

activation = {}
def getActivation(name, eigen_vector= None):
  # the hook signature
  def hook(model, input, output):
    activation[name] = output.squeeze().detach()
    if eigen_vector is not None:
      output+= eigen_vector
  return hook

def set_torch_gpu(gpuid=1):
    device = torch.cuda.set_device(gpuid)

def get_init_image(url):
    response = requests.get(url)
    init_img = Image.open(BytesIO(response.content))
    return init_img

@torch.no_grad()
def pil_to_latent(pipeline, init_image, height=512, width=512, seed=33):

    generator = torch.Generator(device=pipeline.device).manual_seed(seed)

    init_image = get_init_image(init_image)
    init_image = init_image.resize((width, height), resample=Image.LANCZOS)
    init_image = np.array(init_image) / 255.0 * 2.0 - 1.0
    init_image = torch.Tensor(init_image[np.newaxis, ...].transpose(0, 3, 1, 2))

    #If there is alpha channel, composite alpha for white, as the diffusion model does not support alpha channel
    if init_image.shape[1] > 3:
        init_image = init_image[:, :3] * init_image[:, 3:] + (1 - init_image[:, 3:])

    #Move image to GPU
    init_image = init_image.to(pipeline.device)
    #print (init_image.shape)

    #Encode image
    with torch.autocast("cuda"):
        init_latent = pipeline.vae.encode(init_image).latent_dist.sample(generator=generator) * 0.18215

    return init_latent


def latents_to_pil(pipeline, latents):
    # bath of latents -> list of images
    
    latents = (1 / 0.18215) * latents
    latents = torch.Tensor(latents)
    # with torch.no_grad():
    with torch.autocast("cuda"):
        image = pipeline.vae.decode(latents).sample
    image = ((image / 2) + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    pil_images[0].show()
    return pil_images


def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.Tensor(image)
    return 2.0 * image - 1.0


def get_text_conditioned_embeddings(pipeline, prompts):

  batch_size = 1
  if isinstance(prompts, str):
    prompts = [prompts]

  embeddings = []
  for i,prompt in enumerate(prompts):
    text_input = pipeline.tokenizer(prompt, padding="max_length", max_length= pipeline.tokenizer.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
      text_embeddings = pipeline.text_encoder(text_input.input_ids.to(device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = pipeline.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    with torch.no_grad():
      uncond_embeddings = pipeline.text_encoder(uncond_input.input_ids.to(device))[0]
    #text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    # if (i==0):
    #   print (text_embeddings)
    embeddings.append(text_embeddings)

  return embeddings


def fit_pca(embeddings, n_components=47):
  # text_prompts = np.load(train_on)
  # text_embeddings = get_text_conditioned_embeddings(pipeline, text_prompts)
  nx, ny, nz = embeddings[0].shape
  # print (nx, ny, nz)
  embeddings = [embedding.cpu().tolist() for embedding in embeddings]

  nsamples = len(embeddings)
  X = np.array(embeddings).reshape((nsamples,-1))
  X = (X - np.min(X))/(np.max(X) - np.min(X))
  # pca_plots(X)

  pcamodel = PCA(n_components)
  pca = pcamodel.fit(X)
  return pca

def get_principal_components(pca, nx, ny, nz, number=5):

  prinicipal_components = []
  # video_writer_names_for_components = []
  for i in range(number):
    # video_writer_names_for_components.append('video_of_principal_component_' + str(i+1))
    component = pca.components_[i].reshape((nx, ny, nz)).astype(np.double)
    prinicipal_components.append(torch.from_numpy(component).to(device).to(dtype=torch.float32))
  print (torch.norm(prinicipal_components[0]))
  return prinicipal_components

def get_principal_components_hspace(pipeline, generator, latents, train_on = 'annotations_car.npy', t_start=0, height=512, width=512, guidance_scale= 7.5, num_inference_steps=40, output_type='numpy', upsample=False):
  
  text_prompts = np.load(train_on)
  text_prompts = ['the color of the car is red']
  # text_embeddings = get_text_conditioned_embeddings(pipeline, text_prompts)
  embeddings = []
  print (text_prompts)
  for i,prompt in enumerate(text_prompts):
    print ('----getting prompt: ' + str(i) + ' -------')
    text_embedding = pipeline.embed_text(prompt)
    outputs = pipeline(
                  latents=latents,
                  t_start = t_start,
                  text_embeddings=text_embedding,
                  height=height,
                  width=width,
                  guidance_scale=guidance_scale,
                  num_inference_steps=num_inference_steps,
                  generator = generator,
                  output_type='pil' if not upsample else 'numpy'
              )
    
    embeddings.append(activation['mid_block_last_layer'].cpu().numpy())
  embeddings = np.array(embeddings).astype('float64')
  return embeddings
  print ('embeddings shape: ', embeddings.shape)
  eigen_vectors = get_eigen_vectors(embeddings)
  return eigen_vectors

def get_eigen_vectors(embeddings):
  
  embeddings = embeddings / np.linalg.norm(embeddings, axis=0, keepdims=True)
  # print (embeddings.shape)
  embeddings = np.einsum('abcd,abcd->abcd', embeddings,embeddings)
  eigen_values, eigen_vectors = np.linalg.eig(embeddings)
  print ('eigen_values: ',eigen_values.shape)
  e_indices = np.argsort(eigen_values)[::-1]
  eigen_vectors = eigen_vectors[:,e_indices]
  print ('eigen_vectors: ', eigen_vectors.shape)
  return eigen_vectors
  

def get_directional_text_embeddings_with_pca(pca_model, nx, ny, nz, text_embedding, alphas=[0.5], component_ind=2):

  print ('text embedding', text_embedding.shape)
  prinicipal_components = get_principal_components(pca= pca_model, nx=nx, ny=ny, nz=nz)

  directional_text_embeddings = []
  captions = []
  for alpha in alphas:
    directional_text_embeddings.append(torch.sum(torch.stack([text_embedding, alpha*torch.norm(text_embedding)*prinicipal_components[component_ind]]), dim=0))
    captions.append('principal component: '+str(component_ind+1)+ '\n with alpha: ' + str(alpha))

  return directional_text_embeddings, captions

def gen_rand_vecs(number=50, nx=1, ny=77, nz=768):

    from numpy import random
    vecs = random.normal(size=(number,nx,ny,nz))
    video_writer_names_for_vectors = [ 'video_of_unit_vector_'+str(i+1) for i in range(number) ]
    mags = linalg.norm(vecs, axis=-1)

    return vecs / mags[..., newaxis]

def get_centered_text_embeddings(text_embedding, epsilon, vector_ind= 0, nx=1, ny=77, nz=768):

  text_embedding = text_embedding.tolist()

  unit_vector_directions = gen_rand_vecs(nx = nx, ny=ny, nz= nz)

  centered_text_embeddings = []
  captions = []
  for unit_vector in unit_vector_directions:
    centered_text_embedding = text_embedding + (epsilon*unit_vector)
    centered_text_embedding = torch.Tensor(centered_text_embedding).to(device)
    centered_text_embeddings.append(centered_text_embedding)
    captions.append('epsilon: '+str(epsilon)+ ' with \n unit vector direction: ' + str(vector_ind+1))

  return centered_text_embeddings, captions
