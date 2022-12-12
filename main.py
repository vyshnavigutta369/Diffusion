from __future__ import absolute_import
import torch
import numpy as np
import os
import sys
import argparse
import yaml
import json
from src.walk import walk

def create_args():
    
    # This function prepares the variables shared across demo.py
    parser = argparse.ArgumentParser()

    parser.add_argument('--use_centered_text_conditionals', type=bool, default=False, help="use_centered_text_conditionals")
    parser.add_argument('--use_p2p', type=bool, default=False, help="use_p2p")
    parser.add_argument('--use_p2p_directional', type=bool, default=False, help="use_p2p_directional")
    parser.add_argument('--use_directional_text_conditionals', type=bool, default=False, help="use_directional_text_conditionals")
    parser.add_argument('--use_directional_text_conditionals_h_space', type=bool, default=False, help="use_directional_text_conditionals_h_space")
    parser.add_argument('--disable_tqdm', type=bool, default=False, help="disable_tqdm")
    parser.add_argument('--make_video', type=bool, default=False, help="make_video")
    parser.add_argument('--use_lerp_for_text', type=bool, default=False, help="use_lerp_for_text")
    parser.add_argument('--do_loop', type=bool, default=False, help="do_loop")
    parser.add_argument('--upsample', type=bool, default=False, help="upsample")
    parser.add_argument('--less_vram', type=bool, default=False, help="less_vram")
    parser.add_argument('--resume', type=bool, default=False, help="resume")
    parser.add_argument('--walk_left_right', type=bool, default=False, help="walk_left_right")
    parser.add_argument('--use_image_embeddings_only', type=bool, default=False, help="use_image_embeddings_only")
    

    parser.add_argument('--output_dir', type=str, default='outputs/output', help="output_dir")
    parser.add_argument('--scheduler', type=str, default='default', help="scheduler")
    parser.add_argument('--init_image', type=str, default=None, help="path to init image")
    parser.add_argument('--frame_filename_ext', type=str, default='.png', help="frame_filename_ext")

    parser.add_argument('--height', type=int, default=512, help="height")
    parser.add_argument('--width', type=int, default=512, help="width")
    parser.add_argument('--batch_size', type=int, default=1, help="batch_size")
    parser.add_argument('--num_steps', type=int, default=1, help="num_steps")
    parser.add_argument('--num_inference_steps', type=int, default=40, help="num_inference_steps")
    parser.add_argument('--component_ind', type=int, default=0, help="component_ind")
    parser.add_argument('--fps', type=int, default=25, help="fps")
    parser.add_argument('--gpuid', type=int, default=1, help="gpuid")

    parser.add_argument('--epsilon', type=float, default=0.0,  help="epsilon")
    parser.add_argument('--guidance_scale', type=float, default=8.5,  help="guidance_scale")
    parser.add_argument('--init_image_strength', type=float, default=8.5,  help="init_image_strength")
    
    
    parser.add_argument('--seeds', nargs='+', help='<Required> Set flag', required=True, type=int)
    parser.add_argument('--prompts', nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument('--alphas', nargs='+', help='alphas', type=float, default=[0.1,0.2,0.3,0.4,0.5,0.6])

    parser.add_argument('--config', type=str, default="configs/config.yaml", help="yaml experiment config input")


    return parser

def get_args(argv):
    parser=create_args()
    args = parser.parse_args(argv)
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    config.update(vars(args))
    return argparse.Namespace(**config)


if __name__ == '__main__':
    args = get_args(sys.argv[1:])
    walk(args)

    # seeds = [32, 44, 56]
    # for j in range(5):
    #   for i,seed in enumerate(seeds):
    #     walk (
    #         prompts=['a grey car on a road'],
    #         seeds=[seed],
    #         init_image = None, 
    #         use_p2p = True, 
    #         # pca_model = None, 
    #         component_ind = j,
    #         output_dir='walk_videos_component_'+str(j)+'_seed_'+str(i),     # Where images/videos will be saved  # 'walk_videos_centered_seed_'+str(i)+'_e_'+str(epsilon)
    #         name=str(seed),     # Subdirectory of output_dir where images/videos will be saved
    #         guidance_scale=8.5,      # Higher adheres to prompt more, lower lets model take the wheel
    #         num_steps=1,             # Change to 60-200 for better results...3-5 for testing
    #         num_inference_steps=40, 
    #         scheduler='default',        # One of: "klms", "default", "ddim"
    #         disable_tqdm=False,      # Set to True to disable tqdm progress bar
    #         make_video=True,         # If false, just save images
    #         use_lerp_for_text=False,  # Use lerp for text embeddings instead of slerp
    #         do_loop=False           # Change to True if you want last prompt to loop back to first prompt
    #     )

  
        

