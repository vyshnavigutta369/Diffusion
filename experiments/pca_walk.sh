
prompt=("a red car")
init_image='https://media.istockphoto.com/photos/kia-rio-picture-id1273404862?k=20&m=1273404862&s=612x612&w=0&h=P4BwD_I3qYIKCkgUPnG-sflREqygiP6K5Ne15W4XhRQ='
init_image_strength=0.5

# num_inference_steps=50
# batch_size=1
num_steps=6

root_outdir=outputs/pca_walk
mkdir -p $root_outdir

CONFIG=configs/config.yaml

use_directional_text_conditionals=True
alphas=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8)
# alphas=(1)
component_inds=(0 1 2 3 4)
# component_inds=(0)
# use_centered_text_conditionals=True

make_video=True
seeds=(33 42 56)

for ((i=0;i<${#seeds[@]};++i))
do
    seed=(${seeds[i]})
    
    for ((j=0;j<${#component_inds[@]};++j))
    do
        component_ind="${component_inds[j]}"
        epsilon=(${seeds[i]})
        
        python -u main.py  --config ${CONFIG} --prompts "${prompt[@]}" --seeds $seed \
            --use_directional_text_conditionals $use_directional_text_conditionals --alphas "${alphas[@]}" --component_ind $component_ind\
            --init_image $init_image --init_image_strength $init_image_strength \
            --num_steps $num_steps \
            --make_video $make_video --output_dir ${root_outdir}/walk_videos_pca_component_${j}_seed_${seed}_init_color_test \
            --gpuid 2

        # python -u main.py  --config ${CONFIG} --prompts "${prompt[@]}" --seeds $seed \
        #     --use_directional_text_conditionals $use_directional_text_conditionals --alphas "${alphas[@]}" --component_ind $component_ind\
        #     --num_steps $num_steps \
        #     --make_video $make_video --output_dir ${root_outdir}/walk_videos_pca_component_${j}_seed_${seed}_color \
        #     --gpuid 2
            
    done
done