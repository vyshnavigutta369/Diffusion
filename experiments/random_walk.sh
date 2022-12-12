
prompt=("a car parked near a tree")
init_image='https://media.istockphoto.com/photos/kia-rio-picture-id1273404862?k=20&m=1273404862&s=612x612&w=0&h=P4BwD_I3qYIKCkgUPnG-sflREqygiP6K5Ne15W4XhRQ='
init_image_strength=0.65

# num_inference_steps=50
# batch_size=1
# num_steps=5

root_outdir=outputs/random_walk
mkdir -p $root_outdir

CONFIG=configs/config.yaml

use_centered_text_conditionals=True

make_video=True
seeds=(33 42 56)
epsilons=(0.1 0.2 0.3 0.4 0.5 0.6)



for ((i=0;i<${#seeds[@]};++i))
do
    seed=(${seeds[i]})
    #seed=($seed)
    for ((j=0;j<${#epsilons[@]};++j))
    do
        epsilon=${epsilons[j]}
        python -u main.py  --config ${CONFIG} --prompts "${prompt[@]}" --seeds $seed \
            --use_centered_text_conditionals $use_centered_text_conditionals --epsilon $epsilon \
            --init_image $init_image --init_image_strength $init_image_strength \
            --make_video $make_video --output_dir ${root_outdir}/walk_videos_centered_e_${epsilon}_seed_${seed}_init_test \
            --gpuid 0

        python -u main.py  --config ${CONFIG} --prompts "${prompt[@]}" --seeds $seed \
            --use_centered_text_conditionals $use_centered_text_conditionals --epsilon $epsilon \
            --make_video $make_video --output_dir ${root_outdir}/walk_videos_centered_e_${epsilon}_seed_${seed} \
            --gpuid 0
            
    done
done