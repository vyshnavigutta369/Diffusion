
prompt=("a grey car" "a red car" "a blue car" "a yellow car")
init_image='https://media.istockphoto.com/photos/kia-rio-picture-id1273404862?k=20&m=1273404862&s=612x612&w=0&h=P4BwD_I3qYIKCkgUPnG-sflREqygiP6K5Ne15W4XhRQ='
init_image_strength=1

# num_inference_steps=50

root_outdir=outputs/p2p_walk
mkdir -p $root_outdir

CONFIG=configs/config.yaml

use_p2p=True
num_steps=1

make_video=True
seeds=(33 42 56)

for ((i=0;i<${#seeds[@]};++i))
do
    seed=(${seeds[i]})
    #seed=($seed)
    
    python -u main.py  --config ${CONFIG} --prompts "${prompt[@]}" --seeds $seed \
        --use_p2p $use_p2p  \
        --init_image $init_image --init_image_strength $init_image_strength \
        --num_steps $num_steps \
        --make_video $make_video --output_dir ${root_outdir}/p2p_walk_seed_${seed}_init \
        --gpuid 1

    python -u main.py  --config ${CONFIG} --prompts "${prompt[@]}" --seeds $seed \
        --use_p2p $use_p2p \
        --num_steps $num_steps \
        --make_video $make_video --output_dir ${root_outdir}/p2p_walk_seed_${seed} \
        --gpuid 1
        
done


