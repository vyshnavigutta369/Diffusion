
prompt=("a red car")
init_image='https://media.istockphoto.com/photos/kia-rio-picture-id1273404862?k=20&m=1273404862&s=612x612&w=0&h=P4BwD_I3qYIKCkgUPnG-sflREqygiP6K5Ne15W4XhRQ='
init_image_strength=0.6

# num_inference_steps=50
# batch_size=1
num_steps=1

root_outdir=outputs/image_variations
mkdir -p $root_outdir


CONFIG=configs/config.yaml


make_video=True
seeds=(33 42 56)

for ((i=0;i<${#seeds[@]};++i))
do
    seed=(${seeds[i]})
    use_image_embeddings_only=False
    python -u main.py  --config ${CONFIG} --prompts "${prompt[@]}" --seeds $seed \
        # --init_image $init_image --init_image_strength $init_image_strength \
        --num_steps $num_steps \
        --use_image_embeddings_only $use_image_embeddings_only \
        --make_video $make_video --output_dir ${root_outdir}/walk_videos_seed_${seed}_init\
        --gpuid 2
done