gpu=0

CUDA_VISIBLE_DEVICES=$gpu python code/evaluate.py \
    --name dif-net \
    --epoch 400 \
    --dst_list knee_zhao \
    --split test \
    --num_views 10
