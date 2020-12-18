cuda=1

CUDA_VISIBLE_DEVICES=$cuda python3 -m tools.train_net --num-gpus 1 \
        --config-file ./configs/FOOD-detection/faster_rcnn_R_101_FPN_ft_meta.yaml