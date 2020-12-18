CUDA_VISIBLE_DEVICES=0 python3 -m tools.train_net --num-gpus 1 \
        --config-file ./configs/FOOD-detection/faster_rcnn_R_101_FPN_all.yaml 

#CUDA_VISIBLE_DEVICES=1 python3 -m tools.train_net --num-gpus 1 \
#        --config-file ./configs/FOOD-detection/faster_rcnn_R_101_FPN_base.yaml 

CUDA_VISIBLE_DEVICES=0 python3 -m tools.train_net --num-gpus 1 \
        --config-file ./configs/FOOD-detection/faster_rcnn_R_101_FPN_ft_meta.yaml &

CUDA_VISIBLE_DEVICES=1 python3 -m tools.train_net --num-gpus 1 \
        --config-file ./configs/FOOD-detection/faster_rcnn_R_101_FPN_ft_1shot.yaml &

CUDA_VISIBLE_DEVICES=1 python3 -m tools.train_net --num-gpus 1 \
        --config-file ./configs/FOOD-detection/faster_rcnn_R_101_FPN_ft_3shot.yaml &

CUDA_VISIBLE_DEVICES=0 python3 -m tools.train_net --num-gpus 1 \
        --config-file ./configs/FOOD-detection/faster_rcnn_R_101_FPN_ft_5shot.yaml 

CUDA_VISIBLE_DEVICES=1 python3 -m tools.train_net --num-gpus 1 \
        --config-file ./configs/FOOD-detection/faster_rcnn_R_101_FPN_ft_10shot.yaml &

CUDA_VISIBLE_DEVICES=0 python3 -m tools.train_net --num-gpus 1 \
        --config-file ./configs/FOOD-detection/faster_rcnn_R_101_FPN_ft_cos_1shot.yaml &

CUDA_VISIBLE_DEVICES=1 python3 -m tools.train_net --num-gpus 1 \
        --config-file ./configs/FOOD-detection/faster_rcnn_R_101_FPN_ft_cos_3shot.yaml &

CUDA_VISIBLE_DEVICES=0 python3 -m tools.train_net --num-gpus 1 \
        --config-file ./configs/FOOD-detection/faster_rcnn_R_101_FPN_ft_cos_5shot.yaml &

CUDA_VISIBLE_DEVICES=1 python3 -m tools.train_net --num-gpus 1 \
        --config-file ./configs/FOOD-detection/faster_rcnn_R_101_FPN_ft_cos_10shot.yaml 

