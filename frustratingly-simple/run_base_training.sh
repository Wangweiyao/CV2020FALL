cuda=0
data=all

CUDA_VISIBLE_DEVICES=0 python3 -m tools.train_net --num-gpus 1 \
        --config-file ./configs/FOOD-detection/faster_rcnn_R_101_FPN_all.yaml &

CUDA_VISIBLE_DEVICES=1 python3 -m tools.train_net --num-gpus 1 \
        --config-file ./configs/FOOD-detection/faster_rcnn_R_101_FPN_base_no_aug.yaml 

CUDA_VISIBLE_DEVICES=$cuda python3 -m tools.train_net --num-gpus 1 \
        --config-file ./configs/FOOD-detection/faster_rcnn_R_101_FPN_ft_${k_shot}shot.yaml

CUDA_VISIBLE_DEVICES=$cuda python3 -m tools.train_net --num-gpus 1 \
        --config-file ./configs/FOOD-detection/faster_rcnn_R_101_FPN_ft_${k_shot}shot.yaml
#CUDA_VISIBLE_DEVICES=$cuda python3 -m tools.train_net --num-gpus 1 \
#        --config-file ./configs/FOOD-detection/faster_rcnn_R_50_FPN_${data}.yaml 
#CUDA_VISIBLE_DEVICES=$cuda python3 -m tools.train_net --num-gpus 1 \
#        --config-file ./configs/FOOD-detection/faster_rcnn_R_50_FPN_${data}_3x.yaml &
#CUDA_VISIBLE_DEVICES=$cuda python3 -m tools.train_net --num-gpus 1 \
#        --config-file ./configs/FOOD-detection/faster_rcnn_R_50_FPN_${data}_3y.yaml &
