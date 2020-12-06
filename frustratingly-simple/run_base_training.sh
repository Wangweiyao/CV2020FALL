cuda=0
data=all

CUDA_VISIBLE_DEVICES=$cuda python3 -m tools.train_net --num-gpus 1 \
        --config-file ./configs/FOOD-detection/faster_rcnn_R_101_FPN_${data}_no_aug.yaml 
#CUDA_VISIBLE_DEVICES=$cuda python3 -m tools.train_net --num-gpus 1 \
#        --config-file ./configs/FOOD-detection/faster_rcnn_R_50_FPN_${data}.yaml 
#CUDA_VISIBLE_DEVICES=$cuda python3 -m tools.train_net --num-gpus 1 \
#        --config-file ./configs/FOOD-detection/faster_rcnn_R_50_FPN_${data}_3x.yaml &
#CUDA_VISIBLE_DEVICES=$cuda python3 -m tools.train_net --num-gpus 1 \
#        --config-file ./configs/FOOD-detection/faster_rcnn_R_50_FPN_${data}_3y.yaml &
