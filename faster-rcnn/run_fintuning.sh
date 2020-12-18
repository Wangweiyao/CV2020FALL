cuda=1
data=all

for k_shot in 1 3 5 10
do
        CUDA_VISIBLE_DEVICES=$cuda python3 -m tools.train_net --num-gpus 1 \
                --config-file ./configs/FOOD-detection/faster_rcnn_R_101_FPN_ft_${k_shot}shot.yaml
done