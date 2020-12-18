cuda=0

for name in ft_extract_features 
do 
        CUDA_VISIBLE_DEVICES=$cuda python3 -m tools.extract_features --num-gpus 1 \
                --config-file ./configs/FOOD-detection/faster_rcnn_R_101_FPN_${name}.yaml 
done