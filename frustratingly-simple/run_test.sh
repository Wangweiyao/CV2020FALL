#CUDA_VISIBLE_DEVICES=1 python3 -m tools.food_test_net --num-gpus 1 \
#        --config-file ./checkpoints/food/faster_rcnn/faster_rcnn_R_101_FPN_all/config.yaml \
#        --model-path ./checkpoints/food/faster_rcnn/faster_rcnn_R_101_FPN_all/model_final.pth \
#       --test-name final \
#        --eval-only 

cuda=0
name=ft_extract_features 
for model_id in 10shot #1shot 3shot 5shot 
do 
        CUDA_VISIBLE_DEVICES=$cuda python3 -m tools.food_test_net --num-gpus 1 \
                --config-file ./checkpoints/food/faster_rcnn/faster_rcnn_R_101_FPN_${name}/config.yaml \
                --model-path ./checkpoints/food/faster_rcnn/faster_rcnn_R_101_FPN_${name}/${model_id}.pth \
                --test-name ${model_id} \
                --eval-only 
done

exit

cuda=0

for name in ft_meta base ft_1shot ft_3shot ft_5shot ft_10shot ft_cos_1shot ft_cos_3shot ft_cos_5shot ft_cos_10shot  
do 
        CUDA_VISIBLE_DEVICES=$cuda python3 -m tools.food_test_net --num-gpus 1 \
                --config-file ./checkpoints/food/faster_rcnn/faster_rcnn_R_101_FPN_${name}/config.yaml \
                --model-path ./checkpoints/food/faster_rcnn/faster_rcnn_R_101_FPN_${name}/model_final.pth \
                --test-name final \
                --eval-only &
done
