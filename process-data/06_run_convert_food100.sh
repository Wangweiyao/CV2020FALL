for prefix in all_ base_ novel_
do
    for name in train train_1_shot train_3_shot train_5_shot train_10_shot val test
     do 
         python 06_voc2coco.py \
            --ann_dir ~/data/food53/Annotations \
            --ann_ids ~/data/food53/$prefix$name.txt \
            --labels ~/data/food53/new_labels.txt \
            --output ~/data/food53/$prefix$name.json \
            --ext xml

     done 
done