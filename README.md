# CV2020FALL
## Data preprocessing:
Please follow instructions step 1-7 in data preprocessing folder
## Faster RCNN
```
$ cd faster-rcnn
```
Training with base classes: 
```
$ bash run_base_training.sh
```
Finetune with novel classes:
```
$ bash run_finetuning.sh
```
Cosine similairty distance with novel classes:
```
$ bash run_extract_features.sh
```

## YOLO-V5

```bash
$ cd yolov5
```
Requirements:


Python 3.8 or later with all [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) dependencies installed, including `torch>=1.6`. To install run:
```bash
$ pip install -r requirements.txt
```
Training:

```bash
$ python train.py --img 640 --batch 32 --epochs 300 --data food53_base.yaml --weights yolov5s.pt --device 0
```

Fine-tuning:

```bash
$ python train.py --img 640 --batch 32 --epochs 300 --data food53_1_shot.yaml --weights weights/best.pt --device 0 --hyp hyp.finetune.yaml --freeze-backbone
                                                           food53_3_shot.yaml  
                                                           food53_5_shot.yaml
                                                           food53_10_shot.yaml

```

Inference:

```bash
$ pip install -r requirements.txt
```

### few-shot

Requirements:


Python 3.8 or later with all [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) dependencies installed, including `torch>=1.6`. To install run:
```bash
$ pip install -r requirements.txt
```
Training:
<<<<<<< HEAD

cd to the folder yolov5
```bash
$ pip install -r requirements.txt
```


=======
```bash
$ python train.py --img 640 --batch 32 --epochs 300 --data food53_base.yaml --weights yolov5s.pt --device 0
```
Fine-tuning:
```bash
$ python train.py --img 640 --batch 32 --epochs 300 --data food53_1_shot.yaml --weights weights/best.pt --device 0 --hyp hyp.finetune.yaml --freeze-backbone
                                                           food53_3_shot.yaml  
                                                           food53_5_shot.yaml
                                                           food53_10_shot.yaml
```
>>>>>>> 1a549280ebbdaf12092bba0072bc582b9fd42074
Inference:

```bash
$ python detect.py --source data/images --weights weights/best.pt --conf 0.25
```

### Meta-Baseline for Few-Shot Image Classification 
```bash
$ cd few-shot-meta-baseline
```
Baseline Training:
```bash
$ python train_classifier.py --config configs/train_classifier_food53_randaug.yaml --gpu 4,5,6,7 --name baseline_resnet18_randaug
```
Meta-Learning Finetuning:
```bash
$ python train_meta.py --config configs/train_meta_food53_randaug_tune3.yaml --gpu 4,5,6,7 --name baseline_resnet18_randaug_meta
```

## Pose Estimation
Pose estimation for input image with name "img_name": pose_estimation(img_name)

All functions are included in the Section "All Code" of the Python notebook Pose_Estimation.ipynb


## Reference

