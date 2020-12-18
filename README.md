# CV2020FALL
## Data preprocessing:
Please follow instructions step 1-7 in data preprocessing folder
## Faster RCNN
Training with base classes: 
bash run_base_training.sh

Finetune with novel classes:
bash run_finetuning.sh

Cosine similairty distance with novel classes:
bash run_extract_features.sh


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

## few-shot

Requirements:

Python 3.8 or later with all [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) dependencies installed, including `torch>=1.6`. To install run:
```bash
$ pip install -r requirements.txt
```
Training:

cd to the folder yolov5
```bash
$ pip install -r requirements.txt
```


Inference:

```bash
$ pip install -r requirements.txt
```


## Pose Estimation
Pose estimation for input image with name "img_name": pose_estimation(img_name)

All functions are included in the Section "All Code" of the Python notebook Pose_Estimation.ipynb


## Reference

