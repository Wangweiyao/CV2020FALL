"""
Detection Testing Script.

This scripts reads a given config file and runs the evaluation.
It is an entry point that is made to evaluate standard models in FsDet.

In order to let one script support evaluation of many models,
this script contains logic that are specific to these built-in models and
therefore may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use FsDet as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import numpy as np
import torch

from fsdet.config import get_cfg, set_global_cfg
from fsdet.engine import DefaultTrainer, default_argument_parser, default_setup

import detectron2.utils.comm as comm
import json
import logging
import os
import time
from collections import OrderedDict
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.engine import hooks, launch
from fsdet.evaluation import (
    FOODEvaluator, COCOEvaluator, DatasetEvaluators, LVISEvaluator, PascalVOCDetectionEvaluator, verify_results)

_base_classes = [
    1,2,3,4,6,8,9,12,14,15,
    16,17,18,19,22,24,25,26,28,29,
    32,38,51
]
_novel_classes = [ 
    5,7,10,11,13,20,21,23,27,30,
    31,33,34,35,36,37,39,40,41,42,
    43,44,45,46,47,48,49,50,52,53
]
class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_list.append(
            FOODEvaluator(dataset_name, cfg, True, output_folder)
        )
        '''
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "coco":
            evaluator_list.append(
                COCOEvaluator(dataset_name, cfg, True, output_folder)
            )
        if evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        '''
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)


class Tester:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = Trainer.build_model(cfg)
        self.check_pointer = DetectionCheckpointer(
            self.model, save_dir=cfg.OUTPUT_DIR
        )

        self.best_res = None
        self.best_file = None
        self.all_res = {}

    def test(self, ckpt):
        self.check_pointer._load_model(self.check_pointer._load_file(ckpt))
        print("evaluating checkpoint {}".format(ckpt))
        res = Trainer.test(self.cfg, self.model)

        if comm.is_main_process():
            verify_results(self.cfg, res)
            print(res)
            if (self.best_res is None) or (
                self.best_res is not None
                and self.best_res["bbox"]["AP"] < res["bbox"]["AP"]
            ):
                self.best_res = res
                self.best_file = ckpt
            print("best results from checkpoint {}".format(self.best_file))
            print(self.best_res)
            self.all_res["best_file"] = self.best_file
            self.all_res["best_res"] = self.best_res
            self.all_res[ckpt] = res
            os.makedirs(
                os.path.join(self.cfg.OUTPUT_DIR, "inference"), exist_ok=True
            )
            with open(
                os.path.join(self.cfg.OUTPUT_DIR, "inference", "all_res.json"),
                "w",
            ) as fp:
                json.dump(self.all_res, fp)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()
    set_global_cfg(cfg)
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    device = torch.device('cuda')  
    model = Trainer.build_model(cfg)

    check_pointer = DetectionCheckpointer(
                model, save_dir=cfg.OUTPUT_DIR
            )
    ckpt_file = cfg.MODEL.WEIGHTS
    check_pointer._load_model(check_pointer._load_file(ckpt_file))

    test_loader = Trainer.build_train_loader(cfg)
    feature_bank = {}
    #import pdb; pdb.set_trace()
    counter = 0
    for batch in test_loader:
        batch = batch
        #model(batch)
        #image = cv2.imread('my_image.jpg')
        #height, width = image.shape[:2]
        #image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        #inputs = [{"image": image, "height": height, "width": width}]
        model.eval()
        inputs = batch

        gt_proposal = batch[0]['instances']
        ##
        gt_classes = gt_proposal.gt_classes

        with torch.no_grad():
            images = model.preprocess_image(inputs)  # don't forget to preprocess
            features = model.backbone(images.tensor)  # set of cnn features
            #proposals, _ = model.proposal_generator(images, features, None)  # RPN
            
            useful_layers = model.roi_heads.in_features
            features_ = []
            for layer in useful_layers:
                features_.append(features[layer])
            gt_proposal.gt_boxes.tensor = gt_proposal.gt_boxes.tensor.to(device)
            # 
            #if len(model.proposal_generator(images,features)[0][0].objectness_logits) > 0:
            #    import pdb; pdb.set_trace()
            #if len(model(inputs)[0]['instances'].pred_classes):
            #    import pdb; pdb.set_trace()
            box_features = model.roi_heads.box_pooler(features_, [gt_proposal.gt_boxes])
            box_features = model.roi_heads.box_head(box_features)  # features of all 1k candidates

            for i in range(len(gt_classes)):
                class_idx = int(gt_classes[i].cpu().numpy())

                predicted = model.roi_heads.box_predictor.cls_score(box_features[0]).argmax()
                #print(f'{class_idx} -> {predicted}')
                if class_idx in feature_bank:
                    #import pdb; pdb.set_trace()
                    feature_bank[class_idx] = torch.cat((feature_bank[class_idx], box_features[i:,:]),0)
                else:
                    feature_bank[class_idx] = box_features[i:,:]
            
        counter += 1
        print(counter)
        if counter > 1000:
            break

    print(f'We found {len(feature_bank.keys())} classes')
    for key in feature_bank:
        feature_bank[key] = torch.mean(feature_bank[key], 0, True)
        feature_bank[key] /= feature_bank[key].norm()

    
    for j in feature_bank:
        '''
        class_id = j+1
        if class_id in _base_classes:
            print(f'skip class {class_id}')
            continue

        max_dot = -1
        max_i = 0
        for i in range(53):
            dot = torch.dot()
        '''
        model.roi_heads.box_predictor.cls_score.weight[j] = feature_bank[j][0]

    check_pointer.save('5shot')


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    if args.eval_during_train or args.eval_all:
        args.dist_url = "tcp://127.0.0.1:{:05d}".format(
            np.random.choice(np.arange(0, 65534))
        )
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
