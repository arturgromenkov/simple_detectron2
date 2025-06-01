from config import setup

import os
import argparse

import cv2
import detectron2
from detectron2.engine import DefaultTrainer
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper, build_detection_test_loader, build_detection_train_loader
from detectron2.data.detection_utils import build_augmentation
from detectron2.data.datasets import register_coco_instances


class MyTrainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return detectron2.evaluation.COCOEvaluator(dataset_name)

    @classmethod
    def build_train_loader(cls, cfg):
        dataloader = build_detection_train_loader(
            cfg, mapper=DatasetMapper(cfg, is_train=True))

        return dataloader
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        augmentations = build_augmentation(cfg, False)
        return build_detection_test_loader(
            cfg, dataset_name, mapper=DatasetMapper(cfg, is_train=False, augmentations=augmentations))


def main(args):
    if os.path.exists(args.output_folder) and os.listdir(args.output_folder):
        print('Folder {} exists and not empty'.format(args.output_folder))
        return
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Register datasets
    register_coco_instances('dataset_train', {}, args.train_json, args.images_folder)
    register_coco_instances('dataset_val', {}, args.val_json, args.images_folder)

    # Load cfg from config
    cfg = setup(args.output_folder)

    # Load Trainer class and start training
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False) # If you want to continue you need to make it True
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Trains model on given dataset in COCO format and dumps checkpoints to output folder')
    parser.add_argument('train_json', help='Path to training JSON')
    parser.add_argument('val_json', help='Path to validating JSON')
    parser.add_argument('images_folder', help='Path to folder with images')
    parser.add_argument('output_folder', help='Path to folder where training results will be written')

    main(parser.parse_args())