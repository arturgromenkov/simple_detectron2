import argparse

from config import setup

import cv2
import torch

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


def main(args):
    # Load the configuration for your trained model
    cfg = setup()
    cfg.MODEL.WEIGHTS = args.checkpoint_path

    # Build model from pth
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()

    # Load image
    image = cv2.imread(args.image_path)
    inputs = [{"image": torch.from_numpy(image).permute(2, 0, 1).float()}]

    # Get outputs
    with torch.no_grad():
        outputs = model(inputs)[0]

    # Visualazie results
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("Result", out.get_image()[:, :, ::-1])
    cv2.waitKey(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Show example prediction of model loaded from .pth')
    parser.add_argument('checkpoint_path', help='Path to checkpoint')
    parser.add_argument('image_path', help='Path to image')

    main(parser.parse_args())