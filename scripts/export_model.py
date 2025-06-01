import argparse
import torch
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.export.flatten import TracingAdapter
from config import setup  # your config setup function
import torch.jit as jit
import numpy as np
import cv2


def main(args):
    # Load configuration and weights
    cfg = setup()
    cfg.MODEL.WEIGHTS = args.checkpoint_path
    cfg.MODEL.DEVICE = args.device
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.thresh  # set threshold

    # Build and load model
    model = build_model(cfg)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    model.eval()

    # Move model to device
    model.to(cfg.MODEL.DEVICE)

    # Create dummy input tensor with batch dimension and channels
    dummy_input = torch.randn(cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, 3).to(cfg.MODEL.DEVICE) # TODO:CHANGE
    
    def inference_func(model, image):
        image = image.permute(2, 0, 1).float()  # ensure shape (C,H,W)
        inputs = [{"image": image}]
        return model.inference(inputs, do_postprocess=False)[0]

    wrapper = TracingAdapter(model, dummy_input, inference_func)
    wrapper.eval()
    traced_script_module = torch.jit.trace(wrapper, (dummy_input,))

    # Save TorchScript model
    traced_script_module.save(args.exported_model_path)
    print(f"TorchScript model saved to {args.exported_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Detectron2 model to TorchScript")
    parser.add_argument("checkpoint_path", help="Path to model checkpoint (.pth)")
    parser.add_argument("exported_model_path", help="Path to save TorchScript model (.pt)")
    parser.add_argument("device", help="Device - either cuda or cpu")
    parser.add_argument("--thresh", help="Set threshold for confidence", default=0.5)
    args = parser.parse_args()

    main(args)

