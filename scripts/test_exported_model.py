import argparse

import torch
import cv2

from config import setup

def main(args):
    # Load the configuration for your trained model
    cfg = setup()

    # Load the TorchScript model
    model = torch.jit.load(args.model_path)
    model.eval()

    # Prepare an input tensor (example shape, modify as needed)
    image = cv2.imread(args.image_path)
    input_tensor = torch.from_numpy(image)

    # Run inference
    output = model(input_tensor)

    print('Model output is : ', output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Show example prediction of model loaded from .pt')
    parser.add_argument('model_path', help='Path to model(.pt)')
    parser.add_argument('image_path', help='Path to image')

    main(parser.parse_args())