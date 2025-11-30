from ultralytics import YOLO
from datetime import datetime
import wandb
import argparse
import os
from datetime import date
import torch
print(torch.cuda.is_available())


TODAY = str(date.today())
wandb.login(key="b25571ab37e3a918d9a834e2f7eda26a1259d304")

if __name__ == '__main__':
    # Retrieve dataset path from argparser

    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='What the program does',
        epilog='Text at the bottom of help')

    parser.add_argument('--model', "-m", type=str, required=True,
                        help='Model type.')
    parser.add_argument('--project', "-p", type=str, required=True,
                        help='Project name.')

    args = parser.parse_args()

    # Load a model
    model = YOLO(args.model)

    print(os.getcwd())
    print(os.listdir())
    # Train the model
    train_results = model.train(
        # path to dataset YAML
        data="labels.yaml",
        resume=True,
        epochs=120,  # number of training epochs
        imgsz=(512, 512),  # training image size
        save=True,  # save checkpoint every epoch
        rect=False,
        half=True,
        save_period=3,  # save model after interval
        device="0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
        # resume=True,    # resume training from last.pt
        cache=False,  # cache images for faster training
        # parent directory of the runs
        project=f"runs/train/{args.project}",
        multi_scale=True,  # vary img-size +/- 50%
        hsv_h=0.015,  # image HSV-Hue augmentation (fraction)
        degrees=180.0,  # image rotation (+/- deg)
        mixup=0.2,  # image mixup (fraction)
    )

    # Evaluate model performance on the validation set
    metrics = model.val()

    # Export the model to ONNX format
    path = model.export(format="onnx")  # return path to exported model
