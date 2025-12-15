# Train script

from ultralytics import YOLO
import argparse
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='What the program does',
        epilog='Text at the bottom of help')

    parser.add_argument('--model', "-m", type=str, required=True,
                        help='Model type.')
    parser.add_argument('--project', "-p", type=str, required=True,
                        help='Project name.')
    parser.add_argument('--resume', "-r", type=bool, required=False,
                        help='resume?', default=False)

    args = parser.parse_args()

    model = YOLO(args.model)

    train_results = model.train(
        # path to dataset YAML
        data="labels.yaml",
        resume=args.resume,
        epochs=120,  # number of training epochs
        imgsz=(420, 420),  # training image size
        save=True,  # save checkpoint every epoch
        rect=False,
        half=True,
        save_period=3,  # save model after interval
        device="0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
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
