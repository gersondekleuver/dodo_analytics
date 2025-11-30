from ultralytics import YOLO


# Customize validation settings
# validation_results = model.val(
#     data="labels.yaml", imgsz=512, conf=0.25, iou=0.6, device="0")


# data = "labels.yaml",
# epochs = 150,  # number of training epochs
# imgsz = 512,  # training image size
# save = True,  # save checkpoint every epoch
# # save_period=3,  # save model after interval
# device = "0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
# # resume=True,    # resume training from last.pt
# cache = False,  # cache images for faster training
# # parent directory of the runs
# project = f"runs/train/{args.model}_{year}_{day}",
# multi_scale = True,  # vary img-size +/- 50%
# hsv_h = 0.015,  # image HSV-Hue augmentation (fraction)
# degrees = 180.0,  # image rotation (+/- deg)
# mixup = 0.2,  # image mixup (fraction)


if __name__ == '__main__':
    # Load a model
    model = YOLO(
        "../runs/train/yolo11m.pt_2025_53/train3/weights/best.pt")

    metrics = model.val()  # no arguments needed, dataset and settings remembered
    metrics.box.map  # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps  # a list contains map50-95 of each category

    print("Validation complete")
    print(metrics.box.map)
    print(metrics.box.map50)
    print(metrics.box.map75)
    print(metrics.box.maps)
