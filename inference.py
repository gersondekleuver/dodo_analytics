from ultralytics import YOLO


if __name__ == '__main__':

    # Load a model
    model = YOLO("../runs/train/yolo11m.pt_2025_53/train3/weights/best.pt")

    model.predict("test3.jpg")  # run inference on a single image

    # Run batched inference on a list of images
    # return a list of Results objects
    results = model(task="detect", source="test3.jpg",
                    show_labels=True, show_conf=False, save=True, device="0", augment=False)
    # results.show()  # display bounding boxes

    # results = models(["test.jpg"])  # run inference on a single image

    # # Process results list
    # for result in results:
    #     boxes = result.boxes  # Boxes object for bounding box outputs
    #     masks = result.masks  # Masks object for segmentation masks outputs
    #     keypoints = result.keypoints  # Keypoints object for pose outputs
    #     probs = result.probs  # Probs object for classification outputs
    #     obb = result.obb  # Oriented boxes object for OBB outputs
    #     result.show()  # display bounding boxes
    #     result.save(filename="result.jpg")  # save to disk
