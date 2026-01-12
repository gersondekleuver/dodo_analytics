from PIL import Image
import os
import yaml
import argparse


def list_all_class_names(args):
    """
    Lists all unique class names from the data.yaml files in the given datasets directory.

    Parameters:
        datasets_dir (str): Path to the directory containing the datasets.

    Returns:
        set: A set of all unique class names.
    """
    datasets_dir = args.source_datasets_dir
    target_dataset_dir = args.target_dataset_dir

    # dict of classes which all datasets need to map to if a class has the same substring name.
    unified_ids = {
        "cardboard uht carton": 0,
        "glass": 1,
        "metal aluminium pop tab": 2,
        "paper wrapper": 3,
        "plastic vinyl pet bottle lid cup": 4,
        "styrofoam": 5,
        "trash facemask clustered cigarette litter straw bag": 6,
    }

    from_old_to_new_id_mapping = {}

    for dataset_folder in os.listdir(datasets_dir):
        print(dataset_folder)
        dataset_path = os.path.join(datasets_dir, dataset_folder)
        from_old_to_new_id_mapping[dataset_path] = {}
        if os.path.isdir(dataset_path):
            # open the labels folder
            labels_path = os.path.join(dataset_path, "labels")
            target_labels_path = os.path.join(
                target_dataset_dir, dataset_folder, "labels"
            )
            print(labels_path)
            if os.path.exists(labels_path):
                for file_name in os.listdir(labels_path):
                    with open(os.path.join(labels_path, file_name), "r") as file:
                        lines = file.readlines()
                        if lines:
                            # create file to write remapped labels
                            target_file_path = os.path.join(
                                target_labels_path, file_name
                            )
                            with open(target_file_path, "w") as target_file:
                                classId = 0
                                for line in lines:
                                    bboxes = line.split()[4:8]

                                    # bboxes represent [<left> <top> <right> <bottom>]
                                    # make bounding boxes to be [center_x center_y width height] format normalized by the images size

                                    center_x = (float(bboxes[0]) + float(bboxes[2])) / 2
                                    center_y = (float(bboxes[1]) + float(bboxes[3])) / 2
                                    width = float(bboxes[2]) - float(bboxes[0])
                                    height = float(bboxes[3]) - float(bboxes[1])

                                    # get image size from the image file
                                    image_path = os.path.join(
                                        labels_path, file_name.replace("txt", "jpg")
                                    )
                                    image_path = image_path.replace("labels", "images")

                                    # get image size from image file

                                    image = Image.open(image_path)
                                    image_size = image.size

                                    print(image_size)
                                    # normalize the bounding boxes round to 3 decimal places
                                    center_x = round(center_x / image_size[0], 3)
                                    center_y = round(center_y / image_size[1], 3)
                                    width = round(width / image_size[0], 3)
                                    height = round(height / image_size[1], 3)

                                    bboxes = [
                                        str(center_x),
                                        str(center_y),
                                        str(width),
                                        str(height),
                                    ]

                                    target_file.write(f"{classId} {' '.join(bboxes)}\n")

    # save the unique class names to a data.yaml
    with open(os.path.join(target_dataset_dir, "data.yaml"), "w") as file:
        yaml.dump(
            {
                "names": ["bag"],
                "nc": 1,
                "train": "../train/images",
                "test": "../test/images",
                "val": "../valid/images",
            },
            file,
        )


if __name__ == "__main__":
    # Retrieve dataset path from argparser

    parser = argparse.ArgumentParser(
        prog="ProgramName",
        description="What the program does",
        epilog="Text at the bottom of help",
    )

    parser.add_argument(
        "--source_datasets_dir",
        "-s",
        type=str,
        required=True,
        help="Path to the directory containing the source datasets.",
    )
    parser.add_argument(
        "--target_dataset_dir",
        "-t",
        type=str,
        required=True,
        help="Path to the directory containing the target dataset.",
    )
    args = parser.parse_args()
    # List all unique class names
    list_all_class_names(args)
