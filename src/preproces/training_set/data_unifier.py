import os
import yaml
import argparse
import shutil


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
        "bag": 0,
        "other trash clustered cigarette litter straw lid pop tab scrap": 1,
        "cardboard uht carton": 2,
        "glass": 3,
        "metal aluminium": 4,
        "paper wrapper": 5,
        "plastic vinyl pet bottle cup tyres": 6,
        "styrofoam": 7,
        "wood pallets": 8,
    }

    # test: unified_dataset/test/images
    # train: unified_dataset/train/images
    # val: unified_dataset/valid/images

    from_old_to_new_id_mapping = {}

    # Each dataset has a folder which includes a train, test and valid folder.enumerate
    # Each dataset also includes a data.yaml file which includes the class names and the number of classes
    # The loop below goes through each dataset folder and reads the data.yaml file to get the class names
    # The class names are then checked against the unified_classes dict to see if the class name is in the dict
    # If the class name is in the dict, the class name is appended to the all_classes list with the corresponding value in the dict
    # If the class name is not in the dict its discarded
    discarded_items = 0
    discarded_classes = []
    remapped_items = 0
    remapped_classes = []
    all_classes = []
    for dataset_folder in os.listdir(datasets_dir):
        dataset_path = os.path.join(datasets_dir, dataset_folder)
        from_old_to_new_id_mapping[dataset_path] = {}
        if os.path.isdir(dataset_path):
            yaml_path = os.path.join(dataset_path, "data.yaml")
            if os.path.exists(yaml_path):
                with open(yaml_path, "r") as file:
                    data = yaml.safe_load(file)
                    if "names" in data:
                        for old_id, name in enumerate(data["names"]):
                            name = name.lower()
                            all_classes.append(name)
                            temp = 0
                            # check if the name is a substring of any of the unified_ids keys
                            for unified_id in unified_ids.keys():
                                if temp == 1:
                                    break
                                name = name.replace("-", " ")
                                name = name.replace("_", " ")

                                for word in name.split(" "):
                                    word = word.strip()
                                    if word:
                                        if word in unified_id:
                                            # check if the name is already in the unified_ids dict
                                            if (
                                                unified_id
                                                in from_old_to_new_id_mapping[
                                                    dataset_path
                                                ]
                                            ):
                                                from_old_to_new_id_mapping[
                                                    dataset_path
                                                ][old_id] = from_old_to_new_id_mapping[
                                                    dataset_path
                                                ][unified_id]

                                            else:
                                                from_old_to_new_id_mapping[
                                                    dataset_path
                                                ][old_id] = unified_ids[unified_id]
                                            remapped_items += 1
                                            temp = 1
                                            break

                            if not temp:
                                discarded_items += 1
                                discarded_classes.append(name)
                            else:
                                remapped_classes.append(name)

    declined_files = 0
    remapped_files = 0
    background_files = 0
    remapped_labels = 0
    declined_labels = 0
    # initialize the new dataset dir
    if not os.path.exists(target_dataset_dir):
        os.makedirs(target_dataset_dir)
        # initialize the train, test and valid folders
        for split in ["train", "test", "valid"]:
            if not os.path.exists(os.path.join(target_dataset_dir, split)):
                os.makedirs(os.path.join(target_dataset_dir, split, "images"))
                os.makedirs(os.path.join(target_dataset_dir, split, "labels"))

    simple_names_lookup = {
        "bag": "bag",
        "other trash clustered cigarette litter straw lid pop tab scrap": "litter",
        "cardboard uht carton": "cardboard",
        "glass": "glass",
        "metal aluminium": "metal",
        "paper wrapper": "paper",
        "plastic vinyl pet bottle cup tyres": "plastic",
        "styrofoam": "styrofoam",
        "wood pallets": "wood",
    }

    simple_names = {}
    # remap the unified_ids dict to simple names
    for key in unified_ids.keys():
        simple_names[simple_names_lookup[key]] = unified_ids[key]

    # initalize the data.yaml file
    with open(os.path.join(target_dataset_dir, "data.yaml"), "w") as file:
        yaml.dump(
            {
                "names": list(simple_names.keys()),
                "nc": len(unified_ids),
                "train": "../train/images",
                "test": "../test/images",
                "val": "../valid/images",
            },
            file,
        )

    # dump the data.yaml file in the root folder
    with open("labels.yaml", "w") as file:
        yaml.dump(
            {
                "names": list(simple_names.keys()),
                "nc": len(unified_ids),
                "train": f"{target_dataset_dir.replace('datasets', '..')}/train/images",
                "test": f"{target_dataset_dir.replace('datasets', '..')}/test/images",
                "val": f"{target_dataset_dir.replace('datasets', '..')}/valid/images",
            },
            file,
        )

    for mapping in from_old_to_new_id_mapping:
        print(mapping)
        print(from_old_to_new_id_mapping[mapping])

    for dataset_folder in os.listdir(datasets_dir):
        dataset_path = os.path.join(datasets_dir, dataset_folder)
        # Now we overwrite the label files in the train,test and valid folders of the dataset with the new class ids
        for split in ["train", "test", "valid"]:
            split_path = os.path.join(dataset_path, split)
            if os.path.isdir(split_path):
                for file_name in os.listdir(os.path.join(split_path, "labels")):
                    class_id = None
                    file_path = os.path.join(split_path, "labels", file_name)
                    target_file_path = os.path.join(
                        target_dataset_dir, split, "labels", file_name
                    )

                    with open(file_path, "r") as file:
                        lines = file.readlines()

                    if lines:
                        with open(target_file_path, "w") as file:
                            switch = 0
                            for line in lines:
                                class_id = int(line.split()[0])
                                if class_id in from_old_to_new_id_mapping[dataset_path]:
                                    new_class_id = from_old_to_new_id_mapping[
                                        dataset_path
                                    ][class_id]
                                    file.write(
                                        f"{new_class_id} {' '.join(line.split()[1:])}\n"
                                    )

                                    # copy the image file to the new dataset folder
                                    image_path = os.path.join(
                                        split_path,
                                        "images",
                                        file_name.replace("txt", "jpg"),
                                    )
                                    target_image_path = os.path.join(
                                        target_dataset_dir,
                                        split,
                                        "images",
                                        file_name.replace("txt", "jpg"),
                                    )
                                    image_path_2 = os.path.join(
                                        split_path,
                                        "images",
                                        file_name.replace("txt", "PNG"),
                                    )

                                    if not os.path.exists(target_image_path):
                                        try:
                                            shutil.copy(image_path, target_image_path)
                                        except:
                                            shutil.copy(image_path_2, target_image_path)

                                    remapped_labels += 1
                                    switch = 1
                                else:
                                    declined_labels += 1
                                    print(
                                        f"Discarded class {class_id} in {split_path + file_name}"
                                    )

                        if switch == 0:
                            declined_files += 1
                            os.remove(target_file_path)

                        else:
                            remapped_files += 1
                    else:
                        background_files += 1
                        image_path = os.path.join(
                            split_path, "images", file_name.replace("txt", "jpg")
                        )
                        target_image_path = os.path.join(
                            target_dataset_dir,
                            split,
                            "images",
                            file_name.replace("txt", "jpg"),
                        )
                        image_path_2 = os.path.join(
                            split_path, "images", file_name.replace("txt", "PNG")
                        )

                        if not os.path.exists(target_image_path):
                            try:
                                shutil.copy(image_path, target_image_path)
                            except:
                                shutil.copy(image_path_2, target_image_path)

    print(f"Discarded {discarded_items} / {remapped_items + discarded_items} classes")
    print(f"Remapped {remapped_items} classes")

    print(f"Remapped {remapped_files} files")
    print(f"{declined_files} files not copied")

    print(f"Remapped {remapped_labels} classes")
    print(f"{declined_labels} classes not copied")

    print(f"{background_files} background files copied")

    print(f"Total copied files: {remapped_files + background_files}")


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
