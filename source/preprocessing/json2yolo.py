import json
from collections import defaultdict

from ..utils.utils import *


def convert_coco_json(json_dir="../coco/annotations/", use_segments=False):
    """Converts COCO JSON format to YOLO label format, with options for segments and class mapping."""
    save_dir = json_dir.replace("annotations/", "")
    json_dir = Path(json_dir)
    if not json_dir.is_absolute():
        json_dir = Path(__file__).resolve().parent.parent / json_dir

    # Import json
    for json_file in sorted(Path(json_dir).resolve().glob("*.json")):

        fn = Path(save_dir) / "labels"   # folder name
        fn.mkdir(parents=True, exist_ok=True)
        with open(json_file) as f:
            data = json.load(f)

        # Create image dict
        images = {"{:g}".format(x["id"]): x for x in data["images"]}
        # Create image-annotations dict
        imgToAnns = defaultdict(list)
        for ann in data["annotations"]:
            imgToAnns[ann["image_id"]].append(ann)

        # Write labels file
        for img_id, anns in tqdm(imgToAnns.items(), desc=f"Annotations {json_file}"):
            img = images[f"{img_id:g}"]
            h, w, f = img["height"], img["width"], img["file_name"]

            bboxes = []
            segments = []
            for ann in anns:
                if ann["iscrowd"]:
                    continue
                # The COCO box format is [top left x, top left y, width, height]
                box = np.array(ann["bbox"], dtype=np.float64)
                box[:2] += box[2:] / 2  # xy top-left corner to center
                box[[0, 2]] /= w  # normalize x
                box[[1, 3]] /= h  # normalize y
                if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                    continue

                cls = ann["category_id"]

                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)

            # Write
            with open((fn / f).with_suffix(".txt"), "a") as file:
                print(fn, f)
                for i in range(len(bboxes)):
                    # cls, box or segments
                    line = (*(bboxes[i]),)
                    file.write(("%g " * len(line)).rstrip() % line + "\n")

if __name__ == "__main__":

    convert_coco_json(
        "datasets/bbox/preprocessed_yolo_formatted_files/dronewaste_V1/annotations/",
        use_segments=False,
    )