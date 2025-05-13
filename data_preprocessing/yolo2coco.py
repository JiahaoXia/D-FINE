import os, sys
from os.path import dirname, abspath, join, exists
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import yaml
import json
import shutil
from PIL import Image
from tqdm import tqdm

root_dir = dirname(dirname(abspath(__file__)))


def process(input_file, output_image_dir, output_annotation_file, class_id2name=None):
    """
    Convert YOLO format annotations to COCO format.

    Args:
        input_file (str): Path to the input file containing absolute paths to images.
        output_image_dir (str): Directory to copy images to.
        output_annotation_file (str): Path to output the COCO-format JSON annotation.
        class_id2name (dict): Mapping of class IDs to class names.
    """
    os.makedirs(output_image_dir, exist_ok=True)

    coco = {"images": [], "annotations": [], "categories": []}

    category_set = {}  # class_id to category_id mapping
    annotation_id = 1
    image_id = 1

    with open(input_file, "r") as f:
        image_paths = [line.strip() for line in f.readlines() if line.strip()]

    for img_path in tqdm(image_paths, desc="Processing images"):
        if not os.path.isfile(img_path):
            print(f"Image not found: {img_path}")
            continue

        # Copy image
        img_filename = os.path.basename(img_path)
        dst_image_path = os.path.join(output_image_dir, img_filename)
        shutil.copyfile(img_path, dst_image_path)

        # Get image size
        with Image.open(img_path) as img:
            width, height = img.size

        # Add to COCO images
        coco["images"].append(
            {
                "id": image_id,
                "file_name": img_filename,
                "width": width,
                "height": height,
            }
        )

        # Get corresponding label file
        label_path = img_path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        if not os.path.isfile(label_path):
            print(f"Label not found: {label_path}")
            image_id += 1
            continue

        with open(label_path, "r") as lf:
            for line in lf:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue  # skip invalid lines

                class_id, x_center, y_center, w, h = map(float, parts)

                if class_id not in category_set:
                    category_id = int(class_id)
                    category_set[class_id] = category_id
                    coco["categories"].append(
                        {
                            "id": category_id,
                            "name": (
                                f"class_{int(class_id)}"
                                if class_id2name is None
                                else class_id2name[int(class_id)]
                            ),
                            "supercategory": "none",
                        }
                    )
                else:
                    category_id = category_set[class_id]

                # Convert YOLO to COCO bbox
                bbox_width = w * width
                bbox_height = h * height
                bbox_x = (x_center * width) - (bbox_width / 2)
                bbox_y = (y_center * height) - (bbox_height / 2)

                coco["annotations"].append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [bbox_x, bbox_y, bbox_width, bbox_height],
                        "area": bbox_width * bbox_height,
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1

        image_id += 1

    # Save COCO annotations to file
    with open(output_annotation_file, "w") as jf:
        json.dump(coco, jf, indent=2)


@hydra.main(
    version_base=None,
    config_path=join(root_dir, "configs", "data_preprocessing"),
    config_name="FV4Training_parcels_sliced",
)
def main(args: DictConfig) -> None:
    print("============ START ============")
    print(OmegaConf.to_yaml(args))

    output_path = Path(args.output_path)
    os.makedirs(output_path, exist_ok=True)

    # Save the YAML configuration to the output path
    with open(join(args.output_path, "cfg.yaml"), "w") as file:
        yaml.safe_dump(OmegaConf.to_container(args, resolve=True), file)

    os.makedirs(output_path / "images", exist_ok=True)
    os.makedirs(output_path / "annotations", exist_ok=True)
    os.makedirs(output_path / "images/train", exist_ok=True)
    os.makedirs(output_path / "images/val", exist_ok=True)
    os.makedirs(output_path / "images/test", exist_ok=True)

    # convert the dataset to coco format
    process(
        join(args.input_path, args.train),
        output_path / "images/train",
        output_path / "annotations/train.json",
    )
    process(
        join(args.input_path, args.val),
        output_path / "images/val",
        output_path / "annotations/val.json",
    )
    process(
        join(args.input_path, args.test),
        output_path / "images/test",
        output_path / "annotations/test.json",
    )

    print("============ END ============")


if __name__ == "__main__":
    main()
