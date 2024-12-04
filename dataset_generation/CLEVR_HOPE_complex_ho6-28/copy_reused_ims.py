import argparse
import os
import json
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_image_dir",
        default="outputs/images/train/",
        help="The directory where output images will be copied to. It will be "
        + "created if it does not exist.",
    )
    parser.add_argument(
        "--scene_graphs_path",
        default="outputs/scenes_to_render/tmp_train_sg_for_generation.json",
        type=str,
        help="Path to the CLEVR scene graphs to render into images. File should follow the"
        "standard CLEVR scenegraph. For each image, the keys 'camera_jitter', "
        "'Lamp_Key', 'Lamp_Back', and 'Lamp_Fill', should"
        "be added; otherwise they will be set to all-zeros. Each should be a list"
        "of three floats, representing the 3D offset for each of these elements.",
    )
    parser.add_argument(
        "--reusable_ims_config",
        default="reuse_config.json",
        type=str,
        help="Path to a .json config file specifying the scenegraphs file & corresponding image directories for image we can reuse.",
    )

    args = parser.parse_args()
    os.makedirs(args.output_image_dir, exist_ok=True)

    with open(args.reusable_ims_config, "r") as infile:
        config = json.load(infile)
    ori_im_paths = config["reuse_image_directories"]

    with open(args.scene_graphs_path, "r") as infile:
        scenegraphs = json.load(infile)["scenes"]

    # Copy over the images
    for scene in scenegraphs:
        if "reuse_dataset" in scene:
            assert "reuse_filename" in scene
            shutil.copy(
                os.path.join(
                    ori_im_paths[scene["reuse_dataset"]], scene["reuse_filename"]
                ),
                os.path.join(args.output_image_dir, scene["image_filename"]),
            )
