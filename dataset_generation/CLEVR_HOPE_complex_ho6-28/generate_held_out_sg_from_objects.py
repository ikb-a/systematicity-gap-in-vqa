"""
Given the output of generate_held_out_objects.py, generate the actual
scene-graphs that can be rendered.

Where re-use is specified, check whether the CLEVR scenegraph can indeed be
reused; mark for reuse or create a new scenegraph accordingly.
"""

"""
# Testing: First create index, then create scenegraph .json files
blender --background --python generate_held_out_sg_from_objects.py -- --split train --objects_path held_out_objects_train.json --create_index --reusable_ims_config small_reuse_config.json
blender --background --python generate_held_out_sg_from_objects.py -- --split train --objects_path held_out_objects_train.json --start_idx 0 --num_glob_indices_to_process 1000000 --reusable_ims_config small_reuse_config.json
# Repeat for val_iid:
blender --background --python generate_held_out_sg_from_objects.py -- --split val_iid --objects_path held_out_objects_val_iid.json --create_index --reusable_ims_config small_reuse_config.json
blender --background --python generate_held_out_sg_from_objects.py -- --split val_iid --objects_path held_out_objects_val_iid.json --start_idx 0 --num_glob_indices_to_process 100000 --reusable_ims_config small_reuse_config.json
# Repeat for test:
blender --background --python generate_held_out_sg_from_objects.py -- --split test --objects_path held_out_objects_test.json --create_index --reusable_ims_config small_reuse_config.json
blender --background --python generate_held_out_sg_from_objects.py -- --split test --objects_path held_out_objects_test.json --start_idx 0 --num_glob_indices_to_process 100000 --reusable_ims_config small_reuse_config.json
"""

import sys

try:
    import bpy, bpy_extras
except ImportError as e:
    print("Please run from blender; e.g.")
    print(
        "blender --background --python generate_held_out_sg_from_objects.py -- [additional arguments]"
    )
    exit(-1)

try:
    import question_engine as qeng
except ImportError as e:
    print("\nERROR")
    print(e.msg)
    print("Cannot import question_engine")
    print("into held_out_clevr_additional/generate_held_out_sg_from_objects.py")
    print("You may need to add a .pth file to the site-packages of Blender's")
    print("bundled python with a command like this:\n")
    # print("echo $PWD >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/held_out_clevr_add.pth")
    # print("\nWhere $BLENDER is the directory where Blender is installed, and")
    # print("$VERSION is your Blender version (such as 2.78).")
    sys.exit(1)

try:
    from held_out_add_utils_generate_questions_blender_safe import (
        instantiate_templates_dfs,
    )
except ImportError as e:
    print("\nERROR")
    print(e.msg)
    print("Cannot import held_out_add_utils_generate_questions_blender_safe.py")
    print("into held_out_add_clevr_additional/generate_held_out_sg_from_objects.py")
    print("You may need to add a .pth file to the site-packages of Blender's")
    print("bundled python with a command like this:\n")
    print(
        "echo $PWD >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/held_out_clevr_add.pth"
    )
    print("\nWhere $BLENDER is the directory where Blender is installed, and")
    print("$VERSION is your Blender version (such as 2.78).")
    sys.exit(1)

try:
    from held_out_add_utils_generate_sg import (
        validate_scenegraph,
        create_scenegraph,
        select_attrs_unif_rand,
    )
except ImportError as e:
    print("\nERROR")
    print(e.msg)
    print("Cannot import held_out_utils_generate_sg.py")
    print("into held_out_add_clevr_additional/generate_held_out_sg_from_objects.py")
    print("You may need to add a .pth file to the site-packages of Blender's")
    print("bundled python with a command like this:\n")
    print(
        "echo $PWD >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/held_out_clevr_add.pth"
    )
    print("\nWhere $BLENDER is the directory where Blender is installed, and")
    print("$VERSION is your Blender version (such as 2.78).")
    sys.exit(1)

try:
    import atom_utils as utils
except ImportError as e:
    print("\nERROR")
    print(e.msg)
    print(
        "Running held_out_utils_generate_sg_from_objects.py from Blender and cannot import atom_utils.py."
    )
    print("You may need to add a .pth file to the site-packages of Blender's")
    print("bundled python with a command like this:\n")
    print(
        "echo {full path to dataset_generation/atom_clevr/} >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/atom_clevr.pth"
    )
    print("easier to change directory to dataset_generation/atom_clevr/ and then run:")
    print(
        "echo $PWD >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/atom_clevr.pth"
    )
    print("\nWhere $BLENDER is the directory where Blender is installed, and")
    print("$VERSION is your Blender version (such as 2.78).")
    sys.exit(1)


import json
import copy
import os
from typing import Tuple, List, Optional, Dict
import random
import argparse

CLEVR_COLORS = [
    None,
    "blue",
    "brown",
    "cyan",
    "gray",
    "green",
    "purple",
    "red",
    "yellow",
]
CLEVR_MATERIALS = [None, "rubber", "metal"]
CLEVR_SHAPES = [None, "cube", "cylinder", "sphere"]
CLEVR_SIZES = [None, "large", "small"]


def select_specified_attributes(
    obj_id: int,
    objects: List[Tuple[str, str, str, str]],
    size_mapping: List[Tuple[str, float]],
    material_mapping: List[Tuple[str, str]],
    object_mapping: List[Tuple[str, str]],
    color_name_to_rgba: List[Tuple[str, List[float]]],
) -> Tuple[
    Tuple[str, float], Tuple[str, str], Tuple[str, str], Tuple[str, List[float]]
]:
    vis_HO = objects[obj_id]
    # reorder (size, col, sh, mat) -> (size, mat, sh, col)
    vis_HO = (vis_HO[0], vis_HO[3], vis_HO[2], vis_HO[1])
    assert vis_HO[0] in CLEVR_SIZES
    assert vis_HO[1] in CLEVR_MATERIALS
    assert vis_HO[2] in CLEVR_SHAPES
    assert vis_HO[3] in CLEVR_COLORS

    # Set render object attributes to match what we want
    size_dict = {k: v for (k, v) in size_mapping}
    col_dict = {k: v for (k, v) in color_name_to_rgba}
    mat_dict = {k: v for (v, k) in material_mapping}
    obj_dict = {k: v for (v, k) in object_mapping}
    # (size, mat, shape, color)
    attr_dicts = [size_dict, mat_dict, obj_dict, col_dict]

    final_attrs = []
    for attr_dict, ho_attr, attr_name in zip(
        attr_dicts, vis_HO, ["sz", "mt", "sh", "co"]
    ):
        attr_val = attr_dict[ho_attr]
        if attr_name in ["sz", "co"]:
            final_attrs.append((ho_attr, attr_val))
        else:
            assert attr_name in ["sh", "mt"]
            final_attrs.append((attr_val, ho_attr))

    return tuple(final_attrs)


# objects must be list of (size, color, shape, material)
def objects_to_sg(
    objects: List[Tuple], img_name: str, img_idx: int, split: str, use_gpu: int = 1
):
    # attr_selection_fun chooses attributes matching {objects} argument
    selection_fun = (
        lambda id, sz_map, mat_map, obj_map, col_map: select_specified_attributes(
            obj_id=id,
            objects=objects,
            size_mapping=sz_map,
            material_mapping=mat_map,
            object_mapping=obj_map,
            color_name_to_rgba=col_map,
        )
    )

    return create_scenegraph(
        output_image_path=img_name,
        num_objects=len(objects),
        output_index=img_idx,
        output_split=split,
        use_gpu=use_gpu,
        attr_selection_fun=selection_fun,
    )


def randomize_lighting_and_camera(scenegraph: Dict) -> Dict:
    def rand(L):
        return 2.0 * L * (random.random() - 0.5)

    sg = copy.deepcopy(scenegraph)
    sg.update(
        {
            "camera_jitter": tuple((rand(0.5) for i in range(3))),
            "Lamp_Key": tuple((rand(1.0) for i in range(3))),
            "Lamp_Back": tuple((rand(1.0) for i in range(3))),
            "Lamp_Fill": tuple((rand(1.0) for i in range(3))),
        }
    )
    return sg


def obj_struct_to_tuple(obj_struct):
    return (
        obj_struct["size"],
        obj_struct["color"],
        obj_struct["shape"],
        obj_struct["material"],
    )


def create_proposals_from_base_sg(
    proposed_base_sg: Dict,
    scenes_to_create: List[List[Tuple[str, str, str]]],
    obj_idx_to_sg_obj_idx: Dict[int, int],
    global_idx: int,
    glob_loc_to_final_idx_map: Dict[Tuple[int, int], int],
    img_template: str,
):
    """
    Given a scenegraph for scenes_to_create[0], return a list of scene graphs
    for every collection of object descriptions in scenes_to_create

    Precondition: All scenes in scenes_to_create have the same number of objects
    Precondition: Object order is preserved between scenes in scenes_to_create

    :param proposed_base_sg: Scenegraph of the image without any alterations (dict in the same structure as the .json)
    :param scenes_to_create: List of lists of object attribute-tuples (size, color, shape, material). All have the same object order. The first element of the list should match the base scenegraph
    :param obj_idx_to_sg_obj_idx: mapping from the index i to index j s.t. the object at scenes_to_create[0][i] corresponds to proposed_base_sg['objects'][j]
    :return:
    """
    # make sure all tweaks to this scene have the same number of objects
    assert len(set(len(x) for x in scenes_to_create)) == 1

    results = []

    # Make sure our mapping is bijective and complete
    assert set(obj_idx_to_sg_obj_idx.values()) == set(range(len(scenes_to_create[0])))
    assert set(obj_idx_to_sg_obj_idx.keys()) == set(range(len(scenes_to_create[0])))

    for i, scene in enumerate(scenes_to_create):
        # Confirm the first scene_to_create corresponds to the base scenegraph,
        # and add it to the list of scenegraphs to return
        if i == 0:
            results.append(copy.deepcopy(proposed_base_sg))
            expected_img_idx = glob_loc_to_final_idx_map[(global_idx, 0)]
            assert proposed_base_sg["image_index"] == expected_img_idx
            assert proposed_base_sg["image_filename"] == img_template % expected_img_idx
            for obj_idx, obj in enumerate(scene):
                obj_sg = proposed_base_sg["objects"][obj_idx_to_sg_obj_idx[obj_idx]]
                assert obj == obj_struct_to_tuple(obj_sg)

        else:
            # Otherwise, change the object attribtues.
            # Here it is important to assume that object order is not changed
            img_idx = glob_loc_to_final_idx_map[(global_idx, i)]
            new_sg = copy.deepcopy(proposed_base_sg)
            new_sg["image_index"] = img_idx
            new_sg["image_filename"] = img_template % img_idx
            for obj_idx, obj in enumerate(scene):
                obj_sg = new_sg["objects"][obj_idx_to_sg_obj_idx[obj_idx]]

                # Adjust the z-axis coordinate if we changed the object's size
                if obj_sg["size"] != obj[0]:
                    z = 0.699999988079071 if obj[0] == "large" else 0.3499999940395355
                    obj_sg["3d_coords"] = (
                        obj_sg["3d_coords"][0],
                        obj_sg["3d_coords"][1],
                        z,
                    )

                obj_sg["size"] = obj[0]
                obj_sg["color"] = obj[1]
                obj_sg["shape"] = obj[2]
                obj_sg["material"] = obj[3]

            # Delete the image reuse info *if* it exists, since we are not
            # reusing this image. (we do not reuse paired images)
            new_sg.pop("reuse_dataset", None)
            new_sg.pop("reuse_filename", None)

            results.append(new_sg)

    assert len(results) == len(scenes_to_create)
    return results


def merge_scene_files(
    input_dirs: List[str], out_path: str, VERSION_DATE="2022-10-26", VERSION="1.0"
):
    scenes = []
    split = None
    for input_dir in input_dirs:
        for filename in os.listdir(input_dir):
            if not filename.endswith(".json"):
                continue
            path = os.path.join(input_dir, filename)
            with open(path, "r") as f:
                scene = json.load(f)
            scenes.append(scene)
            if split is not None:
                msg = "Input directory contains scenes from multiple splits"
                assert scene["split"] == split, msg
            else:
                split = scene["split"]
    scenes.sort(key=lambda s: s["image_index"])
    for s in scenes:
        print(s["image_filename"])
    output = {
        "info": {
            "date": VERSION_DATE,
            "version": VERSION,
            "split": "train",
            "license": "Creative Commons Attribution (CC-BY 4.0",
        },
        "scenes": scenes,
    }
    with open(out_path, "w") as f:
        json.dump(output, f)


def main(
    save_path,
    split,
    index_map_output_dir,
    objects_path,
    reuse_scene_graph_paths,
    reuse_dataset_names,
    start_idx,
    num_glob_indices_to_process,
    verbose,
    create_index,
    use_gpu,
    filename_prefix,
    merge_scenegraphs,
):
    os.makedirs(index_map_output_dir, exist_ok=True)
    ori_save_path = save_path
    save_path = os.path.join(save_path, split)
    os.makedirs(save_path, exist_ok=True)

    # 1) produce or load mapping from global index to final image index
    # We order the indices w.r.t. everything required by one held-out combination
    # at a time.

    print("Loading .json of objects")
    with open(objects_path, "r") as infile:
        objects_json = json.load(infile)

    total_global_indices = len(objects_json["image_data"])

    # Convert string keys back to integers
    string_keys = list(objects_json["image_data"].keys())
    for key in string_keys:
        objects_json["image_data"][int(key)] = objects_json["image_data"].pop(key)

    assert set(objects_json["image_data"].keys()) == set(range(total_global_indices))

    # Convert attribute lists to tuples
    for global_idx in objects_json["image_data"]:
        for scene in objects_json["image_data"][global_idx]:
            scene["objects"] = [tuple(x) for x in scene["objects"]]

    if create_index:  # If we're creating the index, do that and exit
        index_breakpoints = ""

        next_img_index = 0
        glob_loc_to_final_idx_map = {}

        # Need to ensure determinism in re-indexing between different parallel processes
        held_out_combos = list(objects_json["match_ho"].keys())
        held_out_combos.sort(key=lambda x: str(x))
        # Make sure the held-out combinations are the same across all types of images.
        for other_img_type in ["match_ho_pair", "not_match_ho"]:
            assert set(held_out_combos) == set(objects_json[other_img_type].keys())

        print("Start re-indexing")
        for ho in held_out_combos:
            print(ho)
            for img_type in ["match_ho", "match_ho_pair", "not_match_ho"]:
                print(img_type)
                for tmp_i, img_index in enumerate(objects_json[img_type][ho]):
                    global_idx = img_index["global_idx"]
                    local_idx = img_index["local_idx"]

                    if (global_idx, local_idx) not in glob_loc_to_final_idx_map:
                        glob_loc_to_final_idx_map[
                            (global_idx, local_idx)
                        ] = next_img_index
                        next_img_index += 1

            print(ho)
            print("Ended at image index: %i" % next_img_index)
            index_breakpoints += str(ho) + ": " + str(next_img_index) + "\n"

            # if tmp_i % 5000 == 0:
            # print(tmp_i)

        glob_loc_to_final_idx = [None] * len(glob_loc_to_final_idx_map)
        for glob_loc_idx in glob_loc_to_final_idx_map:
            img_idx = glob_loc_to_final_idx_map[glob_loc_idx]
            glob_loc_to_final_idx[img_idx] = glob_loc_idx

        assert all(x is not None for x in glob_loc_to_final_idx)

        index_map_path = os.path.join(
            index_map_output_dir, "glob_loc_index_to_img_index_%s.json" % split
        )
        if not os.path.exists(index_map_path):
            # Save the max range required for each ho
            with open(
                os.path.join(index_map_output_dir, "index_breakpoints_%s.txt" % split),
                "w",
            ) as outfile:
                print(index_breakpoints, file=outfile)
            # Save the index mapping
            with open(index_map_path, "w") as outfile:
                json.dump(glob_loc_to_final_idx, outfile, indent=2)
            print("Finished creating index map! Exiting.")
        else:
            print("Skipping saving index b/c file already exists: %s" % index_map_path)
        exit(0)
    elif merge_scenegraphs:  # If we're merging scenegraphs, merge & exit
        merged_sg_path = os.path.join(
            ori_save_path, "tmp_%s_sg_for_generation.json" % split
        )
        merge_scene_files(input_dirs=[save_path], out_path=merged_sg_path)
        print(
            "Done merging all %s scenegraphs into %s; now exiting."
            % (split, merged_sg_path)
        )
        exit(0)
    else:  # Else, load the data needed to start scenegraph generation
        index_map_path = os.path.join(
            index_map_output_dir, "glob_loc_index_to_img_index_%s.json" % split
        )
        if not os.path.exists(index_map_path):
            print(
                "Index mapping doesn't exist; please run the script with the --create_index flag first. Thank you."
            )
            exit(1)
        with open(index_map_path, "r") as outfile:
            glob_loc_to_final_idx = json.load(outfile)

    glob_loc_to_final_idx_map = {}
    for i, glob_loc_idx in enumerate(glob_loc_to_final_idx):
        glob_loc_to_final_idx_map[tuple(glob_loc_idx)] = i

    # 2) Generate scenegraphs & save them.

    # Setup filename templates
    num_digits = 6
    prefix = "%s_%s_" % (filename_prefix, split)
    img_template = "%s%%0%dd.png" % (prefix, num_digits)
    scene_template = "%s%%0%dd.json" % (prefix, num_digits)

    # Load in the scenegraphs we're potentially reusing
    # Map from dataset name, to list of scenegraph jsons
    if verbose:
        print("Load scenegraphs for reuse")
    reused_scene_graphs = {}
    for dset_name in reuse_dataset_names:
        sg_path = reuse_scene_graph_paths[dset_name]
        with open(sg_path, "r") as infile:
            sg_json = json.load(infile)["scenes"]
        reused_scene_graphs[dset_name] = sg_json

    # Create dictionary from global index to CLEVR reuse info
    # Check that we only reuse CLEVR images for the original images, as opposed to the paired images
    reuse_info = {}
    for x in objects_json["reuse_info"]:
        assert x["local_idx"] == 0
        reuse_info[x["global_idx"]] = x

    # Start processing the scenegraphs requested
    for offset in range(num_glob_indices_to_process):
        global_idx = start_idx + offset
        if verbose:
            print("Processing global_idx %i" % global_idx)

        if global_idx not in objects_json["image_data"]:
            if not global_idx >= len(objects_json["image_data"]):
                print(json.dumps(objects_json["image_data"]))
                print(global_idx)
                print(len(objects_json["image_data"]))

            assert global_idx >= len(objects_json["image_data"])
            continue

        img_idx = glob_loc_to_final_idx_map[(global_idx, 0)]
        filename = scene_template % img_idx
        if os.path.exists(os.path.join(save_path, filename)):
            print(
                "Skip global index %i; (image index for loc index 0): %i"
                % (global_idx, img_idx)
            )
            continue

        # Get list of pre-scenegraph descriptions, all these must share the
        # object positions & attributes.
        scenes_to_create = objects_json["image_data"][global_idx]

        # Check that all of the scenes that share camera/lighting/object positions
        # have the same number of objects
        assert len(set(len(x["objects"]) for x in scenes_to_create)) == 1

        # Check the indexing is all OK
        for tmp_i, x in enumerate(scenes_to_create):
            assert x["local_idx"] == tmp_i and x["global_idx"] == global_idx
            assert (global_idx, tmp_i) in glob_loc_to_final_idx_map

        # Check if we are trying to reuse a CLEVR image
        # If so, try using that as the base scene graph. Otherwise, generate a
        # random base scenegraph that has the right objects.
        # NOTE: The "base" scenegraph is a scenegraph that has not had any modifications
        #       i.e., either it doesn't match HO, or it matches HO and has *not*
        #       been altered to stop matching HO.
        base_img_idx = glob_loc_to_final_idx_map[(global_idx, 0)]

        # True only if we are reusing an image, and the image's scenegraph is
        # missing camera data.
        reusing_img_and_guessing_camera = False

        if global_idx in reuse_info:
            if verbose:
                print("Loading reuse info")
            base_img_reuse_info = reuse_info[global_idx]
            assert base_img_reuse_info["global_idx"] == global_idx
            assert base_img_reuse_info["local_idx"] == 0
            base_img_ori_idx = base_img_reuse_info["filename"][-10:-4]
            assert base_img_ori_idx.isnumeric()
            base_img_ori_idx = int(base_img_ori_idx)

            # Retrieve scene graph & check it makes sense
            if verbose:
                print("Checking scenegraph info makes sense")
            ori_sg = reused_scene_graphs[base_img_reuse_info["dataset"]][
                base_img_ori_idx
            ]
            assert ori_sg["image_index"] == base_img_ori_idx
            assert ori_sg["image_filename"] == base_img_reuse_info["filename"]

            # overwrite information from the previous dataset
            if verbose:
                print("Overwriting outdated scenegraph info", flush=True)
            ori_sg["image_index"] = base_img_idx
            ori_sg["image_filename"] = img_template % base_img_idx
            ori_sg["split"] = split

            # Map the index of the object within scenes_to_create[0] to its index
            # inside of the scenegraph.
            sg_objects_list = []
            for obj_idx, obj_struct in enumerate(ori_sg["objects"]):
                sg_objects_list.append((obj_idx, obj_struct_to_tuple(obj_struct)))
                sg_objects_list.sort(key=lambda x: str(x[1]))
            # Make sure we have the same number of objects
            assert len(sg_objects_list) == len(scenes_to_create[0]["objects"])
            obj_idx_to_sg_obj_idx = {}
            for scene_obj_idx in range(len(sg_objects_list)):
                obj_idx_to_sg_obj_idx[scene_obj_idx] = sg_objects_list[scene_obj_idx][0]

            proposed_base_sg = ori_sg

            # Mark that we are reusing this scenegraph by recording the information
            # required to retrieve the information from the original dataset
            # (namely, the name of the original dataset+split, and the filename of the
            #  image within that original dataset)
            proposed_base_sg["reuse_dataset"] = base_img_reuse_info["dataset"]
            proposed_base_sg["reuse_filename"] = base_img_reuse_info["filename"]

            # Randomize the camera & lighting settings; repeat until we find something
            # that works with the original CLEVR scenegraph [ignoring the exact pixel positions]
            # Note that these settings will most likely disagree with the actual
            # CLEVR image, so we will delete them when saving the scenegraph.
            if "camera_jitter" not in proposed_base_sg:
                assert (
                    "Lamp_Key" not in proposed_base_sg
                    and "Lamp_Back" not in proposed_base_sg
                    and "Lamp_Fill" not in proposed_base_sg
                )
                proposed_base_sg = randomize_lighting_and_camera(proposed_base_sg)
                while not validate_scenegraph(
                    proposed_base_sg, check_pixel_coordinates=False
                ):
                    proposed_base_sg = randomize_lighting_and_camera(proposed_base_sg)

                reusing_img_and_guessing_camera = True
            else:
                assert (
                    "Lamp_Key" in proposed_base_sg
                    and "Lamp_Back" in proposed_base_sg
                    and "Lamp_Fill" in proposed_base_sg
                )

        else:
            if verbose:
                print("Creating new random scenegraph for base")

            # produce random SG, and 1-to-1 mapping.
            proposed_base_sg = objects_to_sg(
                objects=scenes_to_create[0]["objects"],
                img_name=img_template % base_img_idx,
                img_idx=base_img_idx,
                split=split,
                use_gpu=use_gpu,
            )
            # We preserved the object order, so the order of objects in
            # scenes_to_create[0]["objects"], matches the order of objects in proposed_base_sg.
            # We create the mapping accordingly:
            obj_idx_to_sg_obj_idx = {}
            for i in range(len(scenes_to_create[0]["objects"])):
                obj_idx_to_sg_obj_idx[i] = i

        # Check that objects match what they are supposed to according to
        #  the mapping; and that we have the correct number of objects
        assert len(proposed_base_sg["objects"]) == len(scenes_to_create[0]["objects"])
        for scene_obj_idx, scene_obj in enumerate(scenes_to_create[0]["objects"]):
            sg_obj = proposed_base_sg["objects"][obj_idx_to_sg_obj_idx[scene_obj_idx]]
            assert scene_obj == obj_struct_to_tuple(sg_obj)

        if verbose:
            print("Creating set of proposal scenegraphs")
        # Modify the scenegraph to meet the other specs
        # NOTE: It is assumed that object order is fixed between all scenes in scenes_to_create
        proposed_scenegraphs = create_proposals_from_base_sg(
            proposed_base_sg,
            [x["objects"] for x in scenes_to_create],
            obj_idx_to_sg_obj_idx,
            global_idx=global_idx,
            glob_loc_to_final_idx_map=glob_loc_to_final_idx_map,
            img_template=img_template,
        )

        if verbose:
            print("Check proposals")
        # Until we have a camera/lighting/position that satisfies all tweaks
        # (including image validity & preservation of relationships), keep retrying
        # Note that we also need to adjust pixel coordinates of all tweaked scenes.
        # This is done via mutation.
        while not all(
            validate_scenegraph(
                x, check_pixel_coordinates=False, overwrite_pixel_coordinates=True
            )
            for x in proposed_scenegraphs[1:]
        ) and not all(
            validate_scenegraph(
                x, check_pixel_coordinates=True, overwrite_pixel_coordinates=False
            )
            for x in proposed_scenegraphs
        ):
            if verbose:
                print("Creating new proposal")
            # Mark that we are not reusing an image
            reusing_img_and_guessing_camera = False
            # produce random SG, and 1-to-1 mapping.
            proposed_base_sg = objects_to_sg(
                objects=scenes_to_create[0]["objects"],
                img_name=img_template % base_img_idx,
                img_idx=base_img_idx,
                split=split,
                use_gpu=use_gpu,
            )
            # We preserved the object order, so the order of objects in
            # scenes_to_create[0]["objects"], matches the order of objects in proposed_base_sg.
            # We create the mapping accordingly:
            obj_idx_to_sg_obj_idx = {}
            for i in range(len(scenes_to_create[0]["objects"])):
                obj_idx_to_sg_obj_idx[i] = i
            proposed_scenegraphs = create_proposals_from_base_sg(
                proposed_base_sg,
                [x["objects"] for x in scenes_to_create],
                obj_idx_to_sg_obj_idx,
                global_idx=global_idx,
                glob_loc_to_final_idx_map=glob_loc_to_final_idx_map,
                img_template=img_template,
            )

        # Strip lighting/camera position from any re-used images which didn't come with that information
        # Note again, we assume that only the zeroeth image may be reused.
        if reusing_img_and_guessing_camera:
            proposed_scenegraphs[0].pop("camera_jitter")
            proposed_scenegraphs[0].pop("Lamp_Key")
            proposed_scenegraphs[0].pop("Lamp_Back")
            proposed_scenegraphs[0].pop("Lamp_Fill")

        # Save the results as individual sg files so we can merge later.
        for local_idx, sg in enumerate(proposed_scenegraphs):
            img_idx = glob_loc_to_final_idx_map[(global_idx, local_idx)]
            filename = scene_template % img_idx
            with open(os.path.join(save_path, filename), "w") as outfile:
                json.dump(sg, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reusable_ims_config",
        default="reuse_config.json",
        type=str,
        help="Path to a .json config file specifying the scenegraphs file & corresponding image directories for image we can reuse.",
    )
    parser.add_argument(
        "--split",
        default="new",
        help="Name of the split for which we are rendering. This will be added to "
        + "the names of rendered images, and will also be stored in the JSON "
        + "scene structure for each image.",
    )
    parser.add_argument(
        "--output_path",
        default="output/scenes_to_render/",
        help="The scenegraphs as individual .json files at {output_path}/{split}",
    )
    parser.add_argument(
        "--index_map_output_path",
        default="output/scenes_to_render/",
        help="The mappings from (global_idx, loc_idx) to final image index will be saved at {index_map_output_path}/glob_loc_index_to_img_index_{split}.json if the --create_index map is set. Else, this is where we expect to find the map.",
    )
    parser.add_argument(
        "--objects_path",
        default="held_out_objects_train.json",
        help="Path to generated object tuples as a .json (pre-scenegraph specifications; i.e. # of objects, object attributes, and nothing else)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print debug messages; only good for small samples",
    )
    parser.add_argument(
        "--create_index",
        action="store_true",
        help="Instead of generating scenegraphs, create the index.",
    )
    parser.add_argument(
        "--start_idx",
        default=0,
        type=int,
        help="The *global_index* at which to start converting object specs to scene graphs. Setting "
        + "this to non-zero values allows you to distribute rendering across "
        + "multiple machines and recombine the results later.",
    )
    parser.add_argument(
        "--num_glob_indices_to_process",
        default=5,
        type=int,
        help="The number of scene graph *global indices* to render -- each global index corresponds to all images that use particular object positions & illuminations.",
    )
    parser.add_argument(
        "--use_gpu",
        default=1,
        type=int,
        help="Setting --use_gpu 0 disables GPU-accelerated rendering using CUDA. "
        + "Setting --use_gpu 1 enables. "
        "You must have an NVIDIA GPU with the CUDA toolkit installed for "
        + " --use_gpu 1 to work.",
    )
    parser.add_argument(
        "--filename_prefix",
        default="CLEVR_held_out",
        help="This prefix will be prepended to the rendered images and JSON scenes",
    )
    parser.add_argument(
        "--merge_scenegraphs",
        action="store_true",
        help="Instead of generating scenegraphs, merge the already created scenegraphs into a single .json file.",
    )

    argv = utils.extract_args()
    args = parser.parse_args(argv)

    with open(args.reusable_ims_config, "r") as infile:
        config = json.load(infile)

    main(
        save_path=args.output_path,
        split=args.split,
        index_map_output_dir=args.index_map_output_path,
        objects_path=args.objects_path,
        reuse_scene_graph_paths=config["reuse_scene_graph_paths"],
        reuse_dataset_names=config["reuse_dataset_names"],
        start_idx=args.start_idx,
        num_glob_indices_to_process=args.num_glob_indices_to_process,
        verbose=args.verbose,
        create_index=args.create_index,
        use_gpu=args.use_gpu,
        filename_prefix=args.filename_prefix,
        merge_scenegraphs=args.merge_scenegraphs,
    )
