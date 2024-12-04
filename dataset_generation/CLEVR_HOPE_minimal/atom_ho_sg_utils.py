from typing import Tuple, List, Optional, Iterable, Dict
import os
import json
import sys
from datetime import datetime as dt
import copy

from sg_utils import AbstractCLEVRTestCase, Variant, Directions

# Note: import bpy & mathutils is only going to work when run by Blender python
INSIDE_BLENDER = True
try:
    import bpy, bpy_extras
    from mathutils import Vector
except ImportError as e:
    INSIDE_BLENDER = False
    if __name__ == "__main__":
        print("This script is intended to be called from blender like this:")
        print()
        print("blender --background --python sg_utils.py")
    else:
        print(
            "sg_utils must be imported by blender python; blender --background --python "
        )
    sys.exit(1)
if INSIDE_BLENDER:
    try:
        import atom_utils
        from sg_question_utils import generate_questions
    except ImportError as e:
        print("\nERROR")
        print(
            "Running render_images.py from Blender and cannot import atom_utils.py or sg_questions_utils."
        )
        print("You may need to add a .pth file to the site-packages of Blender's")
        print("bundled python with a command like this:\n")
        print(
            "echo $PWD >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/atom_clevr.pth"
        )
        print("\nWhere $BLENDER is the directory where Blender is installed, and")
        print("$VERSION is your Blender version (such as 2.78).")
        sys.exit(1)


def sg_matches_variant(sg: dict, variant: dict) -> bool:
    """
    Return true iff sg is an instantiation of this variant.

    :param sg:
    :param variant:
    :return:
    """
    # (variant key, sg key)
    keys = [
        ("lamp_fill", "Lamp_Fill"),
        ("lamp_key", "Lamp_Key"),
        ("lamp_back", "Lamp_Back"),
        ("camera_jitter", "camera_jitter"),
    ]

    # print(sg.keys())
    # print('---------------------')
    # print(variant.keys())

    if not all(sg[sg_key] == variant[var_key] for (var_key, sg_key) in keys):
        # print("Fail due to keys")
        return False

    PRECISION = 4
    variant["positions"] = [
        (round(x, PRECISION), round(y, PRECISION)) for (x, y) in variant["positions"]
    ]

    # angles are fine, positions are only accurate to about 5 decimal places; so be it.
    # Mostly likely blender rounding things internally during full blown rendering.
    var_pos_theta = set(zip(variant["positions"], variant["thetas"]))
    sg_pos_theta = set(
        (
            (round(x["3d_coords"][0], PRECISION), round(x["3d_coords"][1], PRECISION)),
            x["rotation"],
        )
        for x in sg["objects"]
    )

    # if not sg_pos_theta.issubset(var_pos_theta):
    #    print("Fail pos/angle")
    #    print(var_pos_theta)
    #    print(sg_pos_theta)

    # sg may not use all objects in variant; that's fine.
    return sg_pos_theta.issubset(var_pos_theta)


def get_suitable_variant(
    input_variants: List[dict],
    directions_by_var: List[Directions],
    next_idx: int,
    test_cases: List[AbstractCLEVRTestCase],
) -> Tuple[Variant, int, Directions, int]:
    """
    Return variant, variant_idx, directions_vec, reuse_var_index

    :param input_variants:
    :param directions_by_var:
    :param next_idx:
    :param test_cases:
    :return:
    """

    def pass_test_cases(variant: Variant, directions: Directions):
        for tc in test_cases:
            if tc.must_reject_variant(variant=variant, directions=directions):
                return False
        return True

    # NOTE: Assumes all variants should work
    next_dir = directions_by_var[next_idx]

    # Create variant object
    next_var = input_variants[next_idx]
    positions = [tuple(x) for x in next_var["positions"]]
    thetas = next_var["thetas"]

    next_var = Variant(
        camera_jitter=tuple(next_var["camera_jitter"]),
        lamp_key=tuple(next_var["lamp_key"]),
        lamp_back=tuple(next_var["lamp_back"]),
        lamp_fill=tuple(next_var["lamp_fill"]),
    )
    for pos, theta in zip(positions, thetas):
        next_var.add_object(pos=pos, theta=theta)

    # To make less silly; go to next in case of failure. For now this should work though
    assert pass_test_cases(variant=next_var, directions=next_dir)
    return next_var, next_idx, next_dir, next_idx + 1


def fix_scene(input_scene) -> str:
    input_scene = copy.deepcopy(input_scene)
    input_scene.pop("image_filename")
    input_scene.pop("image_index")
    input_scene.pop("split")
    input_scene.pop("relationships")
    input_scene.pop("directions")

    for o in input_scene["objects"]:
        o.pop("pixel_coords")
        o["3d_coords"][-1] = -1
        # 4 decimals. Argh.
        o["3d_coords"][0] = round(o["3d_coords"][0], 4)
        o["3d_coords"][1] = round(o["3d_coords"][1], 4)

    return json.dumps(input_scene, sort_keys=True)


# Modification of generate_sg_and_questions for resuing existing images/scenegraphs.
def reuse_sg_and_generate_questions(
    test_cases: List[Tuple[AbstractCLEVRTestCase, int]],
    input_variants: List[dict],
    input_scenes: List[dict],  # NOTE: scenegraph should contain filename
    split: str = "val",
    version: float = 1.0,
) -> List[dict]:
    """
    Generate a separate questions .json for each test case in test_cases.

    Attempts to reuse existing images & variants (e.g., as created in CLEVR-ATOM)

    :param split: Name of the split
    :param input_variants: List of variant dictionaries; (see sg_utils.Variant's __str__ function).
                           More specifically, we're expecting the variants output of generate_sg_and_questions
                           e.g., s3://systematicity/datasets_dir/ori_atom_CLEVR_val_v1.0/scene_details/variants.json
    :param input_scenes: List of scenes. I *think*  the output of generate_sg_and_questions
                         e.g., full contents of s3://systematicity/datasets_dir/ori_atom_CLEVR_val_v1.0/scene_details/scenes/
    :param test_cases: List of (TestCase, requested # of variants) tuples.
    :return:
    """
    # 1) Organize input scenegraphs by variant:
    input_scenes_by_var = [[]] * len(input_variants)
    directions_by_var = [None] * len(input_variants)
    for sg in input_scenes:
        # print('----------------------------')
        assert "image_filename" in sg
        index = [
            i for (i, var) in enumerate(input_variants) if sg_matches_variant(sg, var)
        ]
        # print(index)
        assert len(index) <= 1
        # SKIP IMAGES THAT DON'T match a provided variant.
        if len(index) == 0:
            continue

        index = index[0]
        input_scenes_by_var[index].append(sg)

        if directions_by_var[index] is None:
            directions_by_var[index] = Directions(  # type: ignore
                behind=tuple(sg["directions"]["behind"]),
                front=tuple(sg["directions"]["front"]),
                left=tuple(sg["directions"]["left"]),
                right=tuple(sg["directions"]["right"]),
            )

    available_scene_graphs = {fix_scene(sg): sg for sg in input_scenes}
    scene_graph_names = {
        sg_str: (sg["image_filename"], sg["image_index"])
        for (sg_str, sg) in available_scene_graphs.items()
    }

    # 2) Begin actual processing

    scene_graphs = set()  # Final set of scenegraphs used
    # final_variants = []
    # scene_graph_names = {} # map scene to filename & index; instead init from input scenegraphs

    header = {
        "info": {
            "date": dt.today().strftime("%m/%d/%Y"),
            "version": version,
            "split": split,
            "license": "Creative Commons Attribution (CC-BY 4.0)",
        },
        "questions": [],
    }

    final_questions = [
        copy.deepcopy(header) for i in range(len(test_cases))
    ]  # Each test case gets its own dict of questions.

    total_variants = max(x[1] for x in test_cases)

    # Sanity check we have enough variants.
    assert len(input_variants) >= total_variants

    curr_img_index = 0
    reuse_var_index = 0  # index of next variant up for reuse that we *haven't* used yet
    reused_variants = []

    for var_idx in range(total_variants):
        # Get index of test case so we put it in the right place in final_questions (since the test cases in use may vary)
        curr_test_cases = [
            (i, x[0]) for (i, x) in enumerate(test_cases) if var_idx < x[1]
        ]

        """
        variant, directions_vec, cam_vec_tmp_debug = generate_variant(test_cases=[x[1] for x in curr_test_cases],
                                                                                variant_id=var_idx,
                                                                                desired_num_objects=desired_num_objects,
                                                                                min_pixels_per_object=min_pixels_per_object,
                                                                                img_width=img_width,
                                                                                img_height=img_height)
        final_variants.append(variant)
        """
        # Look for the next available suitable variant in out input
        variant, variant_idx, directions_vec, reuse_var_index = get_suitable_variant(
            input_variants,
            directions_by_var,  # type: ignore
            reuse_var_index,
            test_cases=[x[1] for x in curr_test_cases],
        )
        reused_variants.append((variant, variant_idx))

        # Check we'll have enough variants left for reuse
        assert len(input_variants) - reuse_var_index + 1 >= total_variants - var_idx - 1

        for i, tc in curr_test_cases:
            tc_scenes, tc_filenames, tc_questions = tc.generate_full_sgs(
                variant, variant_id=var_idx, directions=directions_vec
            )
            for scene, filename in zip(tc_scenes, tc_filenames):
                if scene not in available_scene_graphs:  # scene_graphs:
                    # Can't reuse the scene b/c it doesn't already exist
                    """
                    # If this is a new scene, add it to the set & record its name
                    scene_graphs.add(scene)
                    scene_graph_names[scene]= (filename + "_%06d.png"%curr_img_index, curr_img_index)
                    curr_img_index += 1
                    """

                    tmp_scene = json.loads(scene)
                    # print(tmp_scene.keys())
                    # print(available_scene_graphs)
                    # print(available_scene_graphs.keys())
                    tmp_av_sg = list(available_scene_graphs.keys())
                    # print(tmp_av_sg[:3])
                    tmp_av_sg = tmp_av_sg[0]
                    tmp_av_sg = json.loads(tmp_av_sg)
                    # print(tmp_av_sg.keys())

                    for tmp_key in [
                        "objects"
                    ]:  # ['camera_jitter', 'Lamp_Back', 'Lamp_Key', 'objects', 'Lamp_Fill']:
                        print("======================")
                        print(tmp_key)
                        print(tmp_scene[tmp_key])
                        print(tmp_av_sg[tmp_key])

                    raise NotImplementedError
                elif scene not in scene_graphs:
                    # Reuse the scene, and mark it as in reuse
                    scene_graphs.add(scene)

            for scene in tc_questions:
                for question in tc_questions[scene]:
                    # Set the image filename & index in the question
                    # dictionary, now that we know these values.
                    img_filename, img_idx = scene_graph_names[scene]
                    question["image_filename"] = img_filename
                    question["image_index"] = img_idx
                    question["image"] = os.path.splitext(img_filename)[0]
                    question["split"] = split
                    final_questions[i]["questions"].append(question)

    # Do NOT process scenegraphs, as we'll have to reuse the original CLEVR-ATOM scenegraph file.
    """
    # merge, format & return questions & scene graphs.
    final_scene_graphs = copy.deepcopy(header)
    final_scene_graphs.pop('questions', None)
    final_scene_graphs['scenes'] = []
    for sg in scene_graphs:
        img_filename, img_idx = scene_graph_names[sg]
        # Don't need to include directions & relationships as those will
        # be re-computed when the images are rendered, and the final
        # scene_graph .json file is created.
        sg_json = json.loads(sg)
        sg_json['image_index']=img_idx
        sg_json['image_filename']=img_filename
        sg_json['split']=split

        final_scene_graphs['scenes'].append(sg_json)
    # Image index is a terrible way to sort scene graphs (filename would be better)
    # but we have no choice due to CLEVR loader requiring this order
    final_scene_graphs['scenes'].sort(key=lambda x:x["image_index"])
    """
    for tc_dict in final_questions:
        tc_dict["questions"].sort(
            key=lambda x: (x["row_index"], x["variant_index"], x["col_index"])
        )
        question_idx = 0
        for question in tc_dict["questions"]:
            question["question_index"] = question_idx
            question_idx += 1

    assert curr_img_index == 0  # ASSERT NO IMAGES CREATED.
    return final_questions
