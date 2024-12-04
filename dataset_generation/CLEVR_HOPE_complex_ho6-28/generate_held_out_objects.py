"""
New script for generating the held-out experment

Generates the attributes & number of objects in the required images.
Reuses from CLEVR where possible.

Note: 10 large objects prohibited, b/c it couldn't be made to fit in the scene.
"""
import argparse
import copy

import yaml
from typing import List, Iterable, Tuple, Dict, Optional
import json
from collections import defaultdict, Counter
import random
from tqdm import tqdm

from utils import (
    CLEVR_COLORS,
    CLEVR_MATERIALS,
    CLEVR_SHAPES,
    CLEVR_SIZES,
    load_ho_config,
)

# List of possible held-out combinations
HO_choices = load_ho_config()

DISABLE_TQDM = False

A_choices = None


# Represent an n-object image as an n-tuple of object 4-tuples (size, col, sha, mat)
def standardize_img_rep(
    x: Iterable[Tuple[str, str, str, str]]
) -> Tuple[Tuple[str, str, str, str]]:
    """
    Given an iterable collection of objects, each object represented by
    a (size, col, sha, mat) tuple, we return a tuple of these tuples.
    Additionally, to ensure that we can deduplication/lookup these tuples,
    we sort them alphabetically w.r.t. the string representation of the object
    tuples. (hacky, but it works)
    :param x:
    :return:
    """
    x = list(x)
    x.sort(key=lambda x: str(x))
    return tuple(x)


def get_random_scene(num_obj: int):
    """
    Return a unif. random scene, represented by a tuple of objects.
    Each object is represented by a 4-tuple representing of uniformly random chosen attributes.
    The objects are sorted in alphabetic order

    NOTE: A 10 object scene is not allowed to be all large objects, because
          that causes problems fitting them all in the scene.

    :return: a size {num_obj} tuple of 4-tuples of (size, color, shape, material)
    """
    scene = standardize_img_rep(
        (
            random.choice(CLEVR_SIZES),
            random.choice(CLEVR_COLORS),
            random.choice(CLEVR_SHAPES),
            random.choice(CLEVR_MATERIALS),
        )
        for i in range(num_obj)
    )

    if num_obj == 10 and all(obj[0] == "large" for obj in scene):
        return get_random_scene(num_obj)

    return scene


def load_reusable_images(
    reuse_scene_graph_paths: List[str],
) -> List[Tuple[Tuple, Tuple]]:
    """
    PRECONDITION: input scenegraphs should approximate IID unif. random data.
                        That'll hold for CLEVR-TRAIN, but may not for other ims

    :param reuse_scene_graph_paths: Dict of datasetname to List of scene graph paths
    :return: List of tuples of (scene tuple, (dataset_name, image_filename))
             The scene tuple is in the standard scene representation tuple format
    """
    # List of tuples of (scene tuple, (dataset_name, image_filename))
    reusable_imgs = []
    for dataset_name in reuse_scene_graph_paths:
        sg_path = reuse_scene_graph_paths[dataset_name]
        with open(sg_path, "r") as infile:
            sgs_json = json.load(infile)

        for sg_json in sgs_json["scenes"]:
            sg_tuple = []
            for obj_struct in sg_json["objects"]:
                sg_tuple.append(
                    (
                        obj_struct["size"],
                        obj_struct["color"],
                        obj_struct["shape"],
                        obj_struct["material"],
                    )
                )
            sg_tuple = standardize_img_rep(sg_tuple)
            reusable_imgs.append((sg_tuple, (dataset_name, sg_json["image_filename"])))

    return reusable_imgs


def attr_match(attributes: Tuple, template: Tuple) -> bool:
    """
    Return true iff template[i] is None or template[i] == attributes[i] for all i

    :param attributes: Tuple of strings
    :param template: Tuple of strings or None
    :return:
    """
    assert all(x is not None for x in attributes)

    for temp_attr, attr in zip(template, attributes):
        if temp_attr is not None and temp_attr != attr:
            return False
    return True


def mask_ho(
    scene: Tuple[Tuple[str, str, str, str]],
    ho: Tuple[Optional[str], Optional[str], Optional[str], Optional[str]],
    replacements: Tuple[str, str, str, str],
) -> Tuple[Tuple[str, str, str, str]]:
    """
    Given a scene, modify only the objects that match the held-out combination.
    For objects matching the held-out combination, the held-out attributes are
    replaced with the values in replacements.

    :param scene:
    :param ho: the held-out combination, represented by a 4-tuple of
               (size, color, shape, material) values, with irrelevant values being None
    :param replacements: 4-tuple of replacement (size, color, shape, material) values
    :return: The modified scene. NOTE: The order of objects is preserved from {scene}
    """
    scene = list(scene)
    final = []
    for obj in scene:
        # Objects not matching the held-out combination are added directly
        if not attr_match(obj, ho):
            final.append(obj)
        # Objects matching the held-out combination are masked
        else:
            new_obj = []
            for obj_val, ho_val, replace_val in zip(obj, ho, replacements):
                if ho_val is not None:
                    assert obj_val == ho_val
                    new_obj.append(replace_val)
                else:
                    new_obj.append(obj_val)
            new_obj = tuple(new_obj)
            final.append(new_obj)

    # NOTE: DO NOT STANDARDIZE the image representation, so we can preserve the
    # same order as the original image.
    final = tuple(final)
    # final = standardize_img_rep(final)

    # check no attributes are None
    assert all(attr_x is not None for obj_x in final for attr_x in obj_x)
    return final


def generate_required_obj_attributes(ims_with_HO: int, ims_without_HO: int, existing):
    """
    Generate all of the image descriptions required to have ims_with_HO images
    per held-out combination (each paired with a non-matching image), and
    ims_without_HO images per held-out combination that doesn't match the
    held-out combination.

    Note that each image is described as a list of object attributes. No
    position or lighting data is established by this method.

    PRECONDITION: {existing} images are unif. random & IID!

    :param ims_with_HO: How many images you want that match per HO
    :param ims_without_HO: How many images you want that don't match, per HO
    :param existing: Images available for reuse, represented as a list of (scene tuple, (dataset name, filename)) tuples.
    :return: global_ims, found_images_match, found_images_match_paired, found_images_no_match, global_ims_clevr
             global_ims is the final collection of all images that need to be
             generated. Note that it represents only the images' objects & attributes,
             not the position/lighting/etc...
             It is a dictionary from index to a list of images which share the same positions/illuminations.
             found_images_match is a dictionary from held-out combination, to a list of (global_idx, sub_idx) tuples.
                i.e., global_ims[global_idx][sub_idx] specifies an image that matches our combination
            found_images_match_paired is a dictionary from held-out combination, to a list of (global_idx, sub_idx) tuples.
                i.e., global_ims[global_idx][sub_idx] specifies an image that was altered to no longer match our combination
                      Furthermore, found_images_match_paired[x][y] corresponds to found_images_match[x][y] for arbitrary
                      held-out combination x & index y
            found_images_no_match is a dictionary from held-out combination, to a list of (global_idx, sub_idx) tuples.
            that do not match the combinations, and were *NOT* altered to make that so.
            global_ims_clevr is a dictionary of (global img index, sub index) to the CLEVR details of the reused image.
            The CLEVR details are (dataset name, img filename) tuple
    """
    iid_ims_by_size = {}
    for i in range(3, 11):
        iid_ims_by_size[i] = [
            j for j in range(len(existing)) if len(existing[j][0]) == i
        ]

    # IID ims is a list of lists:
    # Specifically, list of [(scene, (datasetname, filename)), (scene, (datasetname, filename)), ...] lists
    # Inside each sublist, these images should share the same object positions/lighting/etc...
    # If the image is not reused (or is an alteration of a reused image), then (datasetname, filename) is replaced with None
    iid_ims = [[x] for x in existing]

    # Images that matched HO
    found_images_match = {
        key: [] for key in HO_choices
    }  # For each choice of held-out combination, a list of indices: (iid_id, which inside) for matching
    # Alterations of images so that they no longer match HO
    found_images_match_paired = {
        key: [] for key in HO_choices
    }  # For each choice of held-out combination, a list of indices: (iid_id, which inside) for altered to not match

    # Images that never matched HO
    found_images_no_match = {
        key: [] for key in HO_choices
    }  # For each choice of held-out combination, a list of indices: (iid_id, which inside)

    for ho in HO_choices:
        # Get the indices we haven't tried yet for this ho combination.
        remaining_iid_ims_by_size = copy.deepcopy(iid_ims_by_size)

        # Get images not matching ho
        for i in tqdm(
            range((ims_without_HO)), disable=DISABLE_TQDM
        ):  # At most this many images if we have no deduplication
            num_objects = random.randint(3, 10)  # Chose # of objects in the scene

            # Terminate once we have enough non-matching images for this held-out combinations
            if len(found_images_no_match[ho]) >= ims_without_HO:
                break

            # Given num_objects, keep trying until we produce a useful scene containing num_objects objects
            useful_image = False
            while not useful_image:
                # If we've run out of IID images with this many objects, make a new one
                if len(remaining_iid_ims_by_size[num_objects]) == 0:
                    img = get_random_scene(num_objects)
                    iid_ims.append([(img, None)])
                    remaining_iid_ims_by_size[num_objects].append(len(iid_ims) - 1)
                    iid_ims_by_size[num_objects].append(len(iid_ims) - 1)

                # Try the next IID image with this many objects
                next_iid_idx = remaining_iid_ims_by_size[num_objects].pop(0)

                # Retrieve the scene details (i.e., object attributes), and any
                # details if we've reused the image from an existing dataset
                img, clevr_details = iid_ims[next_iid_idx][0]

                # Update counts if the scene does not match the held-out combination, and we needed another such image
                if all([not attr_match(x, ho) for x in img]):  # Does NOT match ho
                    useful_image = True
                    found_images_no_match[ho].append((next_iid_idx, 0))

        # Reset the indices we haven't tried yet for this ho combination
        # [NOTE: this is fine b/c no image can both match & not match HO, ergo
        # any left is just sampling from P(unif|has HO) which is what we want]
        remaining_iid_ims_by_size = copy.deepcopy(iid_ims_by_size)
        # Get images matching HO
        for i in tqdm(
            range((ims_with_HO)), disable=DISABLE_TQDM
        ):  # At most this many images if we have no deduplication
            num_objects = random.randint(3, 10)  # Chose # of objects in the scene
            # num_objects = (i % 8) + 3

            # Terminate once we have enough matching images for this held-out combinations
            if len(found_images_match[ho]) >= ims_with_HO:
                break

            # Given num_objects, keep trying until we produce a useful scene containing num_objects objects
            useful_image = False
            while not useful_image:
                # If we've run out of IID images with this many objects, make a new one
                if len(remaining_iid_ims_by_size[num_objects]) == 0:
                    img = get_random_scene(num_objects)
                    iid_ims.append([(img, None)])
                    remaining_iid_ims_by_size[num_objects].append(len(iid_ims) - 1)
                    iid_ims_by_size[num_objects].append(len(iid_ims) - 1)

                # Try the next IID image with this many objects
                next_iid_idx = remaining_iid_ims_by_size[num_objects].pop(0)

                # Retrieve the scene details (i.e., object attributes), and any
                # details if we've reused the image from an existing dataset
                img, clevr_details = iid_ims[next_iid_idx][0]

                if any([attr_match(x, ho) for x in img]):  # Match at least one object
                    useful_image = True

                    # Record that we have a paired image here
                    paired_im = mask_ho(scene=img, ho=ho, replacements=A_choices)
                    paired_im = (
                        paired_im,
                        None,
                    )  # Note that we are *not* reusing this img
                    if paired_im not in iid_ims[next_iid_idx]:
                        iid_ims[next_iid_idx].append(paired_im)

                    found_images_match_paired[ho].append(
                        (next_iid_idx, iid_ims[next_iid_idx].index(paired_im))
                    )
                    found_images_match[ho].append((next_iid_idx, 0))

    # Consolidate results by deleting unused iid images
    iid_idx_to_global_idx = dict()
    global_img_index = 0  # NOTE: This is *not* the final index used for images; rather this is the number of unique camera/lightning choices [i.e., we're counting all versions of a given image, as one image]
    global_ims = (
        {}
    )  # List from global img index to a LIST of scenes that need the same lighting/positions/etc...; the original comes first, followed by any altered versions of the scene
    global_ims_clevr = (
        {}
    )  # Map of (global img index, sub index) to the CLEVR details of the reused image

    # Populate final dictionaries with the actual images needed
    for ho in HO_choices:
        for found_images in [
            found_images_match,
            found_images_match_paired,
            found_images_no_match,
        ]:
            for iid_idx, local_idx in found_images[ho]:
                # If we've already recorded that we need this image, double check
                # everything is fine
                if iid_idx in iid_idx_to_global_idx:
                    glob_idx = iid_idx_to_global_idx[iid_idx]
                    assert global_ims[glob_idx] == [x[0] for x in iid_ims[iid_idx]]
                    assert len(global_ims[glob_idx]) >= local_idx + 1
                # Otherwise, add this image to the collection of global images
                else:
                    iid_idx_to_global_idx[iid_idx] = global_img_index
                    global_ims[global_img_index] = [x[0] for x in iid_ims[iid_idx]]
                    # Record if this image is reused from an existing dataset
                    if iid_ims[iid_idx][0][1] is not None:
                        global_ims_clevr[(global_img_index, local_idx)] = iid_ims[
                            iid_idx
                        ][0][1]
                    # Assert that no paired images are reused from an existing dataset
                    assert all([x[1] is None for x in iid_ims[iid_idx][1:]])
                    global_img_index += 1

    # Update the found_images dictionaries to use global indices instead of IID indices
    for found_images in [
        found_images_match,
        found_images_match_paired,
        found_images_no_match,
    ]:
        for ho in HO_choices:
            found_images[ho] = [
                (iid_idx_to_global_idx[found_iid_idx], loc_idx)
                for (found_iid_idx, loc_idx) in found_images[ho]
            ]

    # Assert we've met the matching & non-matching quotas
    assert all(
        [len(x) >= ims_without_HO for x in found_images_no_match.values()]
    ) and all([len(x) >= ims_with_HO for x in found_images_match.values()])

    # Assert that no scene both matches & doesn't match any specific HO
    # (more accurately, make sure that the *original* scene before any changes didn't both match & not match)
    for ho in HO_choices:
        assert set(found_images_match[ho]).isdisjoint(set(found_images_no_match[ho]))

    # Make sure every matching image is paired with a non-matching alteration
    for ho in HO_choices:
        assert len(found_images_match[ho]) == len(found_images_match_paired[ho])
        for img_ho_idx, image_altered_idx in zip(
            found_images_match[ho], found_images_match_paired[ho]
        ):
            # same global index (i.e., same object positions & illumination)
            assert img_ho_idx[0] == image_altered_idx[0]
            # Different exact object attributes
            assert img_ho_idx[1] != image_altered_idx[1]
            # The unaltered image should be zeroeth among all versions of this specific image
            assert img_ho_idx[1] == 0

    # Assert every global index is used somewhere
    all_global_indices = set()
    for ho in HO_choices:
        all_global_indices.update(x[0] for x in found_images_no_match[ho])
        all_global_indices.update(x[0] for x in found_images_match[ho])
    assert min(all_global_indices) == 0
    assert max(all_global_indices) == global_img_index - 1
    assert len(all_global_indices) == global_img_index
    all_full_indices = {
        (global_idx, local_idx)
        for global_idx in global_ims
        for local_idx in range(len(global_ims[global_idx]))
    }
    all_used_indices = set()
    for full_indices in found_images_match.values():
        all_used_indices.update(full_indices)
    for full_indices in found_images_no_match.values():
        all_used_indices.update(full_indices)
    for full_indices in found_images_match_paired.values():
        all_used_indices.update(full_indices)
    assert all_full_indices == all_used_indices

    # Assert that matching images match, non-matching don't match
    for ho in HO_choices:
        for glob_idx, loc_idx in found_images_match[ho]:
            assert any(attr_match(obj, ho) for obj in global_ims[glob_idx][loc_idx])
        for glob_idx, loc_idx in found_images_no_match[ho]:
            assert not any(attr_match(obj, ho) for obj in global_ims[glob_idx][loc_idx])
        for glob_idx, loc_idx in found_images_match_paired[ho]:
            assert not any(attr_match(obj, ho) for obj in global_ims[glob_idx][loc_idx])

    # Assert that non-matching images are always zeroeth in the list of images with the same lighting/camera/etc...
    for ho in HO_choices:
        for scene_indices in found_images_no_match[ho]:
            assert scene_indices[1] == 0

    for glob_idx, scene_list in global_ims.items():
        # Check all modifications to the same image have the same number of objects
        assert len({len(scene) for scene in scene_list}) == 1
        # Assert no None's have made their way through into the attributes
        for loc_idx, scene in enumerate(scene_list):
            assert all(attr_x is not None for obj_x in scene for attr_x in obj_x)
            # Make sure we're not trying to create a 10-object scene with only large objects
            if len(scene) == 10:
                assert (
                    any(obj_x[0] == "small" for obj_x in scene)
                    or (glob_idx, loc_idx) in global_ims_clevr
                )
                if (not any(obj_x[0] == "small" for obj_x in scene)) and (
                    glob_idx,
                    loc_idx,
                ) in global_ims_clevr:
                    print("CLEVR ALL LARGE:")
                    print(scene)

    return (
        global_ims,
        found_images_match,
        found_images_match_paired,
        found_images_no_match,
        global_ims_clevr,
    )


def generate_test_data(num_test_ims: int):
    iid_ims = []
    iid_ims_by_size = {i: [] for i in range(3, 11)}
    found_images_match = {ho: [] for ho in HO_choices}  # Map to list of iid indices

    for ho in HO_choices:
        remaining_ims_by_size = copy.deepcopy(iid_ims_by_size)

        for i in range(num_test_ims):
            num_obj = random.randint(3, 10)

            useful_im = False
            while not useful_im:
                if len(remaining_ims_by_size[num_obj]) == 0:
                    iid_ims.append(get_random_scene(num_obj))
                    remaining_ims_by_size[num_obj].append(len(iid_ims) - 1)
                    iid_ims_by_size[num_obj].append(len(iid_ims) - 1)

                next_iid_idx = remaining_ims_by_size[num_obj].pop(0)
                img = iid_ims[next_iid_idx]

                if any([attr_match(x, ho) for x in img]):  # Match at least one object
                    useful_im = True
                    found_images_match[ho].append(next_iid_idx)

    # Consolidate iid_ims into final collection
    # Keep only the images used in global_ims, and save the mapping of indices
    # from iid_ims to global_ims indices
    iid_idx_to_global_idx = {}
    global_ims = []
    for ho in HO_choices:
        for iid_idx in found_images_match[ho]:
            if iid_idx not in iid_idx_to_global_idx:
                global_ims.append(iid_ims[iid_idx])
                iid_idx_to_global_idx[iid_idx] = len(global_ims) - 1
                assert any(
                    [
                        attr_match(x, ho)
                        for x in global_ims[iid_idx_to_global_idx[iid_idx]]
                    ]
                )
            else:
                assert iid_ims[iid_idx] == global_ims[iid_idx_to_global_idx[iid_idx]]
                assert any(
                    [
                        attr_match(x, ho)
                        for x in global_ims[iid_idx_to_global_idx[iid_idx]]
                    ]
                )

    # Convert the indices
    for ho in HO_choices:
        found_images_match[ho] = [
            iid_idx_to_global_idx[iid_idx] for iid_idx in found_images_match[ho]
        ]

    # Final check we're OK
    for ho in HO_choices:
        assert len(found_images_match[ho]) == num_test_ims
        for global_idx in found_images_match[ho]:
            assert any([attr_match(x, ho) for x in global_ims[global_idx]])

    # NOTE: local indices are all 0 b/c we don't have any paired images in the test set.
    final_results = {
        "image_data": {
            glob_idx: [{"global_idx": glob_idx, "local_idx": 0, "objects": scene}]
            for glob_idx, scene in enumerate(global_ims)
        },
        "reuse_info": [],
        "match_ho": {
            str(ho): [
                {"global_idx": glob_idx, "local_idx": 0}
                for glob_idx in found_images_match[ho]
            ]
            for ho in HO_choices
        },
        "match_ho_pair": {str(ho): [] for ho in HO_choices},
        "not_match_ho": {str(ho): [] for ho in HO_choices},
        "split": "test",
    }

    for glob_idx in final_results["image_data"]:
        assert len(final_results["image_data"][glob_idx]) == 1
        assert final_results["image_data"][glob_idx][0]["local_idx"] == 0

    return final_results


def distribution_checking(
    found_images_match,
    found_images_no_match,
    found_images_match_paired,
    global_ims,
    name: str,
):
    print()
    print(
        f"{name} Scene Sizes by HO, divided between matching & non-matching subsets (should be approx balanced):"
    )
    for ho in HO_choices:
        print(ho)
        print("Match")
        print(
            Counter(
                len(global_ims[full_idx[0]][0]) for full_idx in found_images_match[ho]
            )
        )

        print("No Match")
        print(
            Counter(
                len(global_ims[full_idx[0]][0])
                for full_idx in found_images_no_match[ho]
            )
        )

    print()
    print(
        f"{name} Scene -- object distribution [note, averaged over individual objects; not ideal design but good enough b/c objects should be taken independently within a scene]"
    )
    for ho in HO_choices:
        # size, color, shape, material
        print(ho)
        # Number of objects where we match ho; only one object has to match HO,
        # so should be (give or take) a uniform distribution.
        total_match_possibilities = (
            len(CLEVR_SIZES)
            * len(CLEVR_COLORS)
            * len(CLEVR_SHAPES)
            * len(CLEVR_MATERIALS)
        )
        print(f"Match; expected probability: approx. {1 / total_match_possibilities}")
        objects = []
        for global_idx, loc_idx in found_images_match[ho]:
            objects += global_ims[global_idx][loc_idx]
        counts = Counter(objects)
        for key in counts:
            counts[key] /= len(objects)
        if len(counts) > 0:
            print(f"Most likely: {counts.most_common(1)[0]}")
            print(f"Least likely: {counts.most_common()[-1]}")
            print(f"count sums: {sum(counts.values())}")
        else:
            print(f"SKIP: Counts is empty")
        print(
            f"proportion of objects seen: {len(set(objects)) / total_match_possibilities}"
        )

        # Number of choices of the attributes not in ho
        total_no_match_possibilities = (
            (len(CLEVR_SIZES) if ho[0] is None else 1)
            * (len(CLEVR_COLORS) if ho[1] is None else 1)
            * (len(CLEVR_SHAPES) if ho[2] is None else 1)
            * (len(CLEVR_MATERIALS) if ho[3] is None else 1)
        )
        required = [
            (len(CLEVR_SIZES) if ho[0] is not None else None),
            (len(CLEVR_COLORS) if ho[1] is not None else None),
            (len(CLEVR_SHAPES) if ho[2] is not None else None),
            (len(CLEVR_MATERIALS) if ho[3] is not None else None),
        ]
        required = [r for r in required if r is not None]
        assert len(required) == 2
        # multiply the choices of attributes not in HO, with all of the non-matching
        # choices for the attributes in HO.
        # For ho' being a tuple of the two choices, then
        # either ho'[1] + any non-ho'[0] choice for [0],
        # or ho'[0] + any non-ho'[1] choices for [1];
        # or avoiding both ho'[0] and ho'[1]
        total_no_match_possibilities *= (
            (required[0] - 1)
            + (required[1] - 1)
            + (required[0] - 1) * (required[1] - 1)
        )
        print(f"No Match; expected probability: {1 / total_no_match_possibilities}")
        objects = []
        for global_idx, loc_idx in found_images_no_match[ho]:
            objects += global_ims[global_idx][loc_idx]
        counts = Counter(objects)
        for key in counts:
            counts[key] /= len(objects)
        # print(counts)
        if len(counts) > 0:
            print(f"Most likely: {counts.most_common(1)[0]}")
            print(f"Least likely: {counts.most_common()[-1]}")
            print(f"count sums (must be 1): {sum(counts.values())}")
        else:
            print(f"SKIP: Counts is empty")
        print(
            f"proportion of objects seen (ideally 1): {len(set(objects)) / total_no_match_possibilities}"
        )

        print(
            f"Paired images for Match; expected probability: ideally {1 / total_no_match_possibilities}"
        )
        objects = []
        for global_idx, loc_idx in found_images_match_paired[ho]:
            objects += global_ims[global_idx][loc_idx]
        counts = Counter(objects)
        for key in counts:
            counts[key] /= len(objects)
        # print(counts)
        if len(counts) > 0:
            print(f"Most likely: {counts.most_common(1)[0]}")
            print(f"Least likely: {counts.most_common()[-1]}")
            print(f"count sums (must be 1): {sum(counts.values())}")
        else:
            print(f"SKIP: Counts is empty")
        print(
            f"proportion of objects seen (ideally 1): {len(set(objects)) / total_no_match_possibilities}"
        )


def check_size_for_no_atom(
    found_images_no_match, found_images_match_paired, global_ims, name
):
    print(f"{name} Set: # of remaining images after purging occurences of either atom:")
    for ho in HO_choices:
        sub_HO_choices = [
            tuple(ho[i] if i != j else None for i in range(len(ho)))
            for j in range(len(ho))
            if ho[j] is not None
        ]
        print(ho)
        print("No Match:")
        count = 0
        for global_idx, loc_idx in found_images_no_match[ho]:
            count += not any(
                attr_match(obj, sub_ho)
                for obj in global_ims[global_idx][loc_idx]
                for sub_ho in sub_HO_choices
            )
        print(f"{count} of {len(found_images_no_match[ho])}")
        print("Paired ims:")
        count = 0
        for global_idx, loc_idx in found_images_match_paired[ho]:
            count += not any(
                attr_match(obj, sub_ho)
                for obj in global_ims[global_idx][loc_idx]
                for sub_ho in sub_HO_choices
            )
        print(f"{count} of {len(found_images_match_paired[ho])}")


def main(
    train_save_path: str,
    val_iid_save_path: str,
    test_save_path: str,
    reuse_scene_graph_paths: List[str],
    train_ims_with_HO: int,
    train_ims_without_HO: int,
    val_iid_ims_with_HO: int,
    val_iid_ims_without_HO: int,
    test_ims: int,
    verbose=True,
):
    # Determine for train split, re-using images where possible
    reusable_imgs = load_reusable_images(reuse_scene_graph_paths)
    (
        global_ims,
        found_images_match,
        found_images_match_paired,
        found_images_no_match,
        global_ims_clevr,
    ) = generate_required_obj_attributes(
        ims_with_HO=train_ims_with_HO,
        ims_without_HO=train_ims_without_HO,
        existing=reusable_imgs,
    )
    final_results = {
        "image_data": {
            glob_idx: [
                {"global_idx": glob_idx, "local_idx": loc_idx, "objects": scene}
                for loc_idx, scene in enumerate(global_ims[glob_idx])
            ]
            for glob_idx in global_ims
        },
        "reuse_info": [
            {
                "global_idx": glob_idx,
                "local_idx": loc_idx,
                "dataset": global_ims_clevr[(glob_idx, loc_idx)][0],
                "filename": global_ims_clevr[(glob_idx, loc_idx)][1],
            }
            for (glob_idx, loc_idx) in global_ims_clevr
        ],
        "match_ho": {
            str(ho): [
                {"global_idx": glob_idx, "local_idx": loc_idx}
                for (glob_idx, loc_idx) in found_images_match[ho]
            ]
            for ho in HO_choices
        },
        "match_ho_pair": {
            str(ho): [
                {"global_idx": glob_idx, "local_idx": loc_idx}
                for (glob_idx, loc_idx) in found_images_match_paired[ho]
            ]
            for ho in HO_choices
        },
        "not_match_ho": {
            str(ho): [
                {"global_idx": glob_idx, "local_idx": loc_idx}
                for (glob_idx, loc_idx) in found_images_no_match[ho]
            ]
            for ho in HO_choices
        },
        "split": "train",
    }
    with open(train_save_path, "w") as outfile:
        json.dump(final_results, outfile, indent=2)

    print()
    print(f"Train Total images: {sum(len(global_ims[x]) for x in global_ims)}")
    print(f"Train Reused images: {len(global_ims_clevr)}")

    if verbose:
        print("Match:")
        for key in sorted(found_images_match.keys(), key=lambda x: str(x)):
            print(f"{key}: Matching {found_images_match[key]}")
            print(f"{' '*len(str(key))}  Altered  {found_images_match_paired[key]}")
        print("\nNo Match:")
        for key in sorted(found_images_no_match.keys(), key=lambda x: str(x)):
            print(f"{key}: {found_images_no_match[key]}")

        for key in global_ims:
            print(key)
            assert isinstance(global_ims[key], list)
            for i, scene in enumerate(global_ims[key]):
                print(f"    {i}: ", end="")
                print(scene)

            assert len({len(scene) for scene in global_ims[key]}) == 1

        print()
        print("Image reuse:")
        for key in global_ims_clevr:
            print(f"{key}: {global_ims_clevr[key]}")

    distribution_checking(
        found_images_match,
        found_images_no_match,
        found_images_match_paired,
        global_ims,
        name="Train",
    )
    # check_size_for_no_atom(found_images_no_match, found_images_match_paired, global_ims, name="Train")

    # Repeat for validation-IID split; don't attempt to reuse images
    (
        global_ims,
        found_images_match,
        found_images_match_paired,
        found_images_no_match,
        global_ims_clevr,
    ) = generate_required_obj_attributes(
        ims_with_HO=val_iid_ims_with_HO,
        ims_without_HO=val_iid_ims_without_HO,
        existing=[],
    )
    final_results = {
        "image_data": {
            glob_idx: [
                {"global_idx": glob_idx, "local_idx": loc_idx, "objects": scene}
                for loc_idx, scene in enumerate(global_ims[glob_idx])
            ]
            for glob_idx in global_ims
        },
        "reuse_info": [
            {
                "global_idx": glob_idx,
                "local_idx": loc_idx,
                "dataset": global_ims_clevr[(glob_idx, loc_idx)][0],
                "filename": global_ims_clevr[(glob_idx, loc_idx)][1],
            }
            for (glob_idx, loc_idx) in global_ims_clevr
        ],
        "match_ho": {
            str(ho): [
                {"global_idx": glob_idx, "local_idx": loc_idx}
                for (glob_idx, loc_idx) in found_images_match[ho]
            ]
            for ho in HO_choices
        },
        "match_ho_pair": {
            str(ho): [
                {"global_idx": glob_idx, "local_idx": loc_idx}
                for (glob_idx, loc_idx) in found_images_match_paired[ho]
            ]
            for ho in HO_choices
        },
        "not_match_ho": {
            str(ho): [
                {"global_idx": glob_idx, "local_idx": loc_idx}
                for (glob_idx, loc_idx) in found_images_no_match[ho]
            ]
            for ho in HO_choices
        },
        "split": "val_iid",
    }
    with open(val_iid_save_path, "w") as outfile:
        json.dump(final_results, outfile, indent=2)

    print()
    print(f"Val-IID Total images: {sum(len(global_ims[x]) for x in global_ims)}")
    print(f"Val-IID Reused images: {len(global_ims_clevr)}")

    if verbose:
        print("Match:")
        for key in sorted(found_images_match.keys(), key=lambda x: str(x)):
            print(f"{key}: Matching {found_images_match[key]}")
            print(f"{' '*len(str(key))}  Altered  {found_images_match_paired[key]}")
        print("\nNo Match:")
        for key in sorted(found_images_no_match.keys(), key=lambda x: str(x)):
            print(f"{key}: {found_images_no_match[key]}")

        for key in global_ims:
            print(key)
            assert isinstance(global_ims[key], list)
            for i, scene in enumerate(global_ims[key]):
                print(f"    {i}: ", end="")
                print(scene)

            assert len({len(scene) for scene in global_ims[key]}) == 1

        print()
        print("Image reuse:")
        for key in global_ims_clevr:
            print(f"{key}: {global_ims_clevr[key]}")

    distribution_checking(
        found_images_match,
        found_images_no_match,
        found_images_match_paired,
        global_ims,
        name="Val-IID",
    )
    # check_size_for_no_atom(found_images_no_match, found_images_match_paired, global_ims, name="Val-IID")

    test_data = generate_test_data(test_ims)
    print(f"Test Total Images: {len(test_data['image_data'])}")
    with open(test_save_path, "w") as outfile:
        json.dump(test_data, outfile, indent=2)

    print()
    print("Test Scene Sizes by HO (should be approx balanced):")
    for ho in HO_choices:
        print(ho)
        print(
            Counter(
                len(test_data["image_data"][global_idx["global_idx"]][0]["objects"])
                for global_idx in test_data["match_ho"][str(ho)]
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reusable_ims_config",
        default="reuse_config.json",
        type=str,
        help="Path to a .json config file specifying the scenegraphs file & corresponding image directories for image we can reuse.",
    )
    parser.add_argument(
        "--ho_tuples_config",
        default="ho_tuples_config.json",
        type=str,
        help="Path to a .json config file specifying what held-out combinations to generate for",
    )
    parser.add_argument(
        "--train_output_path",
        default="output/held_out_objects_train.json",
        help="Path at which to save the train set generated object tuples as a .json (pre-scenegraph specifications; i.e. # of objects, object attributes, and nothing else)",
    )
    parser.add_argument(
        "--val_iid_output_path",
        default="output/held_out_objects_val_iid.json",
        help="Path at which to save the val-iid set generated object tuples as a .json (pre-scenegraph specifications; i.e. # of objects, object attributes, and nothing else)",
    )
    parser.add_argument(
        "--test_output_path",
        default="output/held_out_objects_test.json",
        help="Path at which to save the test set generated object tuples as a .json (pre-scenegraph specifications; i.e. # of objects, object attributes, and nothing else)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print debug messages; only good for small samples",
    )
    parser.add_argument("--disable_tqdm", action="store_true", help="Disable tqdm")
    # Type1 images
    parser.add_argument(
        "--train_ims_with_HO",
        default=0,
        type=int,
        help="Number of images in the train set with the held-out combination (or using their paired image that has the held-out combination removed)",
    )
    parser.add_argument(
        "--val_iid_ims_with_HO",
        default=0,
        type=int,
        help="Number of images in the val-IID set with the held-out combination (or using their paired image that has the held-out combination removed)",
    )
    parser.add_argument(
        "--test_ims",
        default=15000,
        type=int,
        help="Number of images in the test set; all will contain the held-out combination.",
    )
    # Type 2_3 images
    parser.add_argument(
        "--train_ims_without_HO",
        default=63000,
        type=int,
        help="Number of images in the train set without held-out combination",
    )
    parser.add_argument(
        "--val_iid_ims_without_HO",
        default=13500,
        type=int,
        help="Number of images in the val-IID set without held-out combination",
    )
    args = parser.parse_args()

    config = yaml.load(open(args.reusable_ims_config, "r"), Loader=yaml.Loader)

    HO_choices = load_ho_config(args.ho_tuples_config)

    DISABLE_TQDM = args.disable_tqdm

    main(
        train_save_path=args.train_output_path,
        val_iid_save_path=args.val_iid_output_path,
        test_save_path=args.test_output_path,
        reuse_scene_graph_paths=config["reuse_scene_graph_paths"],
        train_ims_with_HO=args.train_ims_with_HO,
        train_ims_without_HO=args.train_ims_without_HO,
        val_iid_ims_with_HO=args.val_iid_ims_with_HO,
        val_iid_ims_without_HO=args.val_iid_ims_without_HO,
        test_ims=args.test_ims,
        verbose=args.verbose,
    )
