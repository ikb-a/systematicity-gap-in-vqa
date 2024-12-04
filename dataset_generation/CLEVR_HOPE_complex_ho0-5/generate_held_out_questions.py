"""
Generate the question for the held-out CLEVR dataset.
"""
import copy
import random

from held_out_utils_generate_questions import (
    generate_questions_obeying_filter,
    instantiate_templates_dfs,
    default_add_empty_filter_options,
    rejection_heuristic,
)
from typing import Optional
from typing import Tuple, Callable, Any, List, Dict
import os
import json
import argparse

from held_out_utils_generate_questions import (
    node_shallow_copy,
    find_filter_options,
    find_relate_filter_options,
    replace_optionals,
    other_heuristic,
)
import question_engine as qeng
from tqdm import tqdm
import re
from collections import defaultdict

CLEVR_COLORS = ["blue", "brown", "cyan", "gray", "green", "purple", "red", "yellow"]
CLEVR_MATERIALS = ["rubber", "metal"]
CLEVR_SHAPES = ["cube", "cylinder", "sphere"]
CLEVR_SIZES = ["large", "small"]


def attr_match(attributes: Tuple, template: Tuple) -> bool:
    """
    Return true iff template[i] is None or template[i] == attributes[i] for all i

    :param attributes: Tuple of strings
    :param template: Tuple of strings or None
    :return:
    """
    for temp_attr, attr in zip(template, attributes):
        # if both are None then we may yet still match
        # if temp_attr is None and attr is not None then we yet may still match
        #    e.g., template = (None, blue, None, None) and attr = (small, ?, ?, ?)
        #    then we still have the possibility of a match if attr = (small, blue, None, None)

        # if temp_attr is not None and attr *is* None, then we cannot match
        #    e.g., template = (None, blue, None, None); this cannot match (small, None, cube, rubber)
        if temp_attr is not None and attr is None:
            return False
        # if neither is not None, and they disagree, then we cannot match
        if temp_attr is not None and attr is not None and temp_attr != attr:
            return False
    return True


def exclude_ho(
    ho: Tuple[Optional[str], Optional[str], Optional[str], Optional[str]],
    in_options: list,
) -> list:
    """

    :param ho: Held-out tuple to exclude; (size, color, shape, material)
    :param in_options: Instantiations under consideration; NOTE! These tuples are different order: (size, color, material, shape)
    :return:
    """
    # Convert from generators to tuples for simplicity
    in_options = [tuple(x) for x in in_options]

    if len(in_options) == 0:
        return in_options

    # Check all our options are of the same length
    lens = set(len(x) for x in in_options)
    assert len(lens) == 1

    # Define method that gets a tuple out of an entry in in_options
    # note that in_options is either a list of 4-tuples of attributes, or it
    # is a list of (direction, (4-tuple)) tuples
    # Also re-orders the attributes to the standard ordering
    if len(in_options[0]) == 4:
        get_attr_tuple = lambda x: (x[0], x[1], x[3], x[2])
    else:
        get_attr_tuple = lambda x: (x[1][0], x[1][1], x[1][3], x[1][2])

    # Check the method works, and that attributes are in the expected order
    test_tup = get_attr_tuple(in_options[0])
    assert len(test_tup) == 4
    assert test_tup[0] in CLEVR_SIZES + [None]
    assert test_tup[1] in CLEVR_COLORS + [None]
    assert test_tup[2] in CLEVR_SHAPES + [None]
    assert test_tup[3] in CLEVR_MATERIALS + [None]

    # Return the non-matching entries
    return [x for x in in_options if not attr_match(get_attr_tuple(x), template=ho)]


def program_to_filters(program_nodes):
    # Return mapping of topmost index to filter values
    no_attrs = [None, None, None, None]
    found_attrs = {}  # map topmost index to value
    attr_name_to_idx = {"size": 0, "color": 1, "shape": 2, "material": 3}

    for i, node in enumerate(program_nodes):
        if node["type"].startswith("filter_"):
            assert len(node["inputs"]) == 1
            assert len(node["side_inputs"]) == 1
            input_idx = node["inputs"][0]
            input_node = program_nodes[input_idx]
            attr_idx = attr_name_to_idx[node["type"][7:]]
            if input_node["type"].startswith("filter_"):
                assert input_idx in found_attrs
                curr_attrs = found_attrs.pop(input_idx)
                assert curr_attrs[attr_idx] is None
            else:
                curr_attrs = copy.deepcopy(no_attrs)
            curr_attrs[attr_idx] = node["side_inputs"][0]
            found_attrs[i] = curr_attrs

        else:
            assert not node["type"].startswith("filter")

    return [tuple(x) for x in found_attrs.values()]


# NOTE: "small blue cube" is a "blue cube"; for our held-out combination "blue cube"
#       (That's true w.r.t. stuff we don't include, and w.r.t. stuff we test)
def question_matches_ho(
    ho: Tuple[Optional[str], Optional[str], Optional[str], Optional[str]],
    question: dict,
):
    """
    Return true iff there is a series of filter operations in {question} that
    match the template held-out combination, {ho}
    """
    found_attrs = program_to_filters(program_nodes=question["nodes"])
    return any(attr_match(x, template=ho) for x in found_attrs)


def determine_completed_opportunities(state):
    completed_opportunities = 0
    names = ["<Z%s>", "<C%s>", "<M%s>", "<S%s>"]
    for i in range(1, 7):
        label = "" if i == 1 else str(i)
        # Need all of <Z>, <C>, <M>, and <S> to be present, b/c CLEVR has some hacky
        # templates that only specify a single attribute instead of up to all 4
        if all((name % label) in state["vals"] for name in names):
            completed_opportunities += 1
    # Make certain that there isn't any <Z7>'s or the like (i.e., that the upper
    # bound in the loop above is good enough)
    assert not any((name % "7" in state["vals"]) for name in names)
    # Make certain no question has more than 4 slots to fill
    assert completed_opportunities <= 4, state
    return completed_opportunities


def position_matches_ho(
    ho: Tuple[Optional[str], Optional[str], Optional[str], Optional[str]],
    position: int,
    in_options: list,
    template: dict,
    state: dict,
) -> list:
    """
    PRECONDITION: ho contains at least 2 values (we will ignore naked filter_size atoms)

    :param ho: Held-out tuple to include; (size, color, shape, material)
    :param position: Make the {position}th entry match. Zero-indexed
    :param in_options: Instantiations under consideration; NOTE! These tuples are different order: (size, color, material, shape)
    :param template: template being filled
    :param state: current partially filled program
    :return:
    """
    # Convert from generators to tuples for simplicity
    in_options = [tuple(x) for x in in_options]

    if len(in_options) == 0:
        return in_options

    # Check all our options are of the same length
    lens = set(len(x) for x in in_options)
    assert len(lens) == 1

    # Check that there actually *is* a chance to insert the held-out combination
    opportunities = len(
        [
            node
            for node in template["nodes"]
            if "side_inputs" in node and len(node["side_inputs"]) >= 4
        ]
    )
    assert opportunities >= position + 1, template

    # Figure out what's already been filled
    completed_opportunities = determine_completed_opportunities(state)
    # Make sure we haven't somehow filled more slots than exist
    assert completed_opportunities <= opportunities

    # print(str(completed_opportunities)+',', end='')

    # Unless we're currently filling in {position}, then we're done!
    next_node = template["nodes"][state["next_template_node"]]
    if completed_opportunities != position:
        return in_options
    elif len(next_node["side_inputs"]) < 4:
        # If the node we're currently filling isn't able to fully specify the
        # 4 attributes, then check that we're dealing with relate or filter_x raw
        assert next_node["type"] in [
            "relate",
            "filter_shape",
            "filter_size",
            "filter_material",
            "filter_size",
        ], next_node["type"]
        if len(next_node["side_inputs"]) < 4:
            if len(in_options[0]) == 4:
                assert set(
                    sum(1 if y is not None else 0 for y in x) for x in in_options
                ) == {3}
            else:
                assert len(in_options[0]) == 2
                assert set(x[1] for x in in_options) == {(None, None, None, None)}
        return in_options

    # Otherwise, we must filter the posibilities to include only those matching ho

    # Define method that gets a tuple out of an entry in in_options
    # note that in_options is either a list of 4-tuples of attributes, or it
    # is a list of (direction, (4-tuple)) tuples
    # Also re-orders the attributes to the standard ordering
    if len(in_options[0]) == 4:
        get_attr_tuple = lambda x: (x[0], x[1], x[3], x[2])
    else:
        get_attr_tuple = lambda x: (x[1][0], x[1][1], x[1][3], x[1][2])

    # Check the method works, and that attributes are in the expected order
    test_tup = get_attr_tuple(in_options[-1])
    assert len(test_tup) == 4
    assert test_tup[0] in CLEVR_SIZES + [None]
    assert test_tup[1] in CLEVR_COLORS + [None]
    assert test_tup[2] in CLEVR_SHAPES + [None]
    assert test_tup[3] in CLEVR_MATERIALS + [None]

    # Return the matching entries
    return [x for x in in_options if attr_match(get_attr_tuple(x), template=ho)]


class MatchHoAddEmptyOptions:
    """
    Ugly horrid hackiness.
    During question generation, the program normally chooses a set of
    attributes that yield empty sets of objects.
    In this case, we restrict ourselves to choosing from the subset that match HO.

    DONE: Check that the resulting bias for 0/no answers is controlled by
          the answer balancing system.
    NOTE: Test split answers will likely be imbalanced (odds of counting up to 10
          when forcing the object being checked?)
    NOTE: That's only 10% of the train... but it's 100% of the test split
    Note: Yep, results are imbalanced but not as bad as expected. The automatic
          answer balancing prevents things from getting too bad.
    """

    def __init__(
        self,
        test_ho: Tuple[
            Optional[str],
            Optional[str],
            Optional[str],
            Optional[str],
        ],
        insertion_index: int,
    ):
        self.test_ho = test_ho
        self.insertion_index = insertion_index
        self.possible_attribute_choices = [
            (size, col, mat, sha)
            for size in [None] + CLEVR_SIZES
            for col in [None] + CLEVR_COLORS
            for mat in [None] + CLEVR_MATERIALS
            for sha in [None] + CLEVR_SHAPES
        ]

    def __call__(self, attribute_map, metadata, num_to_add, state, next_node):
        # Figure out how many opportunities have already been filled
        completed_opps = determine_completed_opportunities(state)

        # If the next_node allows us to fully specify all 4 attribute, AND
        # *this* is the one we want to make HO, then restrict our negative samples
        # to those matching HO
        if (
            completed_opps == self.insertion_index
            and "side_inputs" in next_node
            and len(next_node["side_inputs"]) >= 4
        ):
            # Filter the 96 possible objects to find what matches HO and isn't
            # in the set of combinations that correspond to some objects
            possible_choices = [
                x
                for x in self.possible_attribute_choices
                if attr_match((x[0], x[1], x[3], x[2]), self.test_ho)
                and x not in attribute_map
            ]
            random.shuffle(possible_choices)
            possible_attribute_choices = possible_choices[:num_to_add]
            for x in possible_attribute_choices:
                attribute_map[x] = []

        # Otherwise, there's no constraints on this value and we can act normally
        else:
            default_add_empty_filter_options(attribute_map, metadata, num_to_add)


class MatchHoInstantiator:
    def __init__(
        self,
        test_ho: Tuple[
            Optional[str],
            Optional[str],
            Optional[str],
            Optional[str],
        ],
        reordered_template_dir: str = "./CLEVR_1.0_templates/reordered_templates/",
        inner_template_instantiator=instantiate_templates_dfs,
    ):
        self.instantiate_template = inner_template_instantiator
        self.test_ho = test_ho
        self.add_empty_matching_ho_at_pos = {}
        for i in range(4):
            self.add_empty_matching_ho_at_pos[i] = MatchHoAddEmptyOptions(
                test_ho=test_ho, insertion_index=i
            )

        self.reordered_templates = {}
        for fn in os.listdir(reordered_template_dir):
            if not fn.endswith(".json"):
                continue
            with open(os.path.join(reordered_template_dir, fn), "r") as f:
                for i, template in enumerate(json.load(f)):
                    key = (fn, i)
                    self.reordered_templates[key] = template

    # PRECONDITION: filter_option_filter is the identity function
    # NOTE: add_empty_filter_options is ignored; replaced by what is needed here
    def __call__(
        self,
        scene_struct,
        template,
        metadata,
        answer_counts,
        synonyms,
        template_name,
        max_instances=None,
        verbose=False,
        filter_option_filter: Callable[[list, dict, dict], list] = None,
        other_rejection_condition: Callable[
            [dict, dict, dict, Any, dict], bool
        ] = lambda q, m, s, a, t: False,
        add_empty_filter_options=None,  # Argument is ignored
    ) -> Tuple[List[str], List[List[Dict]], list]:
        if max_instances is None or max_instances > 1:
            raise NotImplementedError(
                "Need to add de-duplication accross instantiate_templates_dfs calls"
            )

        if filter_option_filter is not None:
            raise NotImplementedError("filter_option_filter must be None")

        scene_attributes = set(
            (x["size"], x["color"], x["shape"], x["material"])
            for x in scene_struct["objects"]
        )

        reordered_template = self.reordered_templates[template_name]

        # Figure out the number of position in which we could potentially insert ho
        opportunities = len(
            [
                node
                for node in template["nodes"]
                if "side_inputs" in node and len(node["side_inputs"]) >= 4
            ]
        )

        # randomize the order in which we attempt these
        positions = list(range(opportunities))
        random.shuffle(positions)

        text_questions = []
        structured_questions = []
        answers = []

        # Try the position in which we'll be inserting the held-out combination
        for opportunity_idx in positions:
            # Exit if we've already found enough questions
            if max_instances is not None and len(text_questions) == max_instances:
                break

            if verbose:
                print(
                    "Attempting to insert HO at position %i of %i"
                    % (opportunity_idx + 1, opportunities)
                )

            # Find the node we're trying to fill
            target_node = [
                node
                for node in template["nodes"]
                if "side_inputs" in node and len(node["side_inputs"]) >= 4
            ][opportunity_idx]

            # Skip if it's impossible to fill this node
            # Specifically, skip if trying to force a relate_filter node to an
            # attribute combination that doesn't exist in the image.
            # This will not work as relate filter insists on only doing well formed
            # things; i.e., object must exist (albiet, may be in a
            # different direction)
            if target_node["type"] in [
                "relate_filter",
                "relate_filter_unique",
                "relate_filter_count",
                "relate_filter_exist",
            ] and not any(attr_match(x, self.test_ho) for x in scene_attributes):
                if verbose:
                    print("Skipping; no obj in image, and it's a relate_filter node")
                continue

            # Run preemptive answer filtering when possible. This only catches
            # a subset of cases, but hopefully speeds things up.
            # Check if answer filtering stops us in counting problems
            if (
                target_node["type"] in ["relate_filter_count", "filter_count"]
                and target_node == template["nodes"][-1]
            ):  # *and* we're the root node
                # Answer cannot exceed the total number of objects in the scene, which
                # match the least constrained version of HO (i.e., just HO without
                # additional constraints)
                max_possible_answer = len(
                    [x for x in scene_attributes if attr_match(x, self.test_ho)]
                )
                if all(
                    rejection_heuristic(answer_counts, ans)
                    for ans in range(max_possible_answer + 1)
                ):
                    if verbose:
                        print(
                            "Skipping; pre-emptive answer filtering; answer is at most: %i"
                            % max_possible_answer
                        )
                    continue
            # Check if answer filtering stops us in existence problems
            elif (
                target_node["type"] in ["filter_exist"]
                and target_node == template["nodes"][-1]
                and not any(attr_match(x, self.test_ho) for x in scene_attributes)
                and rejection_heuristic(answer_counts, False)
            ):
                if verbose:
                    print("Skipping; pre-emptive answer filtering; answer is false")
                continue

            # If there are constraints, check if that makes it impossible to insert
            # the held-out comination at {opportunity_idx}
            if "constraints" in template:
                # Figure out which attributes (e.g., <Z4>) must be NULL
                null_attrs = []
                for c in template["constraints"]:
                    if c["type"] == "NULL":
                        null_attrs += c["params"]

                for attr in target_node["side_inputs"]:
                    assert attr in [
                        "<%s%s>" % (label, num)
                        for label in ["Z", "C", "S", "M", "R"]
                        for num in ["", "2", "3", "4"]
                    ], attr

                    if "<R" in attr:  # Ignore spatial relationship
                        continue
                    if attr in null_attrs:
                        # Need to figure out if this being null prevents us from instantiating HO
                        if (
                            (
                                attr in ["<Z>", "<Z2>", "<Z3>", "<Z4>"]
                                and self.test_ho[0] is not None
                            )
                            or (
                                attr in ["<C>", "<C2>", "<C3>", "<C4>"]
                                and self.test_ho[1] is not None
                            )
                            or (
                                attr in ["<S>", "<S2>", "<S3>", "<S4>"]
                                and self.test_ho[2] is not None
                            )
                            or (
                                attr in ["<M>", "<M2>", "<M3>", "<M4>"]
                                and self.test_ho[3] is not None
                            )
                        ):
                            if verbose:
                                print("Skip b/c template constraints do not allow")
                            return [], [], []

            # If the reordered template has the opportunity closer to the start
            # of the program (and therefore the constraint will be forced sooner
            # in the DFS performed by instantiate_templates_dfs), then use the
            # reordered template instead
            template_in_use = template
            if opportunity_idx > reordered_template["opp_mapping"][opportunity_idx]:
                if verbose:
                    print("Using reordered template")
                template_in_use = reordered_template
                opportunity_idx = reordered_template["opp_mapping"][opportunity_idx]

            filter = lambda x, temp, state: position_matches_ho(
                ho=self.test_ho,
                position=opportunity_idx,
                in_options=x,
                template=temp,
                state=state,
            )

            tq, sq, a = self.instantiate_template(
                scene_struct=scene_struct,
                template=template_in_use,
                metadata=metadata,
                answer_counts=answer_counts,
                synonyms=synonyms,
                max_instances=None
                if max_instances is None
                else max_instances
                - len(text_questions),  # however many are left to be made
                verbose=verbose,
                filter_option_filter=filter,
                other_rejection_condition=other_rejection_condition,
                add_empty_filter_options=self.add_empty_matching_ho_at_pos[
                    opportunity_idx
                ],
            )

            text_questions.extend(tq)
            structured_questions.extend(sq)
            answers.extend(a)

        return text_questions, structured_questions, answers


def load_sg_subset(
    full_sg_path, objects_file_path, glob_loc_to_img_path, target_ho, family
):
    """

    :param full_sg_path: Path to all of the scenegraphs in the dataset (i.e.,
                         the results when the folder of scenegraphs produced by
                         render_images_from_sg.py is merged using
                         collect_scenes.py).
    :param objects_file_path: Path to the objects within each scenegraph (i.e., the
                         output of generate_held_out_objects.py;
                         e.g., held_out_objects_train.json)
    :param glob_loc_to_img_path: Path to the mapping from (global, local) indices
                                 to the final image indices; this mapping is
                                 created by generate_held_out_sg_from_objects.py
                                 (e.g., glob_loc_index_to_img_index_train.json)
    :param family: One of "match_ho", "match_ho_pair", or "not_match_ho"; corresponds
                   to the type of scenegraphs you're going to load.
                   "match_ho" are images that match the held-out combination, {target_ho},
                   "match_ho_pair" are paired with "match_ho", and are altered to no longer match {target_ho},
                   "not_match_ho" are scenes that naturally do not match {target_ho}.

    :return:
    """
    # Load all .json files
    with open(full_sg_path, "r") as infile:
        all_sgs_struct = json.load(infile)
        all_sgs = all_sgs_struct["scenes"]
    with open(objects_file_path, "r") as infile:
        objects = json.load(infile)
    with open(glob_loc_to_img_path, "r") as infile:
        glob_loc_to_img = json.load(infile)

    # Convert mapping to an actual dict for ease of use
    glob_loc_to_img_map = {}
    for i, glob_loc in enumerate(glob_loc_to_img):
        glob_loc_to_img_map[tuple(glob_loc)] = i
    del glob_loc_to_img

    # Convert the indices we need
    desired_glob_loc_indices = objects[family][str(target_ho)]
    desired_glob_loc_indices = [
        (x["global_idx"], x["local_idx"]) for x in desired_glob_loc_indices
    ]
    desired_img_indices = [glob_loc_to_img_map[x] for x in desired_glob_loc_indices]

    # Retrieve and return, confirming that the indices actually match
    results_list = []
    for img_idx in desired_img_indices:
        scene = all_sgs[img_idx]
        assert img_idx == scene["image_index"]
        results_list.append(scene)

    results = {"scenes": results_list, "info": all_sgs_struct["info"]}

    return results


class NoMatchHoAddEmptyOptions:
    """
    Ugly horrid hackiness.
    During question generation, the program normally chooses a set of
    attributes that yield empty sets of objects.
    In this case, we restrict ourselves to choosing from the subset that
    DO NOT match HO.
    """

    def __init__(
        self,
        train_ho: Tuple[Optional[str], Optional[str], Optional[str], Optional[str]],
    ):
        self.train_ho = train_ho
        self.possible_attribute_choices = [
            tuple((size, col, mat, sha))
            for size in [None] + CLEVR_SIZES
            for col in [None] + CLEVR_COLORS
            for mat in [None] + CLEVR_MATERIALS
            for sha in [None] + CLEVR_SHAPES
        ]

    def __call__(self, attribute_map, metadata, num_to_add, state, next_node):
        # Restrict our negative samples to those NOT matching HO

        # Filter the 96 possible objects to find what does NOT match HO and isn't
        # in the set of combinations that correspond to some object in the scene
        possible_choices = [
            x
            for x in self.possible_attribute_choices
            if not attr_match((x[0], x[1], x[3], x[2]), self.train_ho)
            and x not in attribute_map
        ]
        random.shuffle(possible_choices)
        possible_attribute_choices = possible_choices[:num_to_add]
        for x in possible_attribute_choices:
            attribute_map[x] = []


class PairedImgFinder:
    """
    Given the index or scene for an image (either containing HO, or modified
    to no longer contain HO), then we return the paired image (i.e., either modified
    to no longer contain HO, or the original that contained HO).
    """

    def __init__(
        self,
        ho: Tuple[Optional[str], Optional[str], Optional[str], Optional[str]],
        gl_to_img_idx_path: str = "outputs/scenes_to_render/glob_loc_index_to_img_index_train.json",
        objects_path: str = "held_out_objects_train.json",
        sg_path: str = "outputs/scenes/CLEVR_held_out_train.json",
        split: str = "train",
    ):
        # Get mapping from (global, local) indices to image index & vice versa
        with open(gl_to_img_idx_path, "r") as infile:
            glob_loc_to_img_idx = json.load(infile)
        self.glob_loc_to_img_idx = {}
        self.img_idx_to_glob_loc = {}
        for img_idx, glob_loc in enumerate(glob_loc_to_img_idx):
            self.glob_loc_to_img_idx[tuple(glob_loc)] = img_idx
            self.img_idx_to_glob_loc[img_idx] = tuple(glob_loc)

        # Get the categories of all images
        with open(objects_path, "r") as infile:
            tmp = json.load(infile)
            assert tmp["split"] == split
            self.not_match_ho = [
                (x["global_idx"], x["local_idx"]) for x in tmp["not_match_ho"][str(ho)]
            ]

        self.match_to_pair = {}
        self.pair_to_match = {}
        for match, pair in zip(tmp["match_ho"][str(ho)], tmp["match_ho_pair"][str(ho)]):
            assert match["global_idx"] == pair["global_idx"]

            match = (match["global_idx"], match["local_idx"])
            pair = (pair["global_idx"], pair["local_idx"])

            self.match_to_pair[match] = pair
            self.pair_to_match[pair] = match

        # Don't load scenegraphs until needed
        self.scenegraphs = None
        self.sg_path = sg_path

        self.split = split

    def __call__(self, image_index: int) -> int:
        """
        Convert image_index into the image index of the paired image.
        Raises a value error if this index doesn't have a paired image.
        """
        # Convert image index to (global idx, local idx)
        # Recall that global index
        glob_loc = self.img_idx_to_glob_loc[image_index]

        if glob_loc in self.match_to_pair:
            return self.glob_loc_to_img_idx[self.match_to_pair[glob_loc]]
        elif glob_loc in self.pair_to_match:
            return self.glob_loc_to_img_idx[self.pair_to_match[glob_loc]]
        else:
            assert glob_loc in self.not_match_ho
            raise ValueError("Provided image_index does not have a paired image.")

    def scene_to_paired_scene(self, scene: dict) -> dict:
        assert scene["split"] == self.split
        image_index = scene["image_index"]
        return self.idx_to_paired_scene(image_index)

    def idx_to_paired_scene(self, image_index) -> dict:
        # Load the scenegraphs of all images if they aren't already loaded
        if self.scenegraphs is None:
            with open(self.sg_path, "r") as infile:
                self.scenegraphs = json.load(infile)
                assert self.scenegraphs["info"]["split"] == self.split
                self.scenegraphs = self.scenegraphs["scenes"]

        paired_idx = self(image_index)
        paired_scene = self.scenegraphs[paired_idx]
        assert paired_scene["image_index"] == paired_idx
        assert paired_scene["split"] == self.split
        return paired_scene

    def idx_to_paired_filename(self, image_index) -> str:
        paired_scene = self.idx_to_paired_scene(image_index)
        return paired_scene["image_filename"]


class ImageInvariantRejector:
    """
    A rejection condition for the question search that rejects a proposed
    question if changing the scene to its pair changes the answer

    NOTE: PairedImgInstantiator means this rejector shouldn't actually do anything
    """

    def __init__(self, pair_finder: PairedImgFinder, verbose=False):
        self.scene_pairer = pair_finder
        self.verbose = verbose

    def __call__(self, q, metadata, scene_struct, answer, template) -> bool:
        q = copy.deepcopy(q)  # Prevent mutation

        # Purge cached execution data
        for node in q["nodes"]:
            node.pop("_output", None)

        paired_scene = self.scene_pairer.scene_to_paired_scene(scene_struct)
        # cache_outputs=True should mutate the question with new cached values
        outputs = qeng.answer_question(
            q, metadata, paired_scene, cache_outputs=True, all_outputs=True
        )
        paired_answer = outputs[-1]

        if answer != paired_answer:
            if self.verbose:
                # print("REJECT B/C DOES NOT MATCH PAIR")
                print(scene_struct["image_index"], end="")
                print("-", end="")
                print(self.scene_pairer(scene_struct["image_index"]), end="")
                print(paired_answer)
                print("r", end="")
            return True

        # Ensure that the question is not degenerate on the new image
        # This call should reuse the cached values
        has_relate = any(n["type"] == "relate" for n in template["nodes"])
        if has_relate:
            if qeng.is_degenerate(q, metadata, paired_scene, answer=paired_answer):
                if self.verbose:
                    print("REJECT B/C DEGENERATE UNDER PAIR")
                return True

        return False


class PairedImgInstantiator:
    """
    Question instantiator that runs on 2 images simulatneously -- ensure the
    question is valid for both images, and has the same answer on both images.
    Note that the logic that produces the answer is allowed to vary between
    the images -- all that matters is the final answer.
    """

    def __init__(
        self, pair_img_finder: PairedImgFinder, record_paired_programs: bool = True
    ):
        self.pair_img_finder = pair_img_finder
        self.record_paired_programs = record_paired_programs
        if record_paired_programs:
            self.paired_programs = {}

    @staticmethod
    def _intersect_fo(filter_options: dict, paired_filter_options: dict):
        """
        Mutate filter_options & paired_filter_options so that their keys equal
        the intersection of the dictionary's keys
        :param filter_options:
        :param paired_filter_options:
        :return: None; the filter options are mutated
        """
        fo_keys = set(filter_options.keys())
        p_fo_keys = set(paired_filter_options.keys())
        keys = fo_keys.intersection(p_fo_keys)
        for key in fo_keys.union(p_fo_keys):
            if key not in keys:
                filter_options.pop(key, None)
                paired_filter_options.pop(key, None)

    def _get_empty_filter_options(
        self, full_filter_options, full_paired_filter_options, metadata, num_to_add
    ):
        """
        NOTE: Important to get the full filter_options and full paired_filter_options
        (i.e., the true full list of available objects, BEFORE we take the intersection
        of our available choices)
        """
        # Add some filtering criterion that do NOT correspond to objects

        if metadata["dataset"] == "CLEVR-v1.0":
            attr_keys = ["Size", "Color", "Material", "Shape"]
        else:
            assert False, "Unrecognized dataset"

        attr_vals = [metadata["types"][t] + [None] for t in attr_keys]
        if "_filter_options" in metadata:
            attr_vals = metadata["_filter_options"]

        results = set()
        while len(results) < num_to_add:
            k = (random.choice(v) for v in attr_vals)
            if k not in full_filter_options and k not in full_paired_filter_options:
                results.add(tuple(k))

        return results

    def __call__(
        self,
        scene_struct,
        template,
        metadata,
        answer_counts,
        synonyms,
        max_instances=None,
        verbose=False,
        filter_option_filter: Callable[[list, dict, dict], list] = lambda x: x,
        other_rejection_condition: Callable[
            [dict, dict, dict, Any, dict], bool
        ] = lambda q, m, s, a, t: False,
        add_empty_filter_options=None,
        template_name: Tuple[str, int] = ("", -1),
    ) -> Tuple[List[str], List[List[Dict]], list]:
        """
        Note! add_empty_filter_options should be None; we need to override it
        in order to get empty options under both images

        A modification of instantiate_templates_dfs from
        held_out_utils_generate_questions.py; modified the depth-first search
        to operate on the scene, *and* paired scene, at once. Otherwise there's
        too many problems with programs on the scene being invalid on the
        paired scene.

        Given the scenegraph for a file (scene_struct), a template to instantiate,
        synonyms, and variables related to rejection sampling (answer_counts),
        create valid instances of question+programs+answers.

        If filter_option_filter is provided, then the function can be used to
        restrict the types of filter options that this program can produce.
        Specifically, this function takes the full list of filter options under
        consideration, and returns a subset that the program is allowed to use.
        The function recieves either a list of 4-tuples, or a list
        of (direction, (4-tuple)) tuples. The 4 tuples are (size, color, material, shape)
        using the human-readable string names (see the keys of data/properties.json)

        Returns list of text questions, list of the corresponding programs
        (programs being represented as a list of nodes, each node represented by a dict),
        and a list of answers.
        """
        if add_empty_filter_options is not None:
            raise NotImplementedError(
                "Only works with add_empty_filter_options being None"
            )

        assert max_instances > 0

        paired_scene_struct = self.pair_img_finder.scene_to_paired_scene(scene_struct)

        param_name_to_type = {p["name"]: p["type"] for p in template["params"]}
        # (size, colour, mat, shape)
        param_name_to_idx = {"Size": 0, "Color": 1, "Material": 2, "Shape": 3}

        initial_state = {
            "nodes": [node_shallow_copy(template["nodes"][0])],
            "vals": {},
            "input_map": {0: 0},
            "next_template_node": 1,
        }
        states = [initial_state]
        paired_states = copy.deepcopy(states)
        final_states = []
        paired_final_states = []
        while states:
            state = states.pop()
            paired_state = paired_states.pop()

            # Check to make sure the current state is valid
            q = {"nodes": state["nodes"]}
            paired_q = {"nodes": paired_state["nodes"]}
            outputs = qeng.answer_question(q, metadata, scene_struct, all_outputs=True)

            paired_outputs = qeng.answer_question(
                paired_q, metadata, paired_scene_struct, all_outputs=True
            )
            answer = outputs[-1]
            paired_answer = paired_outputs[-1]
            if answer == "__INVALID__" or paired_answer == "__INVALID__":
                if verbose:
                    print("Discarding invalid state")
                continue

            # Check to make sure constraints are satisfied for the current state
            skip_state = False
            for constraint in template["constraints"]:
                if constraint["type"] == "NEQ":
                    p1, p2 = constraint["params"]

                    for v1, v2 in [
                        (state["vals"].get(p1), state["vals"].get(p2)),
                        (paired_state["vals"].get(p1), paired_state["vals"].get(p2)),
                    ]:
                        if v1 is not None and v2 is not None and v1 != v2:
                            if verbose:
                                print("skipping due to NEQ constraint")
                                print(constraint)
                                print(state["vals"])
                            skip_state = True
                            break
                        if skip_state:
                            break
                elif constraint["type"] == "NULL":
                    p = constraint["params"][0]
                    p_type = param_name_to_type[p]
                    for v in [state["vals"].get(p), paired_state["vals"].get(p)]:
                        if v is not None:
                            skip = False
                            if p_type == "Shape" and v != "thing":
                                skip = True
                            if p_type != "Shape" and v != "":
                                skip = True
                            if skip:
                                if verbose:
                                    print("skipping due to NULL constraint")
                                    print(constraint)
                                    print(state["vals"])
                                skip_state = True
                                break
                        if skip_state:
                            break
                elif constraint["type"] == "OUT_NEQ":
                    i_idx, j_idx = constraint["params"]

                    for i, j, inner_outputs in [
                        (
                            state["input_map"].get(i_idx, None),
                            state["input_map"].get(j_idx, None),
                            outputs,
                        ),
                        (
                            paired_state["input_map"].get(i_idx, None),
                            paired_state["input_map"].get(j_idx, None),
                            paired_outputs,
                        ),
                    ]:
                        if (
                            i is not None
                            and j is not None
                            and inner_outputs[i] == inner_outputs[j]
                        ):
                            if verbose:
                                print("skipping due to OUT_NEQ constraint")
                                print(scene_struct["split"])
                                print(scene_struct["image_index"])
                                print(inner_outputs[i])
                                print(inner_outputs[j])
                            skip_state = True
                            break
                        if skip_state:
                            break
                else:
                    assert False, (
                        'Unrecognized constraint type "%s"' % constraint["type"]
                    )

            if skip_state:
                continue

            # We have already checked to make sure the answer is valid, so if we have
            # processed all the nodes in the template then the current state is a valid
            # question, so add it if it passes our rejection sampling tests.
            if state["next_template_node"] == len(template["nodes"]):
                assert paired_state["next_template_node"] == len(template["nodes"])

                # Check we are invariant to our choice of image
                if answer != paired_answer:
                    if verbose:
                        print("Reject -- not invariant to image choice")
                    continue

                if rejection_heuristic(answer_counts, answer, verbose):
                    continue

                # If the template contains a raw relate node then we need to check for
                # degeneracy at the end
                has_relate = any(n["type"] == "relate" for n in template["nodes"])
                if has_relate:
                    degen = qeng.is_degenerate(
                        q, metadata, scene_struct, answer=answer, verbose=verbose
                    )

                    degen = degen or qeng.is_degenerate(
                        paired_q,
                        metadata,
                        paired_scene_struct,
                        answer=paired_answer,
                        verbose=verbose,
                    )
                    if degen:
                        continue

                # Any user-specified rejection /requirements:
                if other_rejection_condition(
                    q, metadata, scene_struct, answer, template
                ):
                    continue

                if other_rejection_condition(
                    paired_q, metadata, paired_scene_struct, paired_answer, template
                ):
                    continue

                answer_counts[answer] += 1
                state["answer"] = answer
                paired_state["answer"] = paired_answer
                final_states.append(state)
                paired_final_states.append(paired_state)
                if max_instances is not None and len(final_states) == max_instances:
                    break
                continue

            # Otherwise fetch the next node from the template
            # Make a shallow copy so cached _outputs don't leak ... this is very nasty
            next_node = template["nodes"][state["next_template_node"]]
            next_node = node_shallow_copy(next_node)

            special_nodes = {
                "filter_unique",
                "filter_count",
                "filter_exist",
                "filter",
                "relate_filter",
                "relate_filter_unique",
                "relate_filter_count",
                "relate_filter_exist",
            }

            if next_node["type"] in special_nodes:
                if next_node["type"].startswith("relate_filter"):
                    unique = next_node["type"] == "relate_filter_unique"
                    include_zero = (
                        next_node["type"] == "relate_filter_count"
                        or next_node["type"] == "relate_filter_exist"
                    )
                    filter_options = find_relate_filter_options(
                        answer,
                        scene_struct,
                        metadata,
                        unique=unique,
                        include_zero=include_zero,
                    )

                    paired_filter_options = find_relate_filter_options(
                        paired_answer,
                        paired_scene_struct,
                        metadata,
                        unique=unique,
                        include_zero=include_zero,
                    )
                else:
                    filter_options = find_filter_options(
                        answer, scene_struct, metadata
                    )  # Dict (size, colour, mat, shape) to list of indices of matching objects. None for not specified.

                    paired_filter_options = find_filter_options(
                        paired_answer, paired_scene_struct, metadata
                    )

                    # Remove the options in filter_options that the filter doesn't like
                    ori_filter_keys = list(filter_options.keys())
                    paired_ori_filter_keys = list(paired_filter_options.keys())
                    filtered_ori_keys = filter_option_filter(
                        ori_filter_keys, template, state
                    )
                    paired_filtered_ori_keys = filter_option_filter(
                        paired_ori_filter_keys, template, paired_state
                    )

                    # NOTE: Keep only the intersection (i.e., keys that are valid for *both* images)
                    for tmp_ori_filter_keys in [
                        ori_filter_keys,
                        paired_ori_filter_keys,
                    ]:
                        for key in tmp_ori_filter_keys:
                            if (
                                key not in filtered_ori_keys
                                or key not in paired_filtered_ori_keys
                            ):
                                filter_options.pop(key, None)
                                paired_filter_options.pop(key, None)

                    if next_node["type"] == "filter":
                        # Remove null filter
                        filter_options.pop((None, None, None, None), None)
                        paired_filter_options.pop((None, None, None, None), None)
                    if next_node["type"] == "filter_unique":
                        # Get rid of all filter options that don't result in a single object
                        filter_options = {
                            k: v for k, v in filter_options.items() if len(v) == 1
                        }
                        paired_filter_options = {
                            k: v
                            for k, v in paired_filter_options.items()
                            if len(v) == 1
                        }
                    else:
                        # Record the full set of objects in both images
                        full_filter_options = set(filter_options.keys())
                        full_paired_filter_options = set(paired_filter_options.keys())
                        # Reduce the filter options to the intersection of their keys
                        self._intersect_fo(filter_options, paired_filter_options)

                        # Add some filter options that do NOT correspond to the scene
                        if next_node["type"] == "filter_exist":
                            # For filter_exist we want an equal number that do and don't
                            num_to_add = len(filter_options)
                            assert num_to_add == len(paired_filter_options)
                        elif (
                            next_node["type"] == "filter_count"
                            or next_node["type"] == "filter"
                        ):
                            # For filter_count add nulls equal to the number of singletons
                            num_to_add = sum(
                                1 for k, v in filter_options.items() if len(v) == 1
                            )
                            # NOTE: here we just take the bias; can't do paired_filter_options here.
                        # NOTE: Both images must agree on the answer being empty:
                        empty_keys = self._get_empty_filter_options(
                            full_filter_options,
                            full_paired_filter_options,
                            metadata,
                            num_to_add,
                        )
                        for empty_key in empty_keys:
                            filter_options[empty_key] = []
                            paired_filter_options[empty_key] = []

                # Merge filter_options & paired_filter_options to retain only
                # the options that are valid for *both* scenes
                self._intersect_fo(filter_options, paired_filter_options)

                filter_option_keys = list(filter_options.keys())
                paired_filter_option_keys = list(paired_filter_options.keys())
                assert set(filter_option_keys) == set(paired_filter_option_keys)
                filter_option_keys = filter_option_filter(
                    filter_option_keys, template, state
                )
                paired_filter_option_keys = filter_option_filter(
                    paired_filter_option_keys, template, paired_state
                )

                # Take the intersection of the keys:
                filter_option_keys = [
                    x for x in filter_option_keys if x in paired_filter_option_keys
                ]
                # paired_filter_option_keys is now a copy of filter_option_keys
                del paired_filter_option_keys

                random.shuffle(filter_option_keys)
                # paired_filter_option_keys is now a copy of filter_option_keys

                for k in filter_option_keys:
                    new_nodes = []
                    cur_next_vals = {k: v for k, v in state["vals"].items()}
                    assert state["vals"] == paired_state["vals"]
                    next_input = state["input_map"][next_node["inputs"][0]]
                    assert (
                        next_input == paired_state["input_map"][next_node["inputs"][0]]
                    )
                    filter_side_inputs = next_node["side_inputs"]
                    if next_node["type"].startswith("relate"):
                        param_name = next_node["side_inputs"][
                            0
                        ]  # First one should be relate
                        filter_side_inputs = next_node["side_inputs"][1:]
                        param_type = param_name_to_type[param_name]
                        assert param_type == "Relation"
                        param_val = k[0]
                        k = k[1]
                        new_nodes.append(
                            {
                                "type": "relate",
                                "inputs": [next_input],
                                "side_inputs": [param_val],
                            }
                        )
                        cur_next_vals[param_name] = param_val
                        next_input = len(state["nodes"]) + len(new_nodes) - 1
                    for param_name, param_val in zip(filter_side_inputs, k):
                        param_type = param_name_to_type[param_name]
                        filter_type = "filter_%s" % param_type.lower()
                        if param_val is not None:
                            new_nodes.append(
                                {
                                    "type": filter_type,
                                    "inputs": [next_input],
                                    "side_inputs": [param_val],
                                }
                            )
                            cur_next_vals[param_name] = param_val
                            next_input = len(state["nodes"]) + len(new_nodes) - 1
                        elif param_val is None:
                            if (
                                metadata["dataset"] == "CLEVR-v1.0"
                                and param_type == "Shape"
                            ):
                                param_val = "thing"
                            else:
                                param_val = ""
                            cur_next_vals[param_name] = param_val
                    input_map = {k: v for k, v in state["input_map"].items()}
                    assert input_map == {
                        k: v for k, v in paired_state["input_map"].items()
                    }
                    extra_type = None
                    if next_node["type"].endswith("unique"):
                        extra_type = "unique"
                    if next_node["type"].endswith("count"):
                        extra_type = "count"
                    if next_node["type"].endswith("exist"):
                        extra_type = "exist"
                    if extra_type is not None:
                        new_nodes.append(
                            {
                                "type": extra_type,
                                "inputs": [
                                    input_map[next_node["inputs"][0]] + len(new_nodes)
                                ],
                            }
                        )
                    input_map[state["next_template_node"]] = (
                        len(state["nodes"]) + len(new_nodes) - 1
                    )
                    states.append(
                        {
                            "nodes": state["nodes"] + new_nodes,
                            "vals": cur_next_vals,
                            "input_map": input_map,
                            "next_template_node": state["next_template_node"] + 1,
                        }
                    )
                    paired_states.append(
                        {
                            "nodes": paired_state["nodes"]
                            + copy.deepcopy(new_nodes),  # copy to prevent mutation
                            "vals": copy.deepcopy(cur_next_vals),
                            "input_map": copy.deepcopy(input_map),
                            "next_template_node": paired_state["next_template_node"]
                            + 1,
                        }
                    )

            elif "side_inputs" in next_node:  # e.g., 'filter_size' atom
                # If the next node has template parameters, expand them out
                # CLEVR: Generalize this to work for nodes with more than one side input
                assert len(next_node["side_inputs"]) == 1, "NOT IMPLEMENTED"

                # Use metadata to figure out domain of valid values for this parameter.
                # Iterate over the values in a random order; then it is safe to bail
                # from the DFS as soon as we find the desired number of valid template
                # instantiations.
                param_name = next_node["side_inputs"][0]
                param_type = param_name_to_type[
                    param_name
                ]  # 'Size', 'Color', 'Material' or 'Shape'
                param_vals = metadata["types"][param_type][:]
                random.shuffle(param_vals)

                # Now perform any needed filtering.
                if param_type != "Relation":
                    # convert the parameter values into a list of (size, color, material, shape) tuple
                    assert param_type in param_name_to_idx
                    tmp_idx = param_name_to_idx[param_type]
                    param_vals = [
                        ((None,) * 4)[:tmp_idx] + (val,) + ((None,) * 4)[tmp_idx + 1 :]
                        for val in param_vals
                    ]
                    # filter & convert back into a list of values
                    param_vals = filter_option_filter(param_vals, template, state)
                    param_vals = filter_option_filter(
                        param_vals, template, paired_state
                    )
                    param_vals = [x[tmp_idx] for x in param_vals]
                else:
                    # convert the parameter values into a list of (relation, (size, color, mat, shape)) tuples
                    param_vals = [(val, (None,) * 4) for val in param_vals]
                    # filter, and undo converstion
                    param_vals = filter_option_filter(param_vals, template, state)
                    param_vals = filter_option_filter(
                        param_vals, template, paired_state
                    )
                    param_vals = [x[0] for x in param_vals]

                # Extra shuffle just in case
                random.shuffle(param_vals)

                for val in param_vals:
                    input_map = {k: v for k, v in state["input_map"].items()}
                    assert input_map == {
                        k: v for k, v in paired_state["input_map"].items()
                    }
                    input_map[state["next_template_node"]] = len(state["nodes"])
                    assert (
                        state["next_template_node"]
                        == paired_state["next_template_node"]
                    )
                    assert len(state["nodes"]) == len(paired_state["nodes"])
                    cur_next_node = {
                        "type": next_node["type"],
                        "inputs": [input_map[idx] for idx in next_node["inputs"]],
                        "side_inputs": [val],
                    }
                    cur_next_vals = {k: v for k, v in state["vals"].items()}
                    assert cur_next_vals == {
                        k: v for k, v in paired_state["vals"].items()
                    }
                    cur_next_vals[param_name] = val

                    states.append(
                        {
                            "nodes": state["nodes"] + [cur_next_node],
                            "vals": cur_next_vals,
                            "input_map": input_map,
                            "next_template_node": state["next_template_node"] + 1,
                        }
                    )
                    paired_states.append(
                        {
                            "nodes": paired_state["nodes"]
                            + copy.deepcopy([cur_next_node]),
                            "vals": copy.deepcopy(cur_next_vals),
                            "input_map": copy.deepcopy(input_map),
                            "next_template_node": paired_state["next_template_node"]
                            + 1,
                        }
                    )
            else:
                input_map = {k: v for k, v in state["input_map"].items()}
                assert input_map == {k: v for k, v in paired_state["input_map"].items()}
                input_map[state["next_template_node"]] = len(state["nodes"])
                assert state["next_template_node"] == paired_state["next_template_node"]
                assert len(state["nodes"]) == len(paired_state["nodes"])
                next_node = {
                    "type": next_node["type"],
                    "inputs": [input_map[idx] for idx in next_node["inputs"]],
                }
                states.append(
                    {
                        "nodes": state["nodes"] + [next_node],
                        "vals": state["vals"],
                        "input_map": input_map,
                        "next_template_node": state["next_template_node"] + 1,
                    }
                )
                paired_states.append(
                    {
                        "nodes": paired_state["nodes"] + copy.deepcopy([next_node]),
                        "vals": paired_state["vals"],
                        "input_map": copy.deepcopy(input_map),
                        "next_template_node": paired_state["next_template_node"] + 1,
                    }
                )

        # Actually instantiate the template with the solutions we've found
        text_questions, structured_questions, answers = [], [], []
        paired_answers = []
        assert len(final_states) == len(paired_final_states)
        for state, paired_state in zip(final_states, paired_final_states):
            structured_questions.append(state["nodes"])
            answers.append(state["answer"])
            paired_answers.append(paired_state["answer"])

            text = random.choice(template["text"])
            assert state["vals"] == paired_state["vals"]
            for name, val in state["vals"].items():
                if val in synonyms:
                    val = random.choice(synonyms[val])
                text = text.replace(name, val)
                text = " ".join(text.split())
            text = replace_optionals(text)
            text = " ".join(text.split())
            text = other_heuristic(text, state["vals"], verbose=verbose)
            text_questions.append(text)

            # record mapping from image index & question, to the corresponding
            # paired program and paired image index
            # Note: the paired program is *mostly* identical; they key difference
            #       is the value of _outputs
            if self.record_paired_programs:
                if (scene_struct["image_index"], text) in self.paired_programs:
                    print(scene_struct["image_index"])
                    print(text)
                    print(paired_state["nodes"])
                    print(paired_scene_struct["image_index"])
                    print(self.paired_programs[(scene_struct["image_index"], text)])

                assert (scene_struct["image_index"], text) not in self.paired_programs
                self.paired_programs[(scene_struct["image_index"], text)] = (
                    paired_state["nodes"],
                    paired_scene_struct["image_index"],
                )

        assert paired_answers == answers
        return text_questions, structured_questions, answers


def replace_paired_optionals(s1, s2):
    """
    Each substring of s that is surrounded in square brackets is treated as
    optional and is removed with probability 0.5. For example the string

    "A [aa] B [bb]"

    could become any of

    "A aa B bb"
    "A  B bb"
    "A aa B "
    "A  B "

    with probability 1/4.
    """
    pat = re.compile(r"\[([^\[]*)\]")

    while True:
        match1 = re.search(pat, s1)
        match2 = re.search(pat, s2)
        if not match1:
            assert not match2
            break
        i0_1 = match1.start()
        i1_1 = match1.end()
        i0_2 = match2.start()
        i1_2 = match2.end()
        # NOTE: Can't quite compare exact positions b/c can be shifted by different
        # args. But, offsets should be equal
        assert i1_1 - i0_1 == i1_2 - i0_2

        if random.random() > 0.5:
            s1 = s1[:i0_1] + match1.groups()[0] + s1[i1_1:]
            s2 = s2[:i0_2] + match2.groups()[0] + s2[i1_2:]
        else:
            s1 = s1[:i0_1] + s1[i1_1:]
            s2 = s2[:i0_2] + s2[i1_2:]
    return s1, s2


class PairedQuestionInstantiator:
    """
    Question instantiator that runs on 2 questions simultaneously, forcing one
    question to contain HO, and forcing another to contain HO with values
    replaced with alt values.
    Assumes the image does not contain HO.
    Also requires both questions to have the same answer.
    Note that the logic that produces the answer is allowed to vary between
    the images -- all that matters is the final answer.

    Paired questions are stored in self.pared_questions
    """

    paired_questions: Dict

    def __init__(
        self,
        train_ho: Tuple[Optional[str], Optional[str], Optional[str], Optional[str]],
        alt_ho: Tuple[Optional[str], Optional[str], Optional[str], Optional[str]],
        reordered_template_dir: str = "./CLEVR_1.0_templates/reordered_templates/",
    ):
        self.paired_questions = {}
        self.train_ho = train_ho
        self.alt_ho = alt_ho

        for x, y in zip(train_ho, alt_ho):
            assert (x is None and y is None) or (
                x is not None and y is not None and x != y
            )

        # Get re-ordered templates for faster search where possible
        self.reordered_templates = {}
        for fn in os.listdir(reordered_template_dir):
            if not fn.endswith(".json"):
                continue
            with open(os.path.join(reordered_template_dir, fn), "r") as f:
                for i, template in enumerate(json.load(f)):
                    key = (fn, i)
                    self.reordered_templates[key] = template

        # Create list of templates that we won't bother with, since they can't
        # be instantiated with HO if HO isn't in the image
        banned_list = [("comparison.json", -1)]
        banned_list += [("one_hop.json", i) for i in [2, 3, 4, 5]]
        banned_list += [("same_relate.json", i) for i in range(8)]
        banned_list += [("single_and.json", i) for i in [1, 2, 3, 4]]
        # single_or.json is fine
        banned_list += [("three_hop.json", i) for i in [2, 3, 4, 5]]
        banned_list += [("two_hop.json", i) for i in [2, 3, 4, 5]]
        banned_list += [("zero_hop.json", i) for i in [2, 3, 4, 5]]
        self.banned_templates = banned_list

        self.possible_attr_labels = [
            "<%s%s>" % (label, num)
            for label in ["Z", "C", "S", "M", "R"]
            for num in ["", "2", "3", "4"]
        ]
        self.possible_attribute_choices = tuple(
            (size, col, mat, sha)
            for size in [None] + CLEVR_SIZES
            for col in [None] + CLEVR_COLORS
            for mat in [None] + CLEVR_MATERIALS
            for sha in [None] + CLEVR_SHAPES
        )
        self.possible_rel_attribute_choices = tuple(
            (dir, (size, col, mat, sha))
            for size in [None] + CLEVR_SIZES
            for col in [None] + CLEVR_COLORS
            for mat in [None] + CLEVR_MATERIALS
            for sha in [None] + CLEVR_SHAPES
            for dir in ["left", "right", "front", "behind"]
        )

    def __call__(
        self,
        scene_struct,
        template,
        metadata,
        answer_counts,
        synonyms,
        template_name,
        max_instances=None,
        verbose=False,
        filter_option_filter: Callable[[list, dict, dict], list] = None,
        other_rejection_condition: Callable[[dict, dict, dict, Any, dict], bool] = None,
        add_empty_filter_options=None,  # Argument is ignored
    ) -> Tuple[List[str], List[List[Dict]], list]:
        if (
            filter_option_filter is not None
            or other_rejection_condition is not None
            or add_empty_filter_options is not None
        ):
            raise NotImplementedError(
                "Paired question generation is not currently supporting these options"
            )

        if max_instances is None or max_instances > 1:
            raise NotImplementedError(
                "Need to add de-duplication across instantiate_templates_dfs calls"
            )

        # If we know in advance that this template can't be instantiated with HO
        # given that HO isn't in the image; skip it.
        if (template_name in self.banned_templates) or (
            template_name[0],
            -1,
        ) in self.banned_templates:
            if verbose:
                print(f"Skip template {template_name}")
            return [], [], []

        reordered_template = self.reordered_templates[template_name]

        scene_attributes = set(
            (x["size"], x["color"], x["shape"], x["material"])
            for x in scene_struct["objects"]
        )

        # Figure out the number of position in which we could potentially insert ho
        opportunities = len(
            [
                node
                for node in template["nodes"]
                if "side_inputs" in node and len(node["side_inputs"]) >= 4
            ]
        )
        opportunity_nodes = [
            node
            for node in template["nodes"]
            if "side_inputs" in node and len(node["side_inputs"]) >= 4
        ]

        # randomize the order in which we attempt these
        positions = list(range(opportunities))

        random.shuffle(positions)

        text_questions = []
        structured_questions = []
        answers = []

        # Try the position in which we'll be inserting the held-out combination
        for opportunity_idx in positions:
            if "unique" in opportunity_nodes[opportunity_idx]["type"]:
                if verbose:
                    print(
                        f"Skip position {opportunity_idx}, template {template_name}, due to uniqueness constraint for node: {opportunity_nodes[opportunity_idx]['type']}"
                    )
                continue

            # NOTE: We have altered the question generator's behaviour; we can now
            #       add empty options to `relate` nodes, if the relate node is the
            #       one we're trying to make match HO.
            """
      if opportunity_nodes[opportunity_idx]['type'] not in ['filter_count', 'filter_exist', 'filter']:
        #if verbose:
        print("WARNING!!!")
        print(f"Skip position {opportunity_idx}, template {template_name}, due to {opportunity_nodes[opportunity_idx]['type']}")
        # Note: can't instantiate these b/c CLEVR questions only use non-trivial
        #       false combinations (i.e., the object exists but not in the
        #       described spatial relationship). As a result, we cannot instantiate
        #       the question if the held-out object is not in the image.
        continue
      """

            # Exit if we've already found enough questions
            if max_instances is not None and len(text_questions) == max_instances:
                break

            if verbose:
                print(
                    f"Attempting to insert HO into template: {template_name}, position: {opportunity_idx + 1} of {len(opportunity_nodes)}. Node: {opportunity_nodes[opportunity_idx]['type']}"
                )

            # Find the node we're trying to fill
            target_node = [
                node
                for node in template["nodes"]
                if "side_inputs" in node and len(node["side_inputs"]) >= 4
            ][opportunity_idx]

            # Run preemptive answer filtering when possible. This only catches
            # a subset of cases, but hopefully speeds things up.
            # Check if answer filtering stops us in counting problems
            if target_node["type"] in ["relate_filter_count", "filter_count"]:
                # Answer cannot exceed the total number of objects in the scene, which
                # match the least constrained version of HO (i.e., just HO without
                # additional constraints)
                max_possible_answer = len(
                    [x for x in scene_attributes if attr_match(x, self.train_ho)]
                )
                assert max_possible_answer == 0, [
                    x for x in scene_attributes if attr_match(x, self.train_ho)
                ]  # Since image shouldn't contain HO
                if all(
                    rejection_heuristic(answer_counts, ans)
                    for ans in range(max_possible_answer + 1)
                ):
                    if verbose:
                        print(
                            "Skipping; pre-emptive answer filtering; answer is at most: %i"
                            % max_possible_answer
                        )
                    continue
                elif verbose:
                    print("Preliminary answer checking OK, proceeding")
            # Check if answer filtering stops us in existence problems
            # NOTE: 'relate_filter_exist' normally can't instantiate at all; we've
            #       changed the code, in which case it'll return False
            elif target_node["type"] in [
                "filter_exist",
                "relate_filter_exist",
            ] and rejection_heuristic(answer_counts, False):
                assert not any(
                    attr_match(x, self.train_ho) for x in scene_attributes
                )  # HO should not be in the image
                if verbose:
                    print("Skipping; pre-emptive answer filtering; answer is 'false'")
                continue
            elif verbose:
                print("Preliminary answer checking OK, proceeding")

            # If there are constraints, check if that makes it impossible to insert
            # the held-out combination at {opportunity_idx}
            if "constraints" in template:
                # Figure out which attributes (e.g., <Z4>) must be NULL
                null_attrs = []
                for c in template["constraints"]:
                    if c["type"] == "NULL":
                        null_attrs += c["params"]

                for null_attr in null_attrs:
                    assert null_attr in self.possible_attr_labels

                for attr in target_node["side_inputs"]:
                    assert attr in self.possible_attr_labels, attr

                    if "<R" in attr:  # Ignore spatial relationship
                        continue
                    if attr in null_attrs:
                        # Need to figure out if this being null prevents us from instantiating HO
                        if (
                            (
                                attr in ["<Z>", "<Z2>", "<Z3>", "<Z4>"]
                                and self.train_ho[0] is not None
                            )
                            or (
                                attr in ["<C>", "<C2>", "<C3>", "<C4>"]
                                and self.train_ho[1] is not None
                            )
                            or (
                                attr in ["<S>", "<S2>", "<S3>", "<S4>"]
                                and self.train_ho[2] is not None
                            )
                            or (
                                attr in ["<M>", "<M2>", "<M3>", "<M4>"]
                                and self.train_ho[3] is not None
                            )
                        ):
                            if verbose:
                                print("Skip b/c template constraints do not allow")
                            return [], [], []

            # If the reordered template has the opportunity closer to the start
            # of the program (and therefore the constraint will be forced sooner
            # in the DFS performed by instantiate_templates_dfs), then use the
            # reordered template instead
            template_in_use = template
            if opportunity_idx > reordered_template["opp_mapping"][opportunity_idx]:
                if verbose:
                    print("Using reordered template")
                template_in_use = reordered_template
                opportunity_idx = reordered_template["opp_mapping"][opportunity_idx]

            tq, sq, a = self.instantiate_pair(
                target_position=opportunity_idx,
                scene_struct=scene_struct,
                template=template_in_use,
                metadata=metadata,
                answer_counts=answer_counts,
                synonyms=synonyms,
                max_instances=None
                if max_instances is None
                else max_instances
                - len(text_questions),  # however many are left to be made
                verbose=verbose,
            )

            text_questions.extend(tq)
            structured_questions.extend(sq)
            answers.extend(a)

        return text_questions, structured_questions, answers

    @staticmethod
    def _paired_shuffle(list1, list2):
        """
        Shuffle 2 lists in the same way, and return them.
        Does NOT mutate the original lists.
        """
        assert len(list1) == len(list2)
        if len(list1) == 0:
            return [], []

        temp = list(zip(list1, list2))
        random.shuffle(temp)
        res1, res2 = zip(*temp)
        return list(res1), list(res2)

    def _intersect_filter_options(
        self, in_options: list, paired_in_options: list
    ) -> Tuple[list, list]:
        """
        Given two lists of possible filter options, return two lists of choices
        that may work.
        The first list is for the main question (containing ho), the second list
        is for the paired question (not containing ho).
        An option may work if it does not contain ho and is in both, or
        if it is in in_options & contains self.train_ho, and by swapping
        self.train_ho for self.alt_ho we get something in paired_in_options.

        NOTE! nothing in the returned paired_in_options list is allowed to contain
        self.train_ho!

        This does NOT mutate the input lists, but returns 2 lists instead.
        """
        # Modified from the filter position_matches_ho
        # Convert from generators to tuples for simplicity
        in_options = [tuple(x) for x in in_options]
        paired_in_options = [tuple(x) for x in paired_in_options]

        # Give up on dead ends
        if len(in_options) == 0 or len(paired_in_options) == 0:
            return [], []

        # Check all our options are of the same length
        lens = set(len(x) for x in in_options)
        assert len(lens) == 1
        paired_lens = set(len(x) for x in paired_in_options)
        assert len(paired_lens) == 1
        # Check both questions have the same number of options
        assert lens == paired_lens

        # Define method that gets a tuple out of an entry in in_options
        # note that in_options is either a list of 4-tuples of attributes, or it
        # is a list of (direction, (4-tuple)) tuples
        # Also re-orders the attributes to the standard ordering
        if len(in_options[0]) == 4:
            get_attr_tuple = lambda x: (x[0], x[1], x[3], x[2])
        else:
            get_attr_tuple = lambda x: (x[1][0], x[1][1], x[1][3], x[1][2])

        final_in_options = []
        paired_final_in_options = []
        for x in in_options:
            # If x doesn't match ho, and is a viable choice for both questions,
            # then include it as a possibility for both questions
            if (
                not attr_match(get_attr_tuple(x), template=self.train_ho)
                and x in paired_in_options
            ):
                final_in_options.append(x)
                paired_final_in_options.append(x)
            # Otherwise, if x matches ho, then we can only include it if the
            # same attributes but replaced with alt_ho are valid for the paired
            # question
            elif attr_match(get_attr_tuple(x), template=self.train_ho):
                attr_tup = get_attr_tuple(x)
                # Get the alt_ho equivalent of this attribute tuple
                pair_x = tuple(
                    None
                    if attr is None
                    else attr
                    if attr != self.train_ho[i]
                    else self.alt_ho[i]
                    for i, attr in enumerate(attr_tup)
                )
                assert not attr_match(pair_x, template=self.train_ho)
                assert attr_match(pair_x, template=self.alt_ho)
                for i in range(4):
                    if self.train_ho[i] is None:
                        assert self.alt_ho[i] is None
                        assert pair_x[i] == attr_tup[i]

                # Reorder to the input/output format:
                pair_x = (pair_x[0], pair_x[1], pair_x[3], pair_x[2])

                if len(x) != 4:  # Then we must include the relationship
                    pair_x = (x[0], pair_x)

                if pair_x in paired_in_options:
                    final_in_options.append(x)
                    paired_final_in_options.append(pair_x)

        assert len(final_in_options) == len(paired_final_in_options)
        return final_in_options, paired_final_in_options

    def _paired_filter(
        self, in_options, paired_in_options, state, paired_state, position, template
    ) -> Tuple[list, list]:
        # LOGIC:
        # -1 indicates we are NOT trying to insert HO here.
        # If at position:
        #     1) Prevent paired_in_options from containing self.train_ho
        #     2) Filter in_options so that all must match self.train_ho
        #     3) Intersect. Note that anything in in_options that matches self.train_ho
        #                   should *not correspond* to the same in paired_in_options,
        #                   but rather should map to the substitution of self.train_ho
        #                   for self.alt_ho
        # If not at position:
        #     1) Prevent paired_in_options from containing self.train_ho
        #     2) Intersect. Note that anything in in_options that matches self.train_ho
        #                   should *not correspond* to the same in paired_in_options,
        #                   but rather should map to the substitution of self.train_ho
        #                   for self.alt_ho
        assert position == -1 or position >= 0

        # Modified from the filter position_matches_ho
        # Convert from generators to tuples for simplicity
        in_options = [tuple(x) for x in in_options]
        paired_in_options = [tuple(x) for x in paired_in_options]

        # Give up on dead ends
        if len(in_options) == 0 or len(paired_in_options) == 0:
            return [], []

        # Check all our options are of the same length
        lens = set(len(x) for x in in_options)
        assert len(lens) == 1
        paired_lens = set(len(x) for x in paired_in_options)
        assert len(paired_lens) == 1
        # Check both questions have the same number of options
        assert lens == paired_lens

        # Define method that gets a tuple out of an entry in in_options
        # note that in_options is either a list of 4-tuples of attributes, or it
        # is a list of (direction, (4-tuple)) tuples
        # Also re-orders the attributes to the standard ordering
        if len(in_options[0]) == 4:
            get_attr_tuple = lambda x: (x[0], x[1], x[3], x[2])
        else:
            get_attr_tuple = lambda x: (x[1][0], x[1][1], x[1][3], x[1][2])

        # Check the method works, and that attributes are in the expected order
        test_tup = get_attr_tuple(in_options[-1])
        assert len(test_tup) == 4
        assert test_tup[0] in CLEVR_SIZES + [None]
        assert test_tup[1] in CLEVR_COLORS + [None]
        assert test_tup[2] in CLEVR_SHAPES + [None]
        assert test_tup[3] in CLEVR_MATERIALS + [None]
        test_tup = get_attr_tuple(paired_in_options[-1])
        assert len(test_tup) == 4
        assert test_tup[0] in CLEVR_SIZES + [None]
        assert test_tup[1] in CLEVR_COLORS + [None]
        assert test_tup[2] in CLEVR_SHAPES + [None]
        assert test_tup[3] in CLEVR_MATERIALS + [None]

        # The paired question CANNOT contain HO! Filter accordingly
        paired_in_options = [
            x
            for x in paired_in_options
            if not attr_match(get_attr_tuple(x), template=self.train_ho)
        ]

        # If the paired question is now a dead end, give up
        if len(paired_in_options) == 0:
            return [], []

        # Check that there actually *is* a chance to insert the held-out combination
        opportunities = len(
            [
                node
                for node in template["nodes"]
                if "side_inputs" in node and len(node["side_inputs"]) >= 4
            ]
        )
        assert position == -1 or opportunities >= position + 1, template

        # Figure out what's already been filled
        completed_opportunities = determine_completed_opportunities(state)
        assert completed_opportunities == determine_completed_opportunities(
            paired_state
        )

        # Make sure we haven't somehow filled more slots than exist
        assert completed_opportunities <= opportunities

        # If we are not currently filling in {position}, then we need to choose
        # the options that work for both questions, *under the substitution of
        # alt_ho for ho!*
        if completed_opportunities != position:
            return self._intersect_filter_options(
                in_options=in_options, paired_in_options=paired_in_options
            )

        # Otherwise, we must filter the in_options to include only those matching ho
        in_options = [
            x
            for x in in_options
            if attr_match(get_attr_tuple(x), template=self.train_ho)
        ]

        # Now return the "intersection"; again under substitution of alt_ho for ho
        # in the paired question.
        return self._intersect_filter_options(
            in_options=in_options, paired_in_options=paired_in_options
        )

    def _add_empty_options(
        self,
        attribute_map,
        paired_attribute_map,
        state,
        paired_state,
        target_position,
        next_node,
        num_to_add,
    ):
        # Note: *slightly* more stringent than required.
        #       For count & exists; we cannot be allowed to have
        #       matching object in either, else will be rejected later due to
        #       different answers.
        #       However, for raw filter then we could plausibly have a situation
        #       where one image has an object, and the other none, and the answer
        #       is the same. For simplicity (and more similarity between splits)
        #       we will discard this possibility.

        # Figure out how many opportunities have already been filled
        completed_opps = determine_completed_opportunities(state)
        assert completed_opps == determine_completed_opportunities(paired_state)

        # In principle, the paired question can have any attribute so long as
        # it doesn't match ho, and it doesn't match an object in the scene
        paired_possible_choices = [
            x
            for x in self.possible_attribute_choices
            if not attr_match((x[0], x[1], x[3], x[2]), self.train_ho)
            and x not in paired_attribute_map
        ]

        # For the main question,
        # If the next_node allows us to fully specify all 4 attributes, AND
        # *this* is the one we want to make HO, then restrict our negative samples
        # to those matching HO
        if (
            completed_opps == target_position
            and "side_inputs" in next_node
            and len(next_node["side_inputs"]) >= 4
        ):
            # Filter the 96 possible objects to find what matches HO and isn't
            # in the set of combinations that correspond to some objects
            possible_choices = [
                x
                for x in self.possible_attribute_choices
                if attr_match((x[0], x[1], x[3], x[2]), self.train_ho)
                and x not in attribute_map
            ]

        else:
            # Otherwise, just filter the 96 possible objects to find what
            # doesn't correspond to an object
            possible_choices = [
                x for x in self.possible_attribute_choices if x not in attribute_map
            ]

        # Our choice must be viable for *both* questions, so take the intersection
        possible_choices, paired_possible_choices = self._intersect_filter_options(
            in_options=possible_choices, paired_in_options=paired_possible_choices
        )
        # Now shuffle, and limit our choices to however many need adding
        possible_choices, paired_possible_choices = self._paired_shuffle(
            possible_choices, paired_possible_choices
        )
        possible_choices = possible_choices[:num_to_add]
        paired_possible_choices = paired_possible_choices[:num_to_add]

        # Note: Critical to preserve order so that options remain aligned!
        # In this case, I think we're fine because we will re-call
        # _intersect_filter_options before we actually use the values, and that
        # will return lists
        for x, paired_x in zip(possible_choices, paired_possible_choices):
            attribute_map[x] = []
            paired_attribute_map[paired_x] = []

    def _add_empty_relate_options_matching_ho(
        self, attribute_map, paired_attribute_map, num_to_add
    ):
        """
        This method is *not* normal CLEVR behaviour. It adds degenerate relate filter
        options. Specifically, matching the held-out combination.
        """
        pos_rel_options = self.possible_rel_attribute_choices

        # In principle, the paired question can have any attribute so long as
        # it doesn't match ho, and it doesn't match an object in the scene
        paired_possible_choices = [
            x
            for x in pos_rel_options
            if not attr_match((x[1][0], x[1][1], x[1][3], x[1][2]), self.train_ho)
            and x not in paired_attribute_map
        ]

        # For the main question,
        # Filter the 96 possible objects to find what matches HO and isn't
        # in the set of combinations that correspond to some objects
        possible_choices = [
            x
            for x in pos_rel_options
            if attr_match((x[1][0], x[1][1], x[1][3], x[1][2]), self.train_ho)
            and x not in attribute_map
        ]

        # Our choice must be viable for *both* questions, so take the intersection
        possible_choices, paired_possible_choices = self._intersect_filter_options(
            in_options=possible_choices, paired_in_options=paired_possible_choices
        )
        # Now shuffle, and limit our choices to however many need adding
        possible_choices, paired_possible_choices = self._paired_shuffle(
            possible_choices, paired_possible_choices
        )
        possible_choices = possible_choices[:num_to_add]
        paired_possible_choices = paired_possible_choices[:num_to_add]

        # Note: Critical to preserve order so that options remain aligned!
        # In this case, I think we're fine because we will re-call
        # _intersect_filter_options before we actually use the values, and that
        # will return lists
        for x, paired_x in zip(possible_choices, paired_possible_choices):
            attribute_map[x] = []
            paired_attribute_map[paired_x] = []

    def instantiate_pair(
        self,
        scene_struct,
        template,
        metadata,
        answer_counts,
        synonyms,
        target_position: int,
        max_instances=None,
        verbose=False,
    ) -> Tuple[List[str], List[List[Dict]], list]:
        """
        Instantiate such that ho is at {target_position}, and paired
        question has the replacement for alt_ho.
        Furthermore, paired question cannot contain ho anywhere.
        """
        assert max_instances > 0

        param_name_to_type = {p["name"]: p["type"] for p in template["params"]}
        # (size, colour, mat, shape)
        param_name_to_idx = {"Size": 0, "Color": 1, "Material": 2, "Shape": 3}

        initial_state = {
            "nodes": [node_shallow_copy(template["nodes"][0])],
            "vals": {},
            "input_map": {0: 0},
            "next_template_node": 1,
        }
        states = [initial_state]
        paired_states = copy.deepcopy(states)
        final_states = []
        paired_final_states = []
        while states:
            state = states.pop()
            paired_state = paired_states.pop()

            # Check to make sure the current state is valid
            q = {"nodes": state["nodes"]}
            paired_q = {"nodes": paired_state["nodes"]}
            outputs = qeng.answer_question(q, metadata, scene_struct, all_outputs=True)
            paired_outputs = qeng.answer_question(
                paired_q, metadata, scene_struct, all_outputs=True
            )

            answer = outputs[-1]
            paired_answer = paired_outputs[-1]
            if answer == "__INVALID__" or paired_answer == "__INVALID__":
                if verbose:
                    print("Discarding invalid state")
                continue

            # Check to make sure constraints are satisfied for the current state
            skip_state = False
            for constraint in template["constraints"]:
                if constraint["type"] == "NEQ":
                    """
                    p1, p2 = constraint['params']
                    for v1, v2 in [(state['vals'].get(p1), state['vals'].get(p2)),
                                   (paired_state['vals'].get(p1), paired_state['vals'].get(p2))]:

                      if v1 is not None and v2 is not None and v1 != v2:
                        if verbose:
                          print('skipping due to NEQ constraint')
                          print(constraint)
                          print(state['vals'])
                        skip_state = True
                        break
                      if skip_state:
                        break
                    """
                    raise ValueError("NEQ constraint isn't used by CLEVR templates.")
                elif constraint["type"] == "NULL":
                    p = constraint["params"][0]
                    p_type = param_name_to_type[p]
                    for v in [state["vals"].get(p), paired_state["vals"].get(p)]:
                        if v is not None:
                            skip = False
                            if p_type == "Shape" and v != "thing":
                                skip = True
                            if p_type != "Shape" and v != "":
                                skip = True
                            if skip:
                                if verbose:
                                    print("skipping due to NULL constraint")
                                    print(constraint)
                                    print(state["vals"])
                                skip_state = True
                                break
                        if skip_state:
                            break
                elif constraint["type"] == "OUT_NEQ":
                    i_idx, j_idx = constraint["params"]

                    for i, j, inner_outputs in [
                        (
                            state["input_map"].get(i_idx, None),
                            state["input_map"].get(j_idx, None),
                            outputs,
                        ),
                        (
                            paired_state["input_map"].get(i_idx, None),
                            paired_state["input_map"].get(j_idx, None),
                            paired_outputs,
                        ),
                    ]:
                        if i is not None and j is not None and outputs[i] == outputs[j]:
                            if verbose:
                                print("skipping due to OUT_NEQ constraint")
                                print(inner_outputs[i])
                                print(inner_outputs[j])
                            skip_state = True
                            break
                        if skip_state:
                            break
                else:
                    assert False, (
                        'Unrecognized constraint type "%s"' % constraint["type"]
                    )

            if skip_state:
                continue

            # We have already checked to make sure the answer is valid, so if we have
            # processed all the nodes in the template then the current state is a valid
            # question, so add it if it passes our rejection sampling tests.
            if state["next_template_node"] == len(template["nodes"]):
                if verbose:
                    print("Testing Proposed Question")
                assert paired_state["next_template_node"] == len(template["nodes"])

                # Check both questions have the same answer
                if answer != paired_answer:
                    if verbose:
                        print("Reject -- questions have different answers")
                    continue

                if rejection_heuristic(answer_counts, answer, verbose):
                    if verbose:
                        print("Rejected by answer counting heuristic")
                    continue

                # If the template contains a raw relate node then we need to check for
                # degeneracy at the end
                # A question is degenerate if replacing any of its relate nodes with a scene
                # node results in a question with the same answer.
                # NOTE: Need to disable degeneracy check for AND
                # And is also the *only* question template that uses a raw relate node,
                # so we simply REMOVE the entire degeneracy check.
                """
        has_relate = any(n['type'] == 'relate' for n in template['nodes'])
        if has_relate:
          degen = qeng.is_degenerate(q, metadata, scene_struct, answer=answer,
                                     verbose=verbose)
          degen = degen or qeng.is_degenerate(paired_q, metadata,
                                              scene_struct,
                                              answer=paired_answer,
                                              verbose=verbose)
          if degen:
            if verbose:
              print("Rejected because degenerate")

            continue
        """

                answer_counts[answer] += 1
                state["answer"] = answer
                paired_state["answer"] = paired_answer
                final_states.append(state)
                paired_final_states.append(paired_state)
                if max_instances is not None and len(final_states) == max_instances:
                    break
                continue

            # Otherwise fetch the next node from the template
            # Make a shallow copy so cached _outputs don't leak ... this is very nasty
            next_node = template["nodes"][state["next_template_node"]]
            next_node = node_shallow_copy(next_node)

            special_nodes = {
                "filter_unique",
                "filter_count",
                "filter_exist",
                "filter",
                "relate_filter",
                "relate_filter_unique",
                "relate_filter_count",
                "relate_filter_exist",
            }

            if next_node["type"] in special_nodes:
                if next_node["type"].startswith("relate_filter"):
                    unique = next_node["type"] == "relate_filter_unique"
                    include_zero = (
                        next_node["type"] == "relate_filter_count"
                        or next_node["type"] == "relate_filter_exist"
                    )
                    filter_options = find_relate_filter_options(
                        answer,
                        scene_struct,
                        metadata,
                        unique=unique,
                        include_zero=include_zero,
                    )
                    paired_filter_options = find_relate_filter_options(
                        paired_answer,
                        scene_struct,
                        metadata,
                        unique=unique,
                        include_zero=include_zero,
                    )
                else:
                    filter_options = find_filter_options(
                        answer, scene_struct, metadata
                    )  # Dict (size, colour, mat, shape) to list of indices of matching objects. None for not specified.
                    paired_filter_options = find_filter_options(
                        paired_answer, scene_struct, metadata
                    )

                    if next_node["type"] == "filter":
                        # Remove null filter
                        filter_options.pop((None, None, None, None), None)
                        paired_filter_options.pop((None, None, None, None), None)
                    if next_node["type"] == "filter_unique":
                        # Get rid of all filter options that don't result in a single object
                        filter_options = {
                            k: v for k, v in filter_options.items() if len(v) == 1
                        }
                        paired_filter_options = {
                            k: v
                            for k, v in paired_filter_options.items()
                            if len(v) == 1
                        }
                    else:
                        # Figure out how many non-empty options we will have once we're done
                        tmp_fo, tmp_p_fo = self._paired_filter(
                            in_options=list(filter_options.keys()),
                            paired_in_options=list(paired_filter_options.keys()),
                            state=state,
                            paired_state=paired_state,
                            position=target_position,
                            template=template,
                        )
                        assert len(tmp_fo) == len(tmp_p_fo)

                        # Add some filter options that do NOT correspond to the scene
                        if next_node["type"] == "filter_exist":
                            # For filter_exist we want an equal number that do and don't
                            num_to_add = len(tmp_fo)
                        else:
                            assert (
                                next_node["type"] == "filter_count"
                                or next_node["type"] == "filter"
                            )  # Note: reworked from original code; possible problem
                            # For filter_count add nulls equal to the number of singletons
                            num_to_add = sum(
                                1
                                for k, v in filter_options.items()
                                if len(v) == 1 and k in tmp_fo
                            )  # can't deal with pair here, it'll be rejected later, fine.

                        # NOTE! Up to this point, the program cannot add descriptions of
                        # objects that do not exist. As HO does not exist in the image,
                        # we therefore have no descriptions matching HO.
                        # Therefore, when we reach the slot we're trying to fill, we will
                        # have no descriptions matching HO, therefore _paired_filter will
                        # purge _everything_
                        # Therefore, num_to_add will be zero. We must override this
                        # behaviour manually; setting to 1 instead should do the trick.
                        if determine_completed_opportunities(state) == target_position:
                            assert len(tmp_fo) == 0
                            assert num_to_add == 0
                            num_to_add = 1

                        self._add_empty_options(
                            attribute_map=filter_options,
                            paired_attribute_map=paired_filter_options,
                            state=state,
                            paired_state=paired_state,
                            target_position=target_position,
                            next_node=next_node,
                            num_to_add=num_to_add,
                        )

                filter_option_keys = list(filter_options.keys())
                paired_filter_option_keys = list(paired_filter_options.keys())
                # Merge & filter our filter option keys.
                # if verbose and (len(filter_option_keys) == 0 or len(paired_filter_option_keys) == 0):
                #  print(f"No valid choices before filtering {len(filter_option_keys)} {len(paired_filter_option_keys)}")
                filter_option_keys, paired_filter_option_keys = self._paired_filter(
                    in_options=filter_option_keys,
                    paired_in_options=paired_filter_option_keys,
                    state=state,
                    paired_state=paired_state,
                    position=target_position,
                    template=template,
                )

                # NEW CODE!: Change the search behaviour around the nodes
                # relate_filter_count, relate_filter, and relate_filter_exist
                # But *only* when we're trying to make them match HO.
                if (
                    next_node["type"]
                    in ["relate_filter_count", "relate_filter", "relate_filter_exist"]
                    and determine_completed_opportunities(state) == target_position
                ):
                    # Normally these nodes only use descriptions of existing objects;
                    # as HO isn't in the image, there shouldn't be *any* available choices
                    # after filtering for those that match HO.
                    assert len(filter_option_keys) == 0

                    # Add an empty option matching HO
                    self._add_empty_relate_options_matching_ho(
                        attribute_map=filter_options,
                        paired_attribute_map=paired_filter_options,
                        num_to_add=1,
                    )

                    # Re-run b/c key lists must be aligned
                    filter_option_keys, paired_filter_option_keys = self._paired_filter(
                        in_options=list(filter_options.keys()),
                        paired_in_options=list(paired_filter_options.keys()),
                        state=state,
                        paired_state=paired_state,
                        position=target_position,
                        template=template,
                    )
                    assert len(filter_option_keys) == 1
                    assert len(paired_filter_option_keys) == 1

                filter_option_keys, paired_filter_option_keys = self._paired_shuffle(
                    filter_option_keys, paired_filter_option_keys
                )

                # if verbose and len(filter_option_keys) == 0:
                #  print("No valid choices left after filtering")

                for k, paired_k in zip(filter_option_keys, paired_filter_option_keys):
                    new_nodes = []
                    paired_new_nodes = []

                    cur_next_vals = {k: v for k, v in state["vals"].items()}
                    paired_cur_next_vals = {
                        k: v for k, v in paired_state["vals"].items()
                    }

                    next_input = state["input_map"][next_node["inputs"][0]]
                    assert (
                        next_input == paired_state["input_map"][next_node["inputs"][0]]
                    )

                    filter_side_inputs = next_node["side_inputs"]
                    if next_node["type"].startswith("relate"):
                        param_name = next_node["side_inputs"][
                            0
                        ]  # First should be relate
                        filter_side_inputs = next_node["side_inputs"][1:]
                        param_type = param_name_to_type[param_name]
                        assert param_type == "Relation"
                        param_val = k[0]
                        paired_param_val = paired_k[0]

                        k = k[1]
                        paired_k = paired_k[1]

                        new_nodes.append(
                            {
                                "type": "relate",
                                "inputs": [next_input],
                                "side_inputs": [param_val],
                            }
                        )
                        paired_new_nodes.append(
                            {
                                "type": "relate",
                                "inputs": copy.deepcopy([next_input]),
                                "side_inputs": [paired_param_val],
                            }
                        )

                        cur_next_vals[param_name] = param_val
                        paired_cur_next_vals[param_name] = paired_param_val

                        next_input = len(state["nodes"]) + len(new_nodes) - 1
                        assert len(state["nodes"]) == len(paired_state["nodes"])
                        assert len(new_nodes) == len(paired_new_nodes)

                    for param_name, param_val, paired_param_val in zip(
                        filter_side_inputs, k, paired_k
                    ):
                        param_type = param_name_to_type[param_name]
                        filter_type = "filter_%s" % param_type.lower()
                        if param_val is not None or paired_param_val is not None:
                            assert (
                                param_val is not None and paired_param_val is not None
                            )  # Good enough for now
                            new_nodes.append(
                                {
                                    "type": filter_type,
                                    "inputs": [next_input],
                                    "side_inputs": [param_val],
                                }
                            )
                            paired_new_nodes.append(
                                {
                                    "type": filter_type,
                                    "inputs": copy.deepcopy([next_input]),
                                    "side_inputs": [paired_param_val],
                                }
                            )
                            cur_next_vals[param_name] = param_val
                            paired_cur_next_vals[param_name] = paired_param_val
                            next_input = len(state["nodes"]) + len(new_nodes) - 1
                            assert len(state["nodes"]) == len(paired_state["nodes"])
                            assert len(new_nodes) == len(paired_new_nodes)
                        elif param_val is None:
                            assert paired_param_val is None
                            if (
                                metadata["dataset"] == "CLEVR-v1.0"
                                and param_type == "Shape"
                            ):
                                param_val = "thing"
                                paired_param_val = "thing"
                            else:
                                param_val = ""
                                paired_param_val = ""
                            cur_next_vals[param_name] = param_val
                            paired_cur_next_vals[param_name] = paired_param_val
                    input_map = {k: v for k, v in state["input_map"].items()}
                    assert input_map == {
                        k: v for k, v in paired_state["input_map"].items()
                    }

                    extra_type = None
                    if next_node["type"].endswith("unique"):
                        extra_type = "unique"
                    if next_node["type"].endswith("count"):
                        extra_type = "count"
                    if next_node["type"].endswith("exist"):
                        extra_type = "exist"
                    if extra_type is not None:
                        new_nodes.append(
                            {
                                "type": extra_type,
                                "inputs": [
                                    input_map[next_node["inputs"][0]] + len(new_nodes)
                                ],
                            }
                        )
                        paired_new_nodes.append(
                            {
                                "type": extra_type,
                                "inputs": copy.deepcopy(
                                    [
                                        input_map[next_node["inputs"][0]]
                                        + len(paired_new_nodes)
                                    ]
                                ),
                            }
                        )
                        assert len(new_nodes) == len(paired_new_nodes)
                    input_map[state["next_template_node"]] = (
                        len(state["nodes"]) + len(new_nodes) - 1
                    )
                    assert (
                        state["next_template_node"]
                        == paired_state["next_template_node"]
                    )
                    assert len(state["nodes"]) == len(paired_state["nodes"])
                    assert len(new_nodes) == len(paired_new_nodes)

                    states.append(
                        {
                            "nodes": state["nodes"] + new_nodes,
                            "vals": cur_next_vals,
                            "input_map": input_map,
                            "next_template_node": state["next_template_node"] + 1,
                        }
                    )
                    paired_states.append(
                        {
                            "nodes": paired_state["nodes"] + paired_new_nodes,
                            "vals": paired_cur_next_vals,
                            "input_map": copy.deepcopy(input_map),
                            "next_template_node": paired_state["next_template_node"]
                            + 1,
                        }
                    )

            elif "side_inputs" in next_node:  # e.g., 'filter_size' atom
                # If the next node has template parameters, expand them out
                # CLEVR: Generalize this to work for nodes with more than one side input
                assert len(next_node["side_inputs"]) == 1, "NOT IMPLEMENTED"

                # Use metadata to figure out domain of valid values for this parameter.
                # Iterate over the values in a random order; then it is safe to bail
                # from the DFS as soon as we find the desired number of valid template
                # instantiations.
                param_name = next_node["side_inputs"][0]
                param_type = param_name_to_type[
                    param_name
                ]  # 'Size', 'Color', 'Material' or 'Shape'
                param_vals = metadata["types"][param_type][
                    :
                ]  # NOTE: question-agnostic up to here
                random.shuffle(param_vals)

                # Now perform any needed filtering.
                if param_type != "Relation":
                    assert next_node["type"] in [
                        "filter_color",
                        "filter_size",
                        "filter_shape",
                        "filter_material",
                    ], next_node["type"]
                    # convert the parameter values into a list of (size, color, material, shape) tuple
                    assert param_type in param_name_to_idx
                    tmp_idx = param_name_to_idx[param_type]
                    param_vals = [
                        ((None,) * 4)[:tmp_idx] + (val,) + ((None,) * 4)[tmp_idx + 1 :]
                        for val in param_vals
                    ]
                    # filter & convert back into a list of values
                    param_vals, paired_param_vals = self._paired_filter(
                        in_options=param_vals,
                        paired_in_options=copy.deepcopy(param_vals),
                        state=state,
                        paired_state=paired_state,
                        position=-1,  # NOTE! This is only a relationship, therefore must indicate to filter that this cannot be where we're trying to insert HO.
                        template=template,
                    )
                    param_vals = [x[tmp_idx] for x in param_vals]
                    paired_param_vals = [x[tmp_idx] for x in paired_param_vals]
                else:
                    assert next_node["type"] in ["relate"], next_node["type"]
                    # convert the parameter values into a list of (relation, (size, color, mat, shape)) tuples
                    param_vals = [(val, (None,) * 4) for val in param_vals]
                    # filter, and undo converstion
                    param_vals, paired_param_vals = self._paired_filter(
                        in_options=param_vals,
                        paired_in_options=copy.deepcopy(param_vals),
                        state=state,
                        paired_state=paired_state,
                        position=-1,  # NOTE! This is only a relationship, therefore must indicate to filter that this cannot be where we're trying to insert HO.
                        template=template,
                    )
                    param_vals = [x[0] for x in param_vals]
                    # In this case, just a relation, so we're good
                    paired_param_vals = [x[0] for x in paired_param_vals]

                # Extra shuffle just in case
                param_vals, paired_param_vals = self._paired_shuffle(
                    param_vals, paired_param_vals
                )

                for val, paired_val in zip(param_vals, paired_param_vals):
                    input_map = {k: v for k, v in state["input_map"].items()}
                    assert input_map == {
                        k: v for k, v in paired_state["input_map"].items()
                    }

                    input_map[state["next_template_node"]] = len(state["nodes"])
                    assert (
                        state["next_template_node"]
                        == paired_state["next_template_node"]
                    )
                    assert len(state["nodes"]) == len(paired_state["nodes"])
                    cur_next_node = {
                        "type": next_node["type"],
                        "inputs": [input_map[idx] for idx in next_node["inputs"]],
                        "side_inputs": [val],
                    }
                    paired_cur_next_node = {
                        "type": next_node["type"],
                        "inputs": copy.deepcopy(
                            [input_map[idx] for idx in next_node["inputs"]]
                        ),
                        "side_inputs": [paired_val],
                    }
                    cur_next_vals = {k: v for k, v in state["vals"].items()}
                    paired_cur_next_vals = {
                        k: v for k, v in paired_state["vals"].items()
                    }
                    cur_next_vals[param_name] = val
                    paired_cur_next_vals[param_name] = paired_val

                    states.append(
                        {
                            "nodes": state["nodes"] + [cur_next_node],
                            "vals": cur_next_vals,
                            "input_map": input_map,
                            "next_template_node": state["next_template_node"] + 1,
                        }
                    )
                    paired_states.append(
                        {
                            "nodes": paired_state["nodes"] + [paired_cur_next_node],
                            "vals": paired_cur_next_vals,
                            "input_map": copy.deepcopy(input_map),
                            "next_template_node": paired_state["next_template_node"]
                            + 1,
                        }
                    )
            else:
                input_map = {k: v for k, v in state["input_map"].items()}
                assert input_map == {k: v for k, v in paired_state["input_map"].items()}
                input_map[state["next_template_node"]] = len(state["nodes"])
                assert state["next_template_node"] == paired_state["next_template_node"]
                assert len(state["nodes"]) == len(paired_state["nodes"])
                next_node = {
                    "type": next_node["type"],
                    "inputs": [input_map[idx] for idx in next_node["inputs"]],
                }
                states.append(
                    {
                        "nodes": state["nodes"] + [next_node],
                        "vals": state["vals"],
                        "input_map": input_map,
                        "next_template_node": state["next_template_node"] + 1,
                    }
                )
                paired_states.append(
                    {
                        "nodes": paired_state["nodes"] + copy.deepcopy([next_node]),
                        "vals": paired_state["vals"],
                        "input_map": copy.deepcopy(input_map),
                        "next_template_node": paired_state["next_template_node"] + 1,
                    }
                )

        # Actually instantiate the template with the solutions we've found
        text_questions, structured_questions, answers = [], [], []
        paired_answers = []
        assert len(final_states) == len(paired_final_states)
        for state, paired_state in zip(final_states, paired_final_states):
            structured_questions.append(state["nodes"])
            answers.append(state["answer"])
            paired_answers.append(paired_state["answer"])

            text = random.choice(template["text"])
            paired_text = copy.deepcopy(text)

            for (name, val), (paired_name, paired_val) in zip(
                state["vals"].items(), paired_state["vals"].items()
            ):
                assert name == paired_name
                if val == paired_val:
                    # Make the same choices for both
                    if val in synonyms:
                        val = random.choice(synonyms[val])
                    text = text.replace(name, val)
                    paired_text = paired_text.replace(
                        name, val
                    )  # NOTE: name == paired_name, val == paired_val
                    text = " ".join(text.split())
                    paired_text = " ".join(paired_text.split())
                else:
                    # Separate choices
                    # First question
                    if val in synonyms:
                        val = random.choice(synonyms[val])
                    text = text.replace(name, val)
                    text = " ".join(text.split())
                    # Paired Question
                    if paired_val in synonyms:
                        paired_val = random.choice(synonyms[paired_val])
                    paired_text = paired_text.replace(paired_name, paired_val)
                    paired_text = " ".join(paired_text.split())

            text, paired_text = replace_paired_optionals(text, paired_text)
            text = " ".join(text.split())
            paired_text = " ".join(paired_text.split())
            text = other_heuristic(text, state["vals"], verbose=verbose)
            # heuristic is deterministic, so we don't need to make a paired version
            paired_text = other_heuristic(
                paired_text, paired_state["vals"], verbose=verbose
            )
            text_questions.append(text)

            # record mapping from image index & question, to the corresponding
            # question
            pair_key = (scene_struct["image_index"], text)
            if pair_key in self.paired_questions:
                print(scene_struct["image_index"])
                print(state)
                print(paired_state)
                print(text)
                print(paired_text)
                print(self.paired_questions[pair_key], flush=True)

            assert pair_key not in self.paired_questions
            self.paired_questions[pair_key] = (paired_state["nodes"], paired_text)

        assert paired_answers == answers
        return text_questions, structured_questions, answers


def run_stage2(args, scene_data, train_ho, OUT_DIR):
    A_choices = ("small", "gray", "cube", "metal")
    split = "train" if "train" in args.split else "val_iid"
    full_sg_path = f"{OUT_DIR}/scenes/CLEVR_held_out_{split}.json"

    # Get alternate combination to be used in the paired questions
    alt_ho = [None] * 4
    for i in range(4):
        if train_ho[i] is not None:
            alt_ho[i] = A_choices[i]
    alt_ho = tuple(alt_ho)

    print("Start Stage 2:")
    stage2_match_ho_path = f"{OUT_DIR}/questions/{split}/tmp/CLEVR_held_out_questions_S2_match_ho_{split}_{str(train_ho)}.json"
    stage2_pair_ho_path = f"{OUT_DIR}/questions/{split}/tmp/CLEVR_held_out_questions_S2_pair_ho_{split}_{str(train_ho)}.json"

    if args.limit >= 0:
        stage2_match_ho_path = f"{OUT_DIR}/questions/{split}/tmp/CLEVR_held_out_questions_S2_match_ho_{split}_lim{args.limit}_{str(train_ho)}.json"
        stage2_pair_ho_path = f"{OUT_DIR}/questions/{split}/tmp/CLEVR_held_out_questions_S2_pair_ho_{split}_lim{args.limit}_{str(train_ho)}.json"

    if os.path.exists(stage2_match_ho_path) and os.path.exists(stage2_pair_ho_path):
        print(
            f"Stage 2 files already exist. Skipping. {stage2_match_ho_path} {stage2_pair_ho_path}"
        )
        return

    subset_scene_data = scene_data["not_match_ho"]

    print("Begin generation")
    pqi = PairedQuestionInstantiator(train_ho=train_ho, alt_ho=alt_ho)
    generate_questions_obeying_filter(
        scene_data=subset_scene_data,
        output_questions_file=stage2_match_ho_path,
        template_instantiator=pqi,
        verbose=args.verbose,
        templates_per_image=1,  # NOTE: only 1/10th of data, so 1 instantiation. Stage 1 will have the other 9 templates.
        instances_per_template=1,
        other_rejection_condition=None,
        filter_option_filter=None,
        add_empty_filter_options=None,
    )

    print("Create paired .json file")
    # Load the Stage 2 .json & create a paired .json for the same questions
    # but with the paired images.
    with open(stage2_match_ho_path, "r") as infile:
        stage2_match_ho_questions = json.load(infile)

    # Copy & update the image indices to those of the paired images
    stage2_pair_ho_questions = copy.deepcopy(stage2_match_ho_questions)
    for i in range(len(stage2_pair_ho_questions["questions"])):
        q = stage2_pair_ho_questions["questions"][i]

        # Retrieve the paired program & text
        q_pair_prog, q_pair_text = pqi.paired_questions[
            (q["image_index"], q["question"])
        ]
        for f in q_pair_prog:
            if "side_inputs" in f:
                f["value_inputs"] = f["side_inputs"]
                del f["side_inputs"]
            else:
                f["value_inputs"] = []

        # Check same node types are present
        node_types = [x["type"] for x in q["program"]]
        paired_node_types = [x["type"] for x in q_pair_prog]
        assert node_types == paired_node_types

        # Overwrite program & text
        q["program"] = q_pair_prog
        q["question"] = q_pair_text

    # Save the pair file
    with open(stage2_pair_ho_path, "w") as outfile:
        json.dump(stage2_pair_ho_questions, outfile, indent=2, sort_keys=True)

    # Read in our creations & start validation
    with open(stage2_pair_ho_path, "r") as infile:
        stage2_pair_ho_questions = json.load(infile)
    with open(stage2_match_ho_path, "r") as infile:
        stage2_match_ho_questions = json.load(infile)

    assert len(stage2_pair_ho_questions["questions"]) == len(
        stage2_match_ho_questions["questions"]
    )

    print("Generation done. Validating results")
    with open(full_sg_path, "r") as infile:
        all_scenes = json.load(infile)["scenes"]
    for q_p, q_m in tqdm(
        zip(
            stage2_pair_ho_questions["questions"],
            stage2_match_ho_questions["questions"],
        ),
        total=len(stage2_pair_ho_questions["questions"]),
    ):
        p_answer = qeng.answer_final_question(q_p, all_scenes[q_p["image_index"]])
        m_answer = qeng.answer_final_question(q_m, all_scenes[q_m["image_index"]])
        assert p_answer == m_answer
        assert p_answer == q_p["answer"]
        assert m_answer == q_m["answer"]
        assert q_p["image_index"] == q_m["image_index"]
    print("All done.")


def run_stage1(args, scene_data, train_ho, OUT_DIR):
    # STAGE 1: 90% Generate normal questions, except not containing HO,
    #          on not_match_ho images

    split = "train" if "train" in args.split else "val_iid"

    print("Start Stage 1:")
    stage1_path = f"{OUT_DIR}/questions/{split}/tmp/CLEVR_held_out_questions_S1_{split}_{str(train_ho)}.json"

    if args.limit >= 0:
        stage1_path = f"{OUT_DIR}/questions/{split}/tmp/CLEVR_held_out_questions_S1_{split}_lim{args.limit}_{str(train_ho)}.json"

    if os.path.exists(stage1_path):
        print(f"Stage 1 files already exist. Skipping. {stage1_path}")
        return

    # Figure out which templates we can't use because we already used them in
    # stage 2. This is because Stages 1 & 2 share images.
    stage2_path = f"{OUT_DIR}/questions/{split}/tmp/CLEVR_held_out_questions_S2_match_ho_{split}_{str(train_ho)}.json"
    if args.limit >= 0:
        stage2_path = f"{OUT_DIR}/questions/{split}/tmp/CLEVR_held_out_questions_S2_match_ho_{split}_lim{args.limit}_{str(train_ho)}.json"

    assert os.path.exists(stage2_path)
    # Need a default dict as it's possible for a question not to have
    # a question generated for it.
    used_templates = defaultdict(lambda: list())
    with open(stage2_path, "r") as infile:
        stage2 = json.load(infile)["questions"]
        for s2_q in stage2:
            assert s2_q["image_index"] not in used_templates
            used_templates[s2_q["image_index"]] = [
                (s2_q["template_filename"], s2_q["question_family_index"])
            ]

    # Note: When excluding a given combination; we modify
    #       add_empty_filter_options so that it only produces options not
    #       matching HO, and thus we help maintain balance.
    option_filter = lambda x, y, z: exclude_ho(ho=train_ho, in_options=x)
    generate_questions_obeying_filter(
        scene_data=scene_data["not_match_ho"],
        output_questions_file=stage1_path,
        filter_option_filter=option_filter,
        templates_per_image=9,  # NOTE: only 9/10th of data, so 9 instantiations. Stage 2 has the other 1 template, which we block using prohibited_templates.
        # Only 9 as opposed to 10, b/c the 10th is produced in Stage 2
        add_empty_filter_options=NoMatchHoAddEmptyOptions(train_ho=train_ho),
        prohibited_templates=used_templates,
        verbose=args.verbose,
    )

    print("Stage 1 generation done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "split",
        choices=["test", "train_S3", "val_iid_S3", "train_S1_S2", "val_iid_S1_S2"],
        type=str,
        help="Which split to generate questions for. S3 signifies Stage 3 (same question with paired images), S1_S2 indicates Stages 1 & 2 (normal questions w/o HO, and paired questions with/without HO, all on images without HO)",
    )
    parser.add_argument(
        "--ho_idx",
        default=0,
        type=int,
        choices=[0, 1, 2, 3, 4, 5],
        help="0-5; which HO to generate questions for",
    )
    parser.add_argument(
        "--limit",
        default=-1,
        type=int,
        help="Limit the number of images to this value.",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(args)

    if args.split == "test":
        # (size, color, shape, material)
        HO_choices = [
            (None, None, "cylinder", "rubber"),
            (None, "cyan", None, "rubber"),
            ("large", None, None, "rubber"),
            (None, "cyan", "cylinder", None),
            ("large", None, "cylinder", None),
            ("large", "cyan", None, None),
        ]

        test_ho = HO_choices[args.ho_idx]

        split = "test"

        if not args.debug:
            os.makedirs(f"outputs/questions/{split}", exist_ok=True)

        scene_data = load_sg_subset(
            full_sg_path=f"outputs/scenes/CLEVR_held_out_{split}.json",
            objects_file_path=f"outputs/held_out_objects_{split}.json",
            glob_loc_to_img_path=f"outputs/scenes_to_render/glob_loc_index_to_img_index_{split}.json",
            target_ho=test_ho,
            family="match_ho",
        )  # "match_ho", "match_ho_pair", or "not_match_ho"

        if args.debug:
            # scene_data['scenes'] = [s for s in scene_data['scenes'] if s['image_index'] in [15004, 24563, 569, 16043, 964, 37088]] #+ scene_data['scenes'][1000:2000]
            scene_data["scenes"] = scene_data["scenes"][:500]

        output_test_path = (
            f"outputs/questions/{split}/CLEVR_held_out_questions_{split}_{str(test_ho)}.json"
            if not args.debug
            else "outputs_small_test/tmp.json"
        )
        if args.debug or not os.path.exists(output_test_path):
            generate_questions_obeying_filter(
                scene_data=scene_data,
                output_questions_file=output_test_path,
                filter_option_filter=None,
                template_instantiator=MatchHoInstantiator(test_ho=test_ho),
                verbose=args.verbose,
                templates_per_image=10 if split != "test" else 1,
                instances_per_template=1,
            )
        else:
            print(f"Skipping generation, path already exists: {output_test_path}")

    else:
        assert "train" in args.split or "val_iid" in args.split
        split = "train" if "train" in args.split else "val_iid"

        # (size, color, shape, material)
        HO_choices = [
            (None, None, "cylinder", "rubber"),
            (None, "cyan", None, "rubber"),
            ("large", None, None, "rubber"),
            (None, "cyan", "cylinder", None),
            ("large", None, "cylinder", None),
            ("large", "cyan", None, None),
        ]

        train_ho = HO_choices[args.ho_idx]

        OUT_DIR = "outputs"  #'outputs_small_test'
        # Load all the scene:
        #  - match_ho (images matching the held-out combination)
        #  - match_ho_pair (images paired with match_ho, modified to no longer match)
        #  - not_match_ho (images that do not match the held-out combination)
        print("Loading Dataset")
        scene_data = {}
        full_sg_path = f"{OUT_DIR}/scenes/CLEVR_held_out_{split}.json"
        for family in ["match_ho", "match_ho_pair", "not_match_ho"]:
            print(f"Loading {family}")
            scene_data[family] = load_sg_subset(
                full_sg_path=full_sg_path,
                objects_file_path=f"{OUT_DIR}/held_out_objects_{split}.json",
                glob_loc_to_img_path=f"{OUT_DIR}/scenes_to_render/glob_loc_index_to_img_index_{split}.json",
                target_ho=train_ho,
                family=family,
            )
            if args.limit >= 0:
                scene_data[family]["scenes"] = scene_data[family]["scenes"][
                    : args.limit
                ]

        # Note: Creating the possible exposures:
        #  - No visual or textual exposure: 1 + 2b + 3 (note: 3 uses match_ho_pair images)
        #  - Only textual exposure: 1 + 2a + 3 (note: 3 uses match_ho_pair images)
        #  - Only visual exposure: 1 + 2b + 3 (note: 3 uses match_ho images)

        # Note; yes, stage 1 & 2 are backwards.
        # STAGE 2: 10% Generate questions with HO (2a), paired with non-HO containing questions (2b)
        #          on not_match_ho images,
        #          s.t. answer is invariant to the change.
        if "S2" in args.split:
            run_stage2(args, scene_data, train_ho, OUT_DIR=OUT_DIR)

        if "S1" in args.split:
            print("Start Stage 1:")
            run_stage1(args, scene_data, train_ho, OUT_DIR=OUT_DIR)

        else:
            assert "S3" in args.split
            # STAGE 3: 100% normal questions, except not containing HO,
            #          on match_ho & match_ho_pair s.t. we are invariant to the image choice.
            # Templates that cannot be instantiated with HO, given that the image does
            # not contain HO
            # compare_integer.json is fine
            print("Start Stage 3")
            stage3_match_ho_path = f"{OUT_DIR}/questions/{split}/tmp/CLEVR_held_out_questions_S3_match_ho_{split}_{str(train_ho)}.json"
            stage3_pair_ho_path = f"{OUT_DIR}/questions/{split}/tmp/CLEVR_held_out_questions_S3_pair_ho_{split}_{str(train_ho)}.json"

            if args.limit >= 0:
                stage3_match_ho_path = f"{OUT_DIR}/questions/{split}/tmp/CLEVR_held_out_questions_S3_match_ho_{split}_lim{args.limit}_{str(train_ho)}.json"
                stage3_pair_ho_path = f"{OUT_DIR}/questions/{split}/tmp/CLEVR_held_out_questions_S3_pair_ho_{split}_lim{args.limit}_{str(train_ho)}.json"

            if os.path.exists(stage3_match_ho_path) and os.path.exists(
                stage3_pair_ho_path
            ):
                print(
                    f"Skip stage 3, files already exists {stage3_match_ho_path} and {stage3_pair_ho_path}"
                )
                exit(0)

            paired_img_finder = PairedImgFinder(
                ho=train_ho,
                gl_to_img_idx_path=f"{OUT_DIR}/scenes_to_render/glob_loc_index_to_img_index_{split}.json",
                objects_path=f"{OUT_DIR}/held_out_objects_{split}.json",
                sg_path=f"{OUT_DIR}/scenes/CLEVR_held_out_{split}.json",
                split=split,
            )

            changed_answer_rejector_obj = ImageInvariantRejector(
                paired_img_finder, verbose=True
            )

            def changed_answer_rejector(q, m, s, a, t):
                reject = changed_answer_rejector_obj(q, m, s, a, t)
                # Should never reject b/c we're using PairedImgInstantiator which should
                # already check this condition
                assert not reject
                return reject

            paired_img_instantiator = PairedImgInstantiator(paired_img_finder)
            generate_questions_obeying_filter(
                scene_data=scene_data["match_ho"],
                output_questions_file=stage3_match_ho_path,
                other_rejection_condition=changed_answer_rejector,
                template_instantiator=paired_img_instantiator,
                add_empty_filter_options=None,
                # Must be None, else PairedImgInstantiator can't work
                filter_option_filter=lambda x, y, z: exclude_ho(
                    ho=train_ho, in_options=x
                ),
                verbose=args.verbose,
            )

            # Load the Stage 3 .json & create a paired .json for the same questions
            # but with the paired images.
            with open(stage3_match_ho_path, "r") as infile:
                stage3_match_ho_questions = json.load(infile)

            # Copy & update the image indices to those of the paired images
            stage3_pair_ho_questions = copy.deepcopy(stage3_match_ho_questions)
            for i in range(len(stage3_pair_ho_questions["questions"])):
                q = stage3_pair_ho_questions["questions"][i]

                # Retrieve the paired program & correct the labels.
                # NOTE: paired program is identical, except for the cached _outputs
                # values (i.e., the intermediate evaluation values)
                q_pair_prog, q_pair_idx = paired_img_instantiator.paired_programs[
                    (q["image_index"], q["question"])
                ]
                for f in q_pair_prog:
                    if "side_inputs" in f:
                        f["value_inputs"] = f["side_inputs"]
                        del f["side_inputs"]
                    else:
                        f["value_inputs"] = []

                paired_filename = paired_img_finder.idx_to_paired_filename(
                    q["image_index"]
                )
                assert paired_filename.endswith(".png")
                q["image"] = paired_filename[:-4]
                q["image_filename"] = paired_filename
                q["image_index"] = paired_img_finder(q["image_index"])
                assert q["image_index"] == q_pair_idx

                # purge cached _output data which may no longer be correct
                for node in q["program"]:
                    node.pop("_output", None)

                # Check that if we purge all cached values, then the program matches
                # the paired program.
                tmp_pair = copy.deepcopy(q_pair_prog)
                for node in tmp_pair:
                    node.pop("_output", None)
                assert q["program"] == tmp_pair

                # overwrite program with the paired program (i.e., correct cached values)
                q["program"] = q_pair_prog

            with open(stage3_pair_ho_path, "w") as outfile:
                json.dump(stage3_pair_ho_questions, outfile, indent=2, sort_keys=True)

            with open(stage3_pair_ho_path, "r") as infile:
                stage3_pair_ho_questions = json.load(infile)
            with open(stage3_match_ho_path, "r") as infile:
                stage3_match_ho_questions = json.load(infile)

            assert len(stage3_pair_ho_questions["questions"]) == len(
                stage3_match_ho_questions["questions"]
            )

            print("Generation done. Validating results")
            with open(full_sg_path, "r") as infile:
                all_scenes = json.load(infile)["scenes"]
            for q_p, q_m in tqdm(
                zip(
                    stage3_pair_ho_questions["questions"],
                    stage3_match_ho_questions["questions"],
                ),
                total=len(stage3_pair_ho_questions["questions"]),
            ):
                p_answer = qeng.answer_final_question(
                    q_p, all_scenes[q_p["image_index"]]
                )
                m_answer = qeng.answer_final_question(
                    q_m, all_scenes[q_m["image_index"]]
                )
                assert p_answer == m_answer
                assert p_answer == q_p["answer"]
                assert m_answer == q_m["answer"]
            print("All done.")


if __name__ == "__main__":
    main()
