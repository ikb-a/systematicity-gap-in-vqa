"""
Script that processes these .json files and creates new ones with
the programs re-ordered. These templates should be completely equivalent
(we've just choosing a slightly different linearization of the program tree).
This is useful because the question generation performs DFS in the same order
as the linearization; when we have a constraint, we want that constraint to
occur as close to the root as possible.
"""
from __future__ import annotations

import copy
import os
import json
from typing import Optional, List, Tuple


class Node:
    def __init__(self, left: Optional[Node], right: Optional[Node], node_dict: dict):
        self.left = left
        self.right = right
        self.values = copy.deepcopy(node_dict)
        self.values.pop("inputs")

        if self.left is None and self.right is not None:
            raise ValueError(
                "Must fill left first; cannot have non-None right and None left"
            )

    def __repr__(self):
        return self.values["type"]

    def as_root_node(self):
        assert self.left is None
        assert self.right is None
        tmp = copy.deepcopy(self.values)
        tmp["inputs"] = []
        return [tmp]

    def left_linearize(self, start_idx=0):
        if self.left is None and self.right is None:
            return self.as_root_node()
        elif self.right is None:
            assert (
                self.left is not None
            )  # By __init__; we cannot have None left & non-None right
            left_program = self.left.left_linearize(start_idx=start_idx)
            tmp = copy.deepcopy(self.values)
            tmp["inputs"] = [len(left_program) - 1 + start_idx]
            return left_program + [tmp]
        else:
            assert self.left is not None and self.right is not None
            left_program = self.left.left_linearize(start_idx=start_idx)
            right_program = self.right.left_linearize(
                start_idx=start_idx + len(left_program)
            )

            tmp = copy.deepcopy(self.values)
            tmp["inputs"] = [
                len(left_program) - 1 + start_idx,
                len(left_program) + len(right_program) - 1 + start_idx,
            ]

            return left_program + right_program + [tmp]

    def right_linearize(self, start_idx=0):
        if self.left is None and self.right is None:
            return self.as_root_node()
        elif self.right is None:
            assert (
                self.left is not None
            )  # By __init__; we cannot have None left & non-None right
            left_program = self.left.right_linearize(start_idx=start_idx)
            tmp = copy.deepcopy(self.values)
            tmp["inputs"] = [len(left_program) - 1 + start_idx]
            return left_program + [tmp]
        else:
            assert self.left is not None and self.right is not None
            right_program = self.right.right_linearize(start_idx=start_idx)
            left_program = self.left.right_linearize(
                start_idx=start_idx + len(right_program)
            )

            tmp = copy.deepcopy(self.values)
            # Note: order must also be reversed to make sure that order sensitive
            #       operations work (i.e., integer comparison) properly
            tmp["inputs"] = [
                len(right_program) + len(left_program) - 1 + start_idx,
                len(right_program) - 1 + start_idx,
            ]

            return right_program + left_program + [tmp]


def program_to_tree(program: List[dict]) -> Node:
    stack = {}
    for i, node in enumerate(program):
        if len(node["inputs"]) == 0:
            stack[i] = Node(left=None, right=None, node_dict=node)
        elif len(node["inputs"]) == 1:
            expected_idx = node["inputs"][0]
            prev = stack.pop(expected_idx)
            stack[i] = Node(left=prev, right=None, node_dict=node)
        else:
            assert len(node["inputs"]) == 2
            first = stack.pop(node["inputs"][0])
            second = stack.pop(node["inputs"][1])
            stack[i] = Node(left=first, right=second, node_dict=node)

    assert len(stack) == 1
    return list(stack.values())[0]


def reorder_program(
    program: list[dict], r_linearize: bool = True
) -> Tuple[list[dict], List[int], List[int]]:
    """
    Convert program into re-ordered program, list that converts old node indices to
    new node indices, and list that converts old opportunity indices into new
    opportunity indices

    :param program: Program dict -- i.e., template['nodes'] for a template
    :return:
    """

    program = copy.deepcopy(program)

    # Update the nodes to record their original indices, and the original
    # ordering of the opportunities in which to specify all 4 attributes.
    opp_count = 0
    for i, node in enumerate(program):
        node["ori_idx"] = i
        if "side_inputs" in node and len(node["side_inputs"]) >= 4:
            node["opportunity"] = opp_count
            opp_count += 1

    # Convert to tree
    tree = program_to_tree(program)
    # Linearize as requested
    # Note that program_to_tree always puts the first argument on the left.
    # So 2 right linearizations will do nothing.
    if r_linearize:
        r_program = tree.right_linearize()
    else:
        r_program = tree.left_linearize()

    index_mapping = [-1] * len(program)  # Map old node index to new node index
    opp_mapping = [-1] * opp_count  # Map old opportunity index to new opportunity index
    # Opportunity as places where we can specify all 4 attributes; ignoring all other atoms

    new_opp_count = 0
    for i, node in enumerate(r_program):
        index_mapping[node["ori_idx"]] = i
        if "opportunity" in node:
            opp_mapping[node["opportunity"]] = new_opp_count
            new_opp_count += 1

    assert not any(x == -1 for x in opp_mapping)
    assert not any(x == -1 for x in index_mapping)

    # purge the added information
    for node in r_program:
        node.pop("ori_idx")
        node.pop("opportunity", None)

    return r_program, index_mapping, opp_mapping


def main(template_dir="."):
    new_templates = {}  # filename -> List[template]

    templates = {}  # (filename, index) -> template (as a dict)
    for fn in os.listdir(template_dir):
        if not fn.endswith(".json"):
            continue
        print(fn)
        with open(os.path.join(template_dir, fn), "r") as f:
            list_of_templates = json.load(f)
            new_templates[fn] = [None] * len(list_of_templates)

            for i, template in enumerate(list_of_templates):
                key = (fn, i)
                templates[key] = template

    for key in templates:
        print(key)
        template = templates[key]
        all_branch_nodes = [x for x in template["nodes"] if len(x["inputs"]) >= 2]
        unknown_branches = [
            x
            for x in all_branch_nodes
            if x["type"]
            not in [
                "equal_size",
                "equal_color",
                "equal_material",
                "equal_shape",
                "union",
                "intersect",
                "equal_integer",
                "less_than",
                "greater_than",
            ]
        ]
        if len(unknown_branches) > 0:
            print("Unknown branching atom found; aborting")
            print(unknown_branches)
            exit(-1)

        # Make sure every template has at most one branching atom
        assert len(all_branch_nodes) <= 1

    # Check the left-linearization just reconstructs the original program
    print("==================================================")
    for key in templates:
        print(key)
        template = templates[key]
        program = template["nodes"]
        tree = program_to_tree(program)
        reconstructed_program = tree.left_linearize()

        # Check our reconstruction matches the original
        # (there are a couple exceptions where the template uses weirdly
        # linearized programs; these can be safely ignored)
        if key not in [
            ("single_or.json", 5),
            ("comparison.json", 0),
            ("comparison.json", 1),
        ]:
            if program != reconstructed_program:
                print(program)
                print(reconstructed_program)
            assert program == reconstructed_program

        # Check that right, then left linearization, should undo things
        r_program, l_to_r_idx, l_to_r_opp = reorder_program(reconstructed_program)
        double_swapped_program, r_to_l_idx, r_to_l_opp = reorder_program(
            r_program, r_linearize=False
        )
        # Two swaps take us back to the same program
        assert reconstructed_program == double_swapped_program

        # Two swaps give us index mapping that undo each other
        assert len(l_to_r_idx) == len(r_to_l_idx)
        for i in range(len(l_to_r_idx)):
            assert r_to_l_idx[l_to_r_idx[i]] == i
            assert l_to_r_idx[r_to_l_idx[i]] == i
        assert len(l_to_r_opp) == len(r_to_l_opp)
        for i in range(len(l_to_r_opp)):
            assert r_to_l_opp[l_to_r_opp[i]] == i
            assert l_to_r_opp[r_to_l_opp[i]] == i

        # Check that double right linearization does nothing
        tmp_r_program, tmp_l_to_r_idx, tmp_l_to_r_opp = reorder_program(r_program)
        assert tmp_r_program == r_program
        assert tmp_l_to_r_idx == list(range(len(tmp_l_to_r_idx)))
        assert tmp_l_to_r_opp == list(range(len(tmp_l_to_r_opp)))

    # Perform the actual conversion
    os.makedirs("reordered_templates", exist_ok=True)

    for key in templates:
        template = copy.deepcopy(templates[key])
        program = template["nodes"]

        r_program, index_map, opp_map = reorder_program(program)

        # Update the program & add the opportunity mapping
        template["nodes"] = r_program
        template["opp_mapping"] = opp_map

        # Update the indices in any constraints
        for constraint in template["constraints"]:
            assert constraint["type"] in ["OUT_NEQ", "NULL"]
            if constraint["type"] == "OUT_NEQ":
                constraint["params"] = [index_map[x] for x in constraint["params"]]

        new_templates[key[0]][key[1]] = template

    for fn in new_templates:
        assert all(x is not None for x in new_templates[fn])
        with open(os.path.join("reordered_templates", fn), "w") as outfile:
            json.dump(new_templates[fn], outfile, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
