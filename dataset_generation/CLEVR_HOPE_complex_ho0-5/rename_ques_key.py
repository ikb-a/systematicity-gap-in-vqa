"""
Just running https://github.com/facebookresearch/clevr-dataset-gen/issues/14#issuecomment-484300688
on all and also fixing the answer names.
"""
import json
import argparse


def fixit(input_questions_file):
    # Load questions
    with open(input_questions_file, "r") as f:
        question_data = json.load(f)
        info = question_data["info"]
        questions = question_data["questions"]
    print("Read %d questions from disk" % len(questions))
    # Rename 'type' to 'program'
    for q in questions:
        # Undocumented fix to answers.
        if isinstance(q["answer"], bool):
            q["answer"] = "yes" if q["answer"] else "no"
        elif isinstance(q["answer"], int):
            assert 0 <= q["answer"] <= 10
            q["answer"] = str(q["answer"])
        else:
            assert isinstance(q["answer"], str)

        # Fix the program
        programs = q["program"]
        for p in programs:
            # Fix type -> function
            if "type" in p:
                assert "function" not in p
                p["function"] = p.pop("type")
            else:
                assert "type" not in p and "function" in p

    # Dump new dict
    with open(input_questions_file, "w") as f:
        print("Writing output to %s" % input_questions_file)
        json.dump(
            {
                "info": info,
                "questions": questions,
            },
            f,
            sort_keys=True,
        )


HO_choices = [
    (None, None, "cylinder", "rubber"),
    (None, "cyan", None, "rubber"),
    ("large", None, None, "rubber"),
    (None, "cyan", "cylinder", None),
    ("large", None, "cylinder", None),
    ("large", "cyan", None, None),
]

OUT_DIR = "outputs"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "ho_idx",
        type=int,
        choices=[0, 1, 2, 3, 4, 5],
        help="0-5; which HO to fix the questions for",
    )
    args = parser.parse_args()

    print(args)

    # for ho in HO_choices:
    ho = HO_choices[args.ho_idx]
    print(f"Fix test {ho}")
    fixit(f"{OUT_DIR}/questions/test/CLEVR_held_out_questions_test_{str(ho)}.json")
    for test_name in [
        "no_text_no_vis",
        "zero_shot_text",
        "exp_text_no_vis",
        "no_text_exp_vis",
        "exp_text_exp_vis",
    ]:
        print(f"Fixing {test_name} {ho}")
        fixit(
            f"{OUT_DIR}/questions/train/CLEVR_ho_train_{str(ho)}_{test_name}_questions.json"
        )
        fixit(
            f"{OUT_DIR}/questions/val_iid/CLEVR_ho_val_iid_{str(ho)}_{test_name}_questions.json"
        )
