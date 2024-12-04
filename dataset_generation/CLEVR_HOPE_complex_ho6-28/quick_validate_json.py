import json
from held_out_add_utils_generate_sg import validate_scenegraph

# blender --background --python quick_validate_json.py

if __name__ == "__main__":
    with open("crashed_CLEVR_held_out_train_043425.json", "r") as infile:
        sg = json.load(infile)
    result = validate_scenegraph(sg)
    print(result)
