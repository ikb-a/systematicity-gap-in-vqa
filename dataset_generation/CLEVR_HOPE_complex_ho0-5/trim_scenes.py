import json
import os

root = "outputs/scenes"

if __name__ == "__main__":
    for file in os.listdir(root):
        if file.endswith(".json"):
            with open(os.path.join(root, file), "r") as infile:
                contents = json.load(infile)

            if "info" not in contents or "scenes" not in contents:
                print(f"Skip {file}")
                continue
            else:
                trimmed = {}
                trimmed["info"] = contents["info"]
                trimmed["scenes"] = []
                for s in contents["scenes"]:
                    trimmed_s = {}

                    # Copy only what we need
                    # Original CLEVR keys: ["split", "image_index", "objects", "relationships", "image_filename", "directions"]
                    # We omit the split & relationships.
                    # Need directions & 3d coords for symbolic execution.
                    # Note! relationships is needed for the question engine; but not for
                    # the train-time symbolic executor. Therefore, it is omitted.
                    for key in [
                        "image_index",
                        "objects",
                        "directions",
                        "image_filename",
                    ]:
                        trimmed_s[key] = s[key]

                    # None of these are required for symbolic execution; remove them
                    for obj in trimmed_s["objects"]:
                        obj.pop("rotation", None)
                        obj.pop("pixel_coords", None)

                    trimmed["scenes"].append(trimmed_s)

                with open(os.path.join(root, file), "w") as outfile:
                    json.dump(trimmed, outfile, sort_keys=True)
