"""
Reformat the .jsons in the local directory to be cleanly formatted.
"""

import json
import os

if __name__ == "__main__":
    for file in os.listdir("."):
        if file.endswith(".json"):
            with open(file, "r") as infile:
                contents = json.load(infile)
            with open(file, "w") as outfile:
                json.dump(contents, outfile, indent=2, sort_keys=True)
