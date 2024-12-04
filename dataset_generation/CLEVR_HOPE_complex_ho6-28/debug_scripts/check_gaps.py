import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "split", help="what split to check", type=str, choices=["train", "test", "val_iid"]
)
args = parser.parse_args()

scenes = sorted(list(int(x[-11:-5]) for x in os.listdir(f"output/scenes/{args.split}")))
images = sorted(list(int(x[-10:-4]) for x in os.listdir(f"output/images/{args.split}")))
# Confirm corresponding  image exists
scenes_set = set(scenes)
images_set = set(images)
print("scenes without images: -- should be []")
print([x for x in scenes if x not in images_set])
print("images without scenes: -- should be []")
print([y for y in images if y not in scenes_set])

index = 0
missing = []
for present_idx in scenes:
    if present_idx == index:
        index += 1
    else:
        missing.append(f"{index} to {present_idx - 1}")
        index = present_idx + 1

print("missing scene range -- should be []")
print(missing)

index = 0
missing = []
for present_idx in images:
    if present_idx == index:
        index += 1
    else:
        missing.append(f"{index} to {present_idx - 1}")
        index = present_idx + 1

print("missing image range -- should be []")
print(missing)
