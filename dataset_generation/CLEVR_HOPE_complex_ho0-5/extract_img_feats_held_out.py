from vqa_framework.data_modules.generic_clevr_loader import (
    GenericCLEVRDataModule,
    ClevrFeature,
)
import argparse
from vislang.models.lxmert.feat_extraction.generic_clevr_frcnn_extr import (
    main as extract_frcnn,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "split",
        choices=["train", "test", "val_iid"],
        type=str,
        help="Which split to extract image features for",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--only_resnet",
        action="store_true",
        help="Only extract the larger ResNet features, skip the rest",
    )
    parser.add_argument(
        "--exclude_resnet",
        action="store_true",
        help="Skip the larger ResNet features, extract the rest",
    )
    args = parser.parse_args()

    generic_split_name = "val" if args.split == "val_iid" else args.split

    # Produce hdf5 for raw images & RESNET features
    for features in [ClevrFeature.IMAGES, ClevrFeature.IEP_RESNET]:
        if args.only_resnet and features == ClevrFeature.IMAGES:
            continue
        elif args.exclude_resnet and features == ClevrFeature.IEP_RESNET:
            continue

        print(f"Producing {args.split} hdf5 file for {features}")

        dm_kwargs = {
            "data_dir": "outputs/hdf5_held_out_CLEVR",
            "dm_images_name": f"{args.split}_v1.0_img",
            "dm_questions_name": f"{args.split}_v1.0_ques",
            "dm_train_images": None,
            "dm_val_images": None,
            "dm_test_images": None,
            "loader_num_workers": 0,
            "image_features": features,
        }

        dm_kwargs[f"dm_{generic_split_name}_images"] = f"outputs/images/{args.split}"

        dm = GenericCLEVRDataModule(**dm_kwargs)
        # Actually perform the hdf5 creation
        dm.prepare_data()

    if not args.only_resnet:
        print(f"Extracting Faster RCNN features")
        extract_frcnn(
            image_hdf5_path=f"outputs/hdf5_held_out_CLEVR/{args.split}_v1.0_img/{generic_split_name}_ims.h5",
            outfile_path=f"outputs/hdf5_held_out_CLEVR/{args.split}_v1.0_img/user_{generic_split_name}_vg_frcnn.h5",
        )
