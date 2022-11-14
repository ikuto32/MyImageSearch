import shutil
import tqdm


def copyImages(setting, args):
    print("CopyImages")
    out_dir = setting["out_dir"]
    image_dir = setting["image_dir"]

    select_image_paths = args.get("metaNames").split(",")
    for select_image_path in tqdm.tqdm(select_image_paths):
        try:
            shutil.copy(f"{image_dir}/{select_image_path}", out_dir)
        except:
            pass
    return None