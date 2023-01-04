import shutil
import tqdm


def copyImages(itemlist, args):
    print("CopyImages")
    out_dir = itemlist.outDir
    image_dir = itemlist.metadataDir

    select_image_paths = args.get("meta_names").split(",")
    for select_image_path in tqdm.tqdm(select_image_paths):
        try:
            shutil.copy(f"{image_dir}/{select_image_path}", out_dir)
        except:
            pass
    return None