#!/usr/bin/env python3
import os
import shutil
import argparse

def copy_matching_folders(src_root, dst_root, overwrite=False):
    """
    Copy files from src_root to dst_root where subfolders have the same name.

    Args:
        src_root (str): Path to the first root folder (source).
        dst_root (str): Path to the second root folder (destination).
        overwrite (bool): If True, overwrite existing files in dst.
    """
    for folder_name in os.listdir(src_root):
        src_path = os.path.join(src_root, folder_name)
        dst_path = os.path.join(dst_root, folder_name)

        if os.path.isdir(src_path) and os.path.isdir(dst_path):
            print(f"📂 Matching folder: {folder_name}")

            for file_name in os.listdir(src_path):
                src_file = os.path.join(src_path, file_name)
                dst_file = os.path.join(dst_path, file_name)

                if os.path.isfile(src_file):
                    if os.path.exists(dst_file) and not overwrite:
                        print(f"⚠️ Skipping existing file: {dst_file}")
                    else:
                        shutil.copy2(src_file, dst_file)
                        print(f"✅ Copied {src_file} → {dst_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy files between matching subfolders.")
    parser.add_argument("src", help="Source root folder")
    parser.add_argument("dst", help="Destination root folder")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")

    args = parser.parse_args()
    copy_matching_folders(args.src, args.dst, args.overwrite)
