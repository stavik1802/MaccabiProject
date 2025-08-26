import os

def delete_csv_files(root_folder):
    # Walk through the folder tree
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename in ("subs.csv", "poss.csv"):
                file_path = os.path.join(dirpath, filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

if __name__ == "__main__":
    # Change this path to the main folder you want to clean
    folder_path = r"games_eval_new"
    delete_csv_files(folder_path)
