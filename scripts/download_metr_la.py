import os
import shutil

import kagglehub


def main() -> None:
    dataset = "annnnguyen/metr-la-dataset"
    dest_dir = os.path.join("data", "raw")
    os.makedirs(dest_dir, exist_ok=True)

    path = kagglehub.dataset_download(dataset)
    print("Downloaded to:", path)

    copied = 0
    for name in os.listdir(path):
        src = os.path.join(path, name)
        if not os.path.isfile(src):
            continue
        if not name.lower().endswith((".h5", ".pkl", ".csv")):
            continue
        dst = os.path.join(dest_dir, name)
        shutil.copy2(src, dst)
        copied += 1
        print("Copied:", dst)

    if copied == 0:
        print("No dataset files copied. Check the download directory.")


if __name__ == "__main__":
    main()
