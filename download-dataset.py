import os
import zipfile
import requests
from tqdm import tqdm

def download_with_progress(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024
    with open(filename, "wb") as file, tqdm(
        desc=filename,
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            bar.update(len(data))
            file.write(data)


# I used my ftp server to host the dataset, just makes it easier
dataset_link = "https://ftp.cameronwsmith.com/pub/other/handwriting-dataset.zip"
local_file = "./handwriting-dataset.zip"


def main():
    if os.path.exists("./handwriting-dataset"):
        print("Dataset already downloaded")
        return

    print("Downloading dataset... (this may take a while)")
    download_with_progress(dataset_link, local_file)

    print("Done downloading, extracting dataset...")
    with zipfile.ZipFile(local_file, "r") as zip_ref:
        zip_ref.extractall("./")

    print("Dataset extracted")
    os.remove(local_file)


if __name__ == "__main__":
    main()