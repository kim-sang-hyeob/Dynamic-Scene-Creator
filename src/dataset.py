import os
import sys
import zipfile
import urllib.request
import shutil

class DatasetManager:
    def __init__(self, config):
        self.config = config
        self.data_root = config['paths']['data_root']
        # Hardcoded URL for this quick start, ideally move to config for full flexibility
        self.tandt_url = "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip"

    def download(self, dataset_name):
        print(f"Dataset Manager: Processing '{dataset_name}'")
        
        if dataset_name == 'tandt':
            self._download_tandt()
        else:
            print(f"[Error] Unknown dataset: {dataset_name}")
            print("Available datasets: tandt")

    def _download_tandt(self):
        dest_dir = self.data_root
        zip_path = os.path.join(dest_dir, "tandt_db.zip")
        
        os.makedirs(dest_dir, exist_ok=True)
        
        if not os.path.exists(zip_path):
            print(f"Downloading from {self.tandt_url}")
            urllib.request.urlretrieve(self.tandt_url, zip_path, self._progress_hook)
            print("\nDownload complete.")
        else:
            print("Zip file exists. Skipping download.")

        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(dest_dir)
        print(f"Extracted to {dest_dir}")

    def _progress_hook(self, block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = downloaded * 100 / total_size
            sys.stdout.write(f"\rDownloading... {percent:.1f}%")
            sys.stdout.flush()
