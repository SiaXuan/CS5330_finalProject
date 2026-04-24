# download_images.py
import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def download_image(asin, url, save_dir):
    save_path = os.path.join(save_dir, f"{asin}.jpg")
    if os.path.exists(save_path):
        return True, asin
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True, asin
        return False, asin
    except:
        return False, asin

def download_category(txt_path, save_dir, max_workers=16):
    os.makedirs(save_dir, exist_ok=True)
    pairs = []
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                pairs.append((parts[0], parts[1]))

    failed = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_image, asin, url, save_dir): asin 
                   for asin, url in pairs}
        for future in tqdm(as_completed(futures), total=len(pairs)):
            success, asin = future.result()
            if not success:
                failed.append(asin)

    print(f"Done. Failed: {len(failed)}/{len(pairs)}")

# download dress only for testing
download_category(
    "fashion-iq-metadata/image_url/asin2url.dress.txt",
    "images/dress"
)