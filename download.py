import urllib.request
import os
'''
Kodak Lossless True Color Image Suite
'''
def download_kodak():
    save_dir = 'data/raw'
    os.makedirs(save_dir, exist_ok=True)
    
    # Kodak set
    base_url = "https://r0k.us/graphics/kodak/kodak/kodim{:02d}.png"
    
    print("Starting download of Kodak dataset (24 images)...")
    for i in range(1, 25):
        file_name = f"kodim{i:02d}.png"
        url = base_url.format(i)
        save_path = os.path.join(save_dir, file_name)
        
        if not os.path.exists(save_path):
            try:
                print(f"Downloading {file_name}...")
                urllib.request.urlretrieve(url, save_path)
            except Exception as e:
                print(f"Failed to download {file_name}: {e}")
        else:
            print(f"{file_name} already exists, skipping.")
    print("Download complete!")

if __name__ == "__main__":
    download_kodak()