import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import random

def load_image(image_path):
    """Đọc ảnh từ đường dẫn, trả về (img_bgr, img_rgb)."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_bgr, img_rgb

def show_images(images, titles=None, cols=3, figsize=(12, 6)):
    n = len(images)
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=figsize)
    for i, img in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        if img.ndim == 2:
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(img)
        if titles is not None and i < len(titles):
            plt.title(titles[i])
        plt.axis("off")
    plt.tight_layout()
    plt.show()

def preprocess_image(img_bgr, target_size=(256, 256)):
    """
    Tiền xử lý cơ bản:
    - Resize về kích thước chuẩn
    - Làm mờ Gaussian nhẹ để giảm nhiễu
    Trả về: img_bgr_resized, img_rgb_resized
    """
    img_resized = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_AREA)
    img_blur = cv2.GaussianBlur(img_resized, (3, 3), 0)
    img_rgb = cv2.cvtColor(img_blur, cv2.COLOR_BGR2RGB)
    return img_blur, img_rgb

def get_random_file(folder_path):
    """
    Selects a random file from the specified folder.

    Args:
        folder_path (str): The path to the directory.

    Returns:
        str: The full path to the randomly selected file, or None if the 
             folder is empty or an error occurs.
    """
    try:
        # Get a list of all entries (files and directories) in the folder
        entries = os.listdir(folder_path)
        # Filter the list to include only actual files (not subdirectories)
        files = [entry for entry in entries if os.path.isfile(os.path.join(folder_path, entry))]
        if not files:
            print("No files found in the specified folder.")
            return None
        # Choose a random file name from the list
        random_file_name = random.choice(files)
        # Join the folder path and file name to get the full path
        random_file_path = os.path.join(folder_path, random_file_name)
        return random_file_path
    except FileNotFoundError:
        print(f"Error: Folder not found at '{folder_path}'")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

