import os
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
TARGET_SIZE = (224, 224)


def resize_image(args):
    img_path, target_size = args
    try:
        img = cv2.imread(img_path)
        if img is None:
            return f"SKIP (unreadable): {img_path}"
        resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(img_path, resized)
        return f"OK: {img_path}"
    except Exception as e:
        return f"ERROR: {img_path} -> {e}"


def main():
    tasks = []
    total = 0
    for label_dir in sorted(os.listdir(DATASET_DIR)):
        label_path = os.path.join(DATASET_DIR, label_dir)
        if not os.path.isdir(label_path):
            continue
        for img_file in os.listdir(label_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(label_path, img_file)
                tasks.append((img_path, TARGET_SIZE))
                total += 1

    print(f"Resizing {total} images to {TARGET_SIZE[0]}x{TARGET_SIZE[1]}...")

    processed = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(resize_image, task): task for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            processed += 1
            if processed % 5000 == 0:
                print(f"  ... {processed}/{total} done")

    print(f"Done! Resized {processed} images.")


if __name__ == '__main__':
    main()

