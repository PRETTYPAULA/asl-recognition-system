import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')

for d in sorted(os.listdir(DATASET_DIR)):
    path = os.path.join(DATASET_DIR, d)
    if os.path.isdir(path):
        files = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"{d}: {len(files)} images")

