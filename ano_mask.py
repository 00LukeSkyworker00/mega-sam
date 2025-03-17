import numpy as np
import cv2
import os
import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Process some images.")

# Add arguments
parser.add_argument("--dir", type=str, help="Path to the directory containing the images.")

# Parse arguments
args = parser.parse_args()

# dir = '/home/skyworker/data/sets/movie_a/train/movi_a_0005'
input_folder = os.path.join(args.dir, 'ano')
output_folder = os.path.join(args.dir, 'masks', 'seq1')
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    img = cv2.imread(os.path.join(input_folder, filename))
    mask = np.any(img != 0, axis=2).astype(np.uint8) * 255
    mask = cv2.GaussianBlur(mask, (9, 9), 0)
    # mask = cv2.dilate(mask, np.ones((4, 4), np.uint8), iterations=1)
    filename = filename.replace("ano", "rgb").replace(".png", ".jpg")
    print(os.path.join(output_folder, filename))
    cv2.imwrite(os.path.join(output_folder, filename), mask)