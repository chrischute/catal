"""Artificially add whiteboards to images."""

import argparse
import os
import numpy as np

from PIL import Image, ImageFile

IMAGENET_SIZE = 224, 224


def main(args):

    # Get channel-wise means from each photo
    all_means = []
    all_stds = []
    i = 0
    for base_path, _, file_names in os.walk(args.input_dir):
        for file_name in file_names:
            if not file_name.endswith('.jpg'):
                continue

            src_path = os.path.join(base_path, file_name)
            dst_path = os.path.join(base_path.replace('val', 'val_pp'), file_name)
            means, stds = pre_process(src_path, dst_path)
            all_means.append(means)
            all_stds.append(stds)
            i += 1
            if i % 10 == 0:
                print('{} / 400 = {:.2f}%'.format(i, 100. * i / 400))

    # Compute mean and standard deviation for each channel
    means = []
    stds = []
    for i in range(3):
        means.append(sum(m[i] for m in all_means) / len(all_means))
        stds.append(sum([s[i] for s in all_stds]) / len(all_stds))

    print(means)
    print(stds)


def pre_process(src_path, dst_path):
    """Pre-process an image at src_path and save to dst_path. Get the mean and standard deviation."""
    print('Opening {}'.format(src_path))

    img = Image.open(src_path, 'r').convert('RGB')
    img = img.resize(IMAGENET_SIZE)

    img_np = np.array(img, dtype=np.float)
    means = list(np.mean(img_np, axis=tuple(range(img_np.ndim-1))).tolist())
    stds = list(np.std(img_np, axis=tuple(range(img_np.ndim-1))).tolist())

    img.save(dst_path)

    return means, stds


if __name__ == '__main__':
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    parser = argparse.ArgumentParser('Preprocess whiteboard images')
    parser.add_argument('--input_dir', type=str, default='data/whiteboard_pilot/val')
    parser.add_argument('--output_dir', type=str, default='data/whiteboard_pilot/val_pp')

    main(parser.parse_args())
