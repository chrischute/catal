"""Artificially add whiteboards to images."""

import argparse
import os
import random

from PIL import Image


def main(args):
    num_generated = 0
    wb_img = Image.open(args.wb_example, 'r').convert('RGB')
    for base_path, _, file_names in os.walk(args.negative_dir):
        for f in file_names:
            img_path = os.path.join(base_path, f)
            try:
                aug_img = img_with_whiteboard(img_path, wb_img)
            except OSError:
                os.unlink(img_path)
                continue
            num_generated += 1
            aug_img.save(os.path.join(args.positive_dir, 'pos_{:05d}.jpg'.format(num_generated)))

            if num_generated % 10 == 0:
                print('Generated {}...'.format(num_generated))

    print('Created {} whiteboard-positive examples'.format(num_generated))


def img_with_whiteboard(src_path, wb_img):
    """Get image with whiteboard artificially added."""
    bg_img = Image.open(src_path, 'r').convert('RGB')
    bg_w, bg_h = bg_img.size

    wb_w, wb_h = wb_img.size
    if wb_w > bg_w - 50 or wb_h > bg_h - 50:
        wb_img = wb_img.resize([bg_w // 4, bg_h // 4])
        wb_w, wb_h = wb_img.size
    offset = (random.randint(0, bg_w - wb_w), random.randint(0, bg_h - wb_h))
    bg_img.paste(wb_img, offset)

    return bg_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate images with whiteboards')
    parser.add_argument('--negative_dir', type=str, default='data/neg_wb')
    parser.add_argument('--positive_dir', type=str, default='data/pos_wb')
    parser.add_argument('--wb_example', type=str, default='data/wb_example.png')

    main(parser.parse_args())
