"""Download photos using metadata from a CSV file."""
import argparse
import os
import pandas as pd
import urllib.request

from catal import CatalPhoto
from PIL import Image
from tqdm import tqdm

IMAGENET_SIZE = 224, 224


def main(args):
    df = pd.read_csv(args.csv_path)
    examples = [CatalPhoto(url=str(row[0]), annotation=str(row[1])) for _, row in df.iterrows()]

    print('Positives: {}'.format(sum(1 for e in examples if e.has_whiteboard is True)))
    print('Negatives: {}'.format(sum(1 for e in examples if e.has_whiteboard is False)))

    # Make directories for holding photos
    for dir_name in ('wb_pos', 'wb_neg'):
        os.makedirs(os.path.join(args.output_dir, dir_name), exist_ok=True)

    # Download photos
    for example in tqdm(examples):
        try:
            subdir_name = 'wb_{}'.format('pos' if example.has_whiteboard else 'neg')
            file_name = '{}.jpg'.format(example.photo_num)
            dst_path = os.path.join(args.output_dir, subdir_name, file_name)
            urllib.request.urlretrieve(example.url, dst_path)
            down_sample_image(dst_path, dst_path)
        except Exception:
            print('Error downloading from {}'.format(example.url))


def down_sample_image(src_path, dst_path):
    img = Image.open(src_path, 'r').convert('RGB')
    img = img.resize(IMAGENET_SIZE)
    img.save(dst_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download photos from CSV file of URLs')

    parser.add_argument('--csv_path', default='data/wb_500sample.csv')
    parser.add_argument('--output_dir', default='/data/catal/wb500')

    main(parser.parse_args())
