"""Download photos using metadata from a CSV file."""
import argparse
import os
import pandas as pd
import pickle
import urllib.request

from catal import CatalPhoto
from PIL import Image
from tqdm import tqdm

IMAGENET_SIZE = 224, 224


def main(args):
    if args.pkl_path:
        with open(args.pkl_path, 'rb') as pkl_fh:
            examples = pickle.load(pkl_fh)
        os.makedirs(os.path.join(args.output_dir, 'unlabeled'), exist_ok=True)
    else:
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
            if example.is_labeled:
                subdir_name = 'wb_{}'.format('pos' if example.has_whiteboard else 'neg')
            else:
                subdir_name = 'unlabeled'
            file_name = '{}.jpg'.format(example.record_id)
            dst_path = os.path.join(args.output_dir, subdir_name, file_name)
            urllib.request.urlretrieve(example.url, dst_path)
            down_sample_image(dst_path, dst_path)
        except Exception as e:
            print('Error downloading from {}: {}'.format(example.url, e))


def down_sample_image(src_path, dst_path):
    img = Image.open(src_path, 'r').convert('RGB')
    img = img.resize(IMAGENET_SIZE)
    img.save(dst_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download photos from CSV file of URLs')

    parser.add_argument('--csv_path', default='data/wb_500sample.csv')
    parser.add_argument('--pkl_path', default='/data/catal/wb_14k.pkl')
    parser.add_argument('--output_dir', default='/data/catal/wb500')

    main(parser.parse_args())
