"""Download photos using metadata from a CSV file."""
import argparse
import os
import pandas as pd
import pickle
import requests
import shutil
import util

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
        examples = [CatalPhoto(url=str(row['netpublish_URL']), annotation=None) for _, row in df.iterrows()]

        # Make directories for holding photos
        for dir_name in ('wb_pos', 'wb_neg', 'unlabeled'):
            os.makedirs(os.path.join(args.output_dir, dir_name), exist_ok=True)

    # Download photos
    for example in tqdm(examples):
        if example.is_labeled:
            subdir_name = 'wb_{}'.format('pos' if example.has_whiteboard else 'neg')
        else:
            subdir_name = 'unlabeled'
        file_name = '{}.jpg'.format(example.record_id)
        img_path = os.path.join(args.output_dir, subdir_name, file_name)
        if os.path.exists(img_path):
            util.print_err('Already downloaded {}'.format(img_path))
        url = example.url.replace('original', 'preview')

        try:
            response = requests.get(url, stream=True)
            with open(img_path, 'wb') as out_file:
                shutil.copyfileobj(response.raw, out_file)
            del response

        except Exception as e:
            print('Error downloading from {}: {}'.format(url, e))

        # Down-sample the image
        img = Image.open(img_path, 'r').convert('RGB')
        img = img.resize(IMAGENET_SIZE)
        img.save(img_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download photos from CSV file of URLs')

    parser.add_argument('--csv_path', default='data/wb130k.csv')
    parser.add_argument('--pkl_path', default='data/catal/wb_14k.pkl')
    parser.add_argument('--output_dir', default='/data/catal/wb500')

    main(parser.parse_args())
