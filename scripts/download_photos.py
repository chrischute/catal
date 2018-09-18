"""Download photos using metadata from a CSV file.

Usage:
  1. Create CSV file with at least the column 'netpublish_URL'.
     Note: 'original' gets replaced with 'preview'.
  2. Make sure you've set up the catal environment, and run `source activate catal`.
  3. Run `python download_photos.py --csv_path <PATH_TO_CSV> --output_dir <PATH_TO_OUTPUT_DIR>`
"""
import argparse
import os
import pandas as pd
import requests
import shutil
import util

from catal import CatalPhoto
from PIL import Image
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


def main(args):
    df = pd.read_csv(args.csv_path)
    examples = [CatalPhoto(url=str(row['netpublish_URL']), annotation=None) for _, row in df.iterrows()]

    # Make directories for holding photos
    for dir_name in ('wb_pos', 'wb_neg', 'unlabeled'):
        os.makedirs(os.path.join(args.output_dir, dir_name), exist_ok=True)

    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

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
            continue
        url = example.url.replace('original', 'preview')

        try:
            response = session.get(url, stream=True, timeout=10)
            with open(img_path, 'wb') as out_file:
                shutil.copyfileobj(response.raw, out_file)
            del response

        except Exception as e:
            print('Error downloading from {}: {}'.format(url, e))
            continue

        # Down-sample the image
        if args.resize_shape is not None:
            img = Image.open(img_path, 'r').convert('RGB')
            img = img.resize(args.resize_shape)
            img.save(img_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download photos from CSV file of URLs')

    parser.add_argument('--csv_path', default='data/wb130k.csv')
    parser.add_argument('--output_dir', default='data/wb130k')
    parser.add_argument('--resize_shape', default=None, type=eval,
                        help='Size to reshape downloaded images. E.g. "(224, 224)".')

    main(parser.parse_args())
