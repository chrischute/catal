import os
import pandas as pd
import torch
import torch.nn.functional as F

from args import TestArgParser
from data_loader import WhiteboardLoader
from tqdm import tqdm
from saver import ModelSaver


def predict(args):
    model, ckpt_info = ModelSaver.load_model(args.ckpt_path, args.gpu_ids)
    args.start_epoch = ckpt_info['epoch'] + 1
    model = model.to(args.device)
    model.eval()

    # Predict outputs
    data_loader = WhiteboardLoader(args.data_dir, args.phase, args.batch_size,
                                   shuffle=False, do_augment=False, num_workers=args.num_workers)
    all_probs, all_paths = [], []
    with tqdm(total=len(data_loader.dataset), unit=' ' + args.phase) as progress_bar:
        for inputs, targets, paths in data_loader:
            bs, n_crops, c, h, w = inputs.size()
            inputs = inputs.view(-1, c, h, w)  # Fuse batch size and n_crops

            with torch.no_grad():
                logits = model.forward(inputs.to(args.device))
                logits = logits.view(bs, n_crops, -1).mean(1)  # Average over n_crops
                probs = F.softmax(logits, -1)

            # Take probability of whiteboard
            all_probs += [p[1] for p in probs]
            all_paths += list(paths)

            progress_bar.update(inputs.size(0))

    # Write CSV
    record_ids = [os.path.basename(p)[:-4] for p in all_paths]  # Convert to record_id

    df = pd.DataFrame([{'record_id': r,
                        'probability': prob,
                        'has_whiteboard_@{:.2f}'.format(args.prob_threshold): int(prob > args.prob_threshold),
                        'url': get_url(r)}
                       for r, prob in zip(record_ids, all_probs)])
    df.to_csv(os.path.join(args.results_dir, 'outputs.csv'), index=False)


def get_url(record_id, use_preview_url=False):
    """Convert a record_id to a URL.

    Args:
        record_id: String record ID to convert to URL
        use_preview_url: Use the preview URL (for smaller image size) instead of original URL.
    """
    if use_preview_url:
        url = 'http://catalhoyuk.com/netpub/server.np?original={}&site=catalhoyuk&catalog=catalog'.format(record_id)
    else:
        url = 'http://catalhoyuk.com/netpub/server.np?original={}&site=catalhoyuk&catalog=catalog'.format(record_id)

    return url


if __name__ == '__main__':
    parser = TestArgParser()
    predict(parser.parse_args())
