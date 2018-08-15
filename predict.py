import numpy as np
import os
import pandas as pd
import torch
import torch.nn.functional as F
import util

from args import TestArgParser
from data_loader import WhiteboardLoader
from evaluator import ModelEvaluator
from logger import TestLogger
from tqdm import tqdm
from saver import ModelSaver


def predict(args):
    model, ckpt_info = ModelSaver.load_model(args.ckpt_path, args.gpu_ids)
    args.start_epoch = ckpt_info['epoch'] + 1
    model = model.to(args.device)
    model.eval()

    # Get logger, evaluator, saver
    data_loader = WhiteboardLoader(args.data_dir, args.phase, args.batch_size,
                                   shuffle=False, do_augment=False, num_workers=args.num_workers)

    # Run a single evaluation
    util.print_err('Running evaluation...')
    eval_loader = WhiteboardLoader(args.data_dir, args.phase, args.batch_size,
                                   shuffle=False, do_augment=False, num_workers=args.num_workers)
    logger = TestLogger(args, len(eval_loader.dataset))
    logger.start_epoch()
    evaluator = ModelEvaluator([eval_loader], logger, num_visuals=args.num_visuals, prob_threshold=args.prob_threshold)
    metrics = evaluator.evaluate(model, args.device, logger.epoch)
    logger.end_epoch(metrics)
    model.eval()

    # Predict outputs
    util.print_err('Generating predictions CSV...')
    all_probs, all_paths = [], []
    with tqdm(total=len(data_loader.dataset), unit=' ' + args.phase) as progress_bar:
        for inputs, targets, paths in data_loader:

            with torch.no_grad():
                logits = model.forward(inputs.to(args.device))
                probs = F.sigmoid(logits)

            # Take probability of whiteboard
            probs = np.array([p[1] for p in probs])
            all_probs += probs.ravel().tolist()
            all_paths += list(paths)

            progress_bar.update(inputs.size(0))

    # Write CSV
    record_ids = [os.path.basename(p)[:-4] for p in all_paths]  # Convert to record_id
    predictions = [int(p > args.prob_threshold) for p in all_probs]
    print('Mean prediction: {}'.format(np.mean(predictions)))

    df = pd.DataFrame([{'record_id': r, 'has_whiteboard': p} for r, p in zip(record_ids, predictions)])
    df.to_csv(os.path.join(args.results_dir, 'outputs.csv'), index=False)


if __name__ == '__main__':
    parser = TestArgParser()
    predict(parser.parse_args())
