import os
import pandas as pd
import torch

from args import TestArgParser
from data_loader import WhiteboardLoader
from saver import ModelSaver


def predict(args):
    model, ckpt_info = ModelSaver.load_model(args.ckpt_path, args.gpu_ids)
    args.start_epoch = ckpt_info['epoch'] + 1
    model = model.to(args.device)
    model.train()

    # Get logger, evaluator, saver
    data_loader = WhiteboardLoader(args.data_dir, args.phase, args.batch_size,
                                   shuffle=True, do_augment=True, num_workers=args.num_workers)

    # Predict outputs
    all_probs, all_paths = [], []
    for inputs, targets, paths in data_loader:

        with torch.no_grad():
            logits = model.forward(inputs.to(args.device))
            probs = torch.sigmoid(logits)

        all_probs += probs.to('cpu').numpy().tolist()
        all_paths += list(paths)

    # Write CSV
    record_ids = [os.path.basename(p)[:-4] for p in all_paths]  # Convert to record_id
    predictions = [int(p > args.prob_threshold) for p in all_probs]
    df = pd.DataFrame([{'record_id': r, 'has_whiteboard': p} for r, p in zip(record_ids, predictions)])
    df.to_csv(os.path.join(args.save_dir, 'outputs.csv'))


if __name__ == '__main__':
    parser = TestArgParser()
    predict(parser.parse_args())
