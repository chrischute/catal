import torch
import torch.nn.functional as F
import util

from args import TestArgParser
from data_loader import FakeDataLoader
from saver import ModelSaver
from tqdm import tqdm


def test(args):

    model, ckpt_info = ModelSaver.load_model(args.ckpt_path, args.gpu_ids)
    args.start_epoch = ckpt_info['epoch'] + 1
    model = model.to(args.device)
    model.eval()

    data_loader = FakeDataLoader(1000, args.num_classes, 'val', args.batch_size,
                                 shuffle=False, do_augment=False, num_workers=args.num_workers)

    # Get model outputs, log to TensorBoard, write masks to disk window-by-window
    util.print_err('Writing model outputs to {}...'.format(args.results_dir))
    with tqdm(total=len(data_loader.dataset), unit=' examples') as progress_bar:
        for i, (inputs, info_dict) in enumerate(data_loader):
            with torch.no_grad():
                logits = model.forward(inputs.to(args.device))
                probs = F.softmax(logits)

            raise NotImplementedError('Test script in progress')

            progress_bar.update(inputs.size(0))


if __name__ == '__main__':
    parser = TestArgParser()
    test(parser.parse_args())
