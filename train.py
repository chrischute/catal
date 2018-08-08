import optim
import torch
import torch.nn as nn
import torchvision.models as models

from args import TrainArgParser
from data_loader import WhiteboardLoader
from evaluator import ModelEvaluator
from logger import TrainLogger
from saver import ModelSaver


def train(args):

    if args.ckpt_path:
        model, ckpt_info = ModelSaver.load_model(args.ckpt_path, args.gpu_ids)
        args.start_epoch = ckpt_info['epoch'] + 1
    else:
        model_fn = models.__dict__[args.model]
        model = model_fn(pretrained=args.pretrained)
        if args.pretrained:
            model.fc = nn.Linear(model.fc.in_features, args.num_classes)
        model = nn.DataParallel(model, args.gpu_ids)
    model = model.to(args.device)
    model.train()

    # Get optimizer and scheduler
    optimizer = optim.get_optimizer(model.parameters(), args)
    lr_scheduler = optim.get_scheduler(optimizer, args)
    if args.ckpt_path:
        ModelSaver.load_optimizer(args.ckpt_path, optimizer, lr_scheduler)

    # Get logger, evaluator, saver
    loss_fn = nn.CrossEntropyLoss()
    train_loader = WhiteboardLoader(args.data_dir, 'train', args.batch_size,
                                    shuffle=True, do_augment=True, num_workers=args.num_workers)
    logger = TrainLogger(args, len(train_loader.dataset))
    eval_loaders = [WhiteboardLoader(args.data_dir, 'val', args.batch_size,
                                     shuffle=False, do_augment=False, num_workers=args.num_workers)]
    evaluator = ModelEvaluator(eval_loaders, logger, args.epochs_per_eval, args.max_eval, args.num_visuals)
    saver = ModelSaver(**vars(args))

    # Train model
    while not logger.is_finished_training():
        logger.start_epoch()

        for inputs, targets in train_loader:
            logger.start_iter()
            
            with torch.set_grad_enabled(True):
                logits = model.forward(inputs.to(args.device))
                loss = loss_fn(logits, targets.to(args.device))

                logger.log_iter(loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            logger.end_iter()

        metrics = evaluator.evaluate(model, args.device, logger.epoch)
        saver.save(logger.epoch, model, args.model, optimizer, lr_scheduler, args.device,
                   metric_val=metrics.get(args.metric_name, None))
        logger.end_epoch(metrics)
        optim.step_scheduler(lr_scheduler, metrics, logger.epoch)


if __name__ == '__main__':
    parser = TrainArgParser()
    train(parser.parse_args())
