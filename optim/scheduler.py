import math
import torch.optim as optim

from functools import partial


def get_scheduler(optimizer, args):
    """Get a learning rate scheduler.

    Args:
        optimizer: The optimizer whose learning rate is modified by the returned scheduler.
        args: Command line arguments.

    Returns:
        PyTorch scheduler that update the learning rate for `optimizer`.
    """
    if args.lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)
    elif args.lr_scheduler == 'multi_step':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=args.lr_decay_gamma)
    elif args.lr_scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.lr_decay_gamma, patience=args.patience)
    elif args.lr_scheduler == 'cosine':
        lambda_fns = []
        if args.pretrained:
            # For pretrained params, delay the warmup to let randomly initialized head settle
            lambda_fns.append(partial(linear_warmup_then_cosine, delay=args.lr_warmup_steps,
                                      warmup=args.lr_warmup_steps, max_iter=args.lr_decay_step))
        lambda_fns.append(partial(linear_warmup_then_cosine, warmup=args.lr_warmup_steps, max_iter=args.lr_decay_step))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda_fns)
    else:
        raise ValueError('Invalid learning rate scheduler: {}.'.format(args.lr_scheduler))

    return scheduler


def linear_warmup_then_cosine(last_iter, warmup, max_iter, delay=None):
    if delay is not None:
        last_iter = max(0, last_iter - delay)

    if last_iter < warmup:
        # Linear warmup period
        return float(last_iter) / warmup
    elif last_iter < max_iter:
        # Cosine annealing
        return (1 + math.cos(math.pi * (last_iter - warmup) / max_iter)) / 2
    else:
        # Done
        return 0.


def step_scheduler(lr_scheduler, metrics=None, epoch=None, global_step=None, best_ckpt_metric='val_loss'):
    """Step a LR scheduler."""
    if global_step is not None:
        if isinstance(lr_scheduler, optim.lr_scheduler.LambdaLR):
            lr_scheduler.step(global_step)
    elif isinstance(lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        if best_ckpt_metric in metrics:
            lr_scheduler.step(metrics[best_ckpt_metric], epoch=epoch)
    else:
        lr_scheduler.step(epoch=epoch)
