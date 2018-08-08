import torch.optim as optim
import util


def get_optimizer(parameters, args):
    """Get a PyTorch optimizer for params.

    Args:
        parameters: Iterator of network parameters to optimize (i.e., model.parameters()).
        args: Command line arguments.

    Returns:
        PyTorch optimizer specified by args_.
    """
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(parameters, args.lr,
                              momentum=args.sgd_momentum,
                              nesterov=True,
                              weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(parameters, args.lr,
                               betas=(args.adam_beta_1, args.adam_beta_2), weight_decay=args.weight_decay)
    else:
        raise ValueError('Unsupported optimizer: {}'.format(args.optimizer))

    return optimizer


def get_parameters(model, args):
    """Get parameter generators for a model.

    Args:
        model: Model to get parameters from.
        args: Command-line arguments.

    Returns:
        Dictionary of parameter generators that can be passed to a PyTorch optimizer.
    """

    def gen_params(boundary_layer_name, fine_tuning):
        """Generate parameters, if fine_tuning generate the params before boundary_layer_name.
        If unfrozen, generate the params at boundary_layer_name and beyond."""
        saw_boundary_layer = False
        for name, param in model.named_parameters():
            if name.startswith(boundary_layer_name):
                saw_boundary_layer = True

            if saw_boundary_layer and fine_tuning:
                return
            elif not saw_boundary_layer and not fine_tuning:
                continue
            else:
                yield param

    # Fine-tune the network's layers from encoder.2 onwards
    if args.pretrained or args.fine_tune:
        optimizer_parameters = [{'params': gen_params(args.fine_tuning_boundary, fine_tuning=True),
                                 'lr': args.fine_tuning_lr},
                                {'params': gen_params(args.fine_tuning_boundary, fine_tuning=False)}]
    else:
        optimizer_parameters = [{'params': gen_params(args.fine_tuning_boundary, fine_tuning=False)}]

    # Debugging info
    util.print_err('Number of fine-tuning layers: {}'
                   .format(sum(1 for _ in gen_params(args.fine_tuning_boundary, fine_tuning=True))))
    util.print_err('Number of regular layers: {}'
                   .format(sum(1 for _ in gen_params(args.fine_tuning_boundary, fine_tuning=False))))

    return optimizer_parameters
