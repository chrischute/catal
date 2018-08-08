from args import TestArgParser
from data_loader import WhiteboardLoader
from logger import TestLogger
from evaluator import ModelEvaluator
from saver import ModelSaver


def test(args):

    model, ckpt_info = ModelSaver.load_model(args.ckpt_path, args.gpu_ids)
    args.start_epoch = ckpt_info['epoch'] + 1
    model = model.to(args.device)
    model.eval()

    # Run a single evaluation
    eval_loader = WhiteboardLoader(args.data_dir, args.phase, args.batch_size,
                                   shuffle=False, do_augment=False, num_workers=args.num_workers)
    logger = TestLogger(args, len(eval_loader.dataset))
    logger.start_epoch()
    evaluator = ModelEvaluator(args.task_type, [eval_loader], logger, args.num_visuals)
    metrics = evaluator.evaluate(model, args.device, logger.epoch)
    logger.end_epoch(metrics)


if __name__ == '__main__':
    parser = TestArgParser()
    test(parser.parse_args())
