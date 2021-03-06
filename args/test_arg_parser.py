import util

from .base_arg_parser import BaseArgParser


class TestArgParser(BaseArgParser):
    """Argument parser for args used only in test mode."""
    def __init__(self):
        super(TestArgParser, self).__init__()
        self.is_training = False

        self.parser.add_argument('--phase', type=str, default='val', choices=('train', 'val', 'test'),
                                 help='Phase to test on.')
        self.parser.add_argument('--results_dir', type=str, default='results/', help='Save dir for test results.')
        self.parser.add_argument('--prob_threshold', type=float, default=0.7,
                                 help='Probability threshold for saying an example is positive.')
