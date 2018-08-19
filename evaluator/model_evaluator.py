import numpy as np
import random
import sklearn.metrics as sk_metrics
import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluator.average_meter import AverageMeter
from tqdm import tqdm


class ModelEvaluator(object):
    """Class for evaluating a model during training."""
    def __init__(self, data_loaders, logger, epochs_per_eval=1, max_eval=None, num_visuals=8, prob_threshold=0.5):
        """
        Args:
            data_loaders: List of Torch `DataLoader`s to sample from.
            logger: Logger for plotting to console and TensorBoard.
            epochs_per_eval: Number of epochs between each evaluation.
            max_eval: Maximum number of examples to evaluate at each evaluation.
            num_visuals: Number of visuals to display from the validation set.
            prob_threshold: Probability threshold for saying an example is positive.
        """
        self.data_loaders = data_loaders
        self.logger = logger
        self.epochs_per_eval = epochs_per_eval
        self.max_eval = None if max_eval is None or max_eval < 0 else max_eval
        self.num_visuals = num_visuals
        self.prob_threshold = prob_threshold
        self.loss_fn = nn.CrossEntropyLoss()

    def evaluate(self, model, device, epoch=None):
        """Evaluate a model at the end of the given epoch.

        Args:
            model: Model to evaluate.
            device: Device on which to evaluate the model.
            epoch: The epoch that just finished. Determines whether to evaluate the model.

        Returns:
            metrics: Dictionary of metrics for the current model.

        Notes:
            Returned dictionary will be empty if not an evaluation epoch.
        """
        metrics = {}

        if epoch is None or epoch % self.epochs_per_eval == 0:
            # Evaluate on the training and validation sets
            model.eval()
            for data_loader in self.data_loaders:
                phase_metrics = self._eval_phase(model, data_loader, data_loader.phase, device)
                metrics.update(phase_metrics)
            model.train()

        return metrics

    def _eval_phase(self, model, data_loader, phase, device):
        """Evaluate a model for a single phase.

        Args:
            model: Model to evaluate.
            data_loader: Torch DataLoader to sample from.
            phase: Phase being evaluated. One of 'train', 'val', or 'test'.
            device: Device on which to evaluate the model.

        Returns:
            metrics: Dictionary of metrics for the phase.
        """

        # Keep track of task-specific records needed for computing overall metrics
        records = {'loss_meter': AverageMeter(), 'probs': [], 'labels': []}

        num_examples = len(data_loader.dataset)
        if self.max_eval is not None:
            num_examples = min(num_examples, self.max_eval)

        # Sample from the data loader and record model outputs
        loss_fn = nn.CrossEntropyLoss()
        num_evaluated = num_visualized = 0
        start_visual = random.randint(0, max(1, num_examples - self.num_visuals))
        with tqdm(total=num_examples, unit=' ' + phase) as progress_bar:
            for inputs, targets, paths in data_loader:
                if num_evaluated >= num_examples:
                    break

                with torch.no_grad():
                    logits = model.forward(inputs.to(device))
                    loss = loss_fn(logits, targets.to(device))

                self._record_batch(logits, targets, loss, **records)

                if start_visual <= num_evaluated and num_visualized < self.num_visuals and phase != 'train':
                    num_visualized += self.logger.visualize(inputs, logits, targets, paths, phase=phase)

                progress_bar.update(min(inputs.size(0), num_examples - num_evaluated))
                num_evaluated += inputs.size(0)

        # Map to summary dictionaries
        metrics = self._get_summary_dict(phase, **records)

        return metrics

    @staticmethod
    def _record_batch(logits, targets, loss, loss_meter=None, probs=None, labels=None):
        """Record results from a batch to keep track of metrics during evaluation.

        Args:
            logits: Batch of logits output by the model.
            loss_meter: AverageMeter keeping track of average loss during evaluation.
        """
        with torch.no_grad():
            batch_probs = F.softmax(logits, dim=-1)
        probs.append(np.array([p[1] for p in batch_probs]))

        # Note: `targets` is assumed to hold the keys for these examples
        labels.append(targets)

        if loss_meter is not None:
            loss_meter.update(loss.item(), logits.size(0))

    def _get_summary_dict(self, phase, loss_meter=None, probs=None, labels=None):
        """Get summary dictionaries given dictionary of records kept during evaluation.

        Args:
            phase: Phase being evaluated. One of 'train', 'val', or 'test'.
            loss_meter: AverageMeter keeping track of average loss during evaluation.
            probs: List of probabilities.
            labels: List of labels, parallel to probs.

        Returns:
            metrics: Dictionary of metrics for the current model.
        """
        metrics = {phase + '_' + 'loss': loss_meter.avg}

        if probs is not None:
            # Convert to flat numpy array
            probs = np.concatenate(probs).ravel()
            preds = (probs > self.prob_threshold)
            labels = np.concatenate(labels).ravel()

            # Update summary dicts
            try:
                metrics.update({
                    phase + '_' + 'AUPRC': sk_metrics.average_precision_score(labels, probs),
                    phase + '_' + 'AUROC': sk_metrics.roc_auc_score(labels, probs),
                })
            except ValueError:
                pass

            metrics.update({
                phase + '_' + 'accuracy': sk_metrics.accuracy_score(labels, preds),
                phase + '_' + 'precision': sk_metrics.precision_score(labels, preds),
                phase + '_' + 'recall': sk_metrics.recall_score(labels, preds)
            })

        return metrics
