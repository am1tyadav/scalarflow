from typing import Callable, Optional, Tuple

from scalarflow.callbacks import Callback
from scalarflow.core.scalar import Scalar
from scalarflow.layers import Dense
from scalarflow.metrics import Metric
from scalarflow.training import optimisation_step
from scalarflow.types import ScalarLike


class MLP:
    def __init__(self, layers: Tuple[Dense]) -> None:
        """Multi-layer perceptron.

        Args:
            layers: Tuple of layers

        Raises:
            AssertionError if the output dimension is not set to 1
            for the final layer.
        """

        self._layers = layers
        self._metrics: Optional[Tuple[Metric]] = None
        self._lr: float = 0.1
        self._history = {"epochs": [], "loss": []}

        assert (
            layers[-1].output_dim == 1
        ), "Final layer of MLP must have output dimension of 1"

    def __call__(self, inputs: Tuple[Scalar]) -> Tuple[Scalar] | Scalar:
        outputs = inputs

        for layer in self._layers:
            outputs = layer(inputs=outputs)

        return outputs

    def compile(self, loss_fn: Callable, lr: float, metrics: Tuple[Metric]) -> None:
        self._loss_fn = loss_fn
        self._lr = lr
        self._metrics = metrics

    def fit(
        self,
        examples: Tuple[Tuple[ScalarLike]],
        labels: Tuple[ScalarLike],
        epochs: int,
        batch_size: int,
        callbacks: Tuple[Callback],
    ) -> dict:
        total_examples = len(labels)
        steps_per_epoch = total_examples // batch_size

        if total_examples % batch_size > 0:
            steps_per_epoch += 1

        for callback in callbacks:
            callback.on_training_start()

        for epoch in range(0, epochs):
            total_loss = 0

            for callback in callbacks:
                callback.on_epoch_start(epoch=epoch)

            for step in range(0, steps_per_epoch):
                start_index = step * batch_size
                end_index = min((step + 1) * batch_size, total_examples)

                batch_examples = examples[start_index:end_index]
                batch_labels = labels[start_index:end_index]

                predictions = []

                for batch_example in batch_examples:
                    predictions.append(self.__call__(inputs=batch_example))

                loss = self._loss_fn(batch_labels, predictions)
                optimisation_step(root=loss, lr=self._lr)

                total_loss += loss.data

            total_loss /= steps_per_epoch

            self._history["epochs"].append(epoch)
            computed_metrics = {"loss": total_loss}

            if len(self._metrics) > 0:
                predictions = []

                for example in examples:
                    predictions.append(self.__call__(inputs=example))

                computed_metrics.update(
                    dict(
                        (metric._name, metric(y_true=labels, y_pred=predictions))
                        for metric in self._metrics
                    )
                )

            for metric_name, metric_value in computed_metrics.items():
                if metric_name not in self._history:
                    self._history[metric_name] = []

                self._history[metric_name].append(metric_value)

            for callback in callbacks:
                callback.on_epoch_end(epoch=epoch, metrics=computed_metrics)

        for callback in callbacks:
            callback.on_training_end(epoch=epoch, metrics=computed_metrics)
        return self._history

    def summary(self) -> None:
        print("=" * 10, "Model", "=" * 10)

        for index, layer in enumerate(self._layers):
            layer.summary(index)
