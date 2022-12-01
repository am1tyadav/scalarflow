from typing import Callable, Tuple

from scalarflow.core.scalar import Scalar
from scalarflow.layers import Dense
from scalarflow.training import accuracy, optimisation_step

ScalarLike = Scalar | int | float


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

        assert (
            layers[-1].output_dim == 1
        ), "Final layer of MLP must have output dimension of 1"

    def __call__(self, inputs: Tuple[Scalar]) -> Tuple[Scalar] | Scalar:
        outputs = inputs

        for layer in self._layers:
            outputs = layer(inputs=outputs)

        return outputs

    def log(
        self,
        examples: Tuple[Tuple[ScalarLike]],
        labels: Tuple[ScalarLike],
        epoch: int,
        total_loss: float,
        show_accuracy: bool,
    ):
        log_output = f"Epoch: {epoch:4d} - Loss: {total_loss: .5f}"

        if show_accuracy:
            predictions = [self.__call__(example) for example in examples]
            epoch_accuracy = accuracy(y_true=labels, y_pred=predictions)
            log_output = f"{log_output} - Accuracy: {epoch_accuracy:.4f}"
        print(log_output)

    def fit(
        self,
        examples: Tuple[Tuple[ScalarLike]],
        labels: Tuple[ScalarLike],
        epochs: int,
        batch_size: int,
        loss_fn: Callable,
        lr: float = 0.1,
        log_interval: int = 1,
        show_accuracy: bool = False,
    ) -> None:

        total_examples = len(labels)
        steps_per_epoch = total_examples // batch_size

        if total_examples % batch_size > 0:
            steps_per_epoch += 1

        for epoch in range(0, epochs):
            total_loss = 0

            for step in range(0, steps_per_epoch):
                start_index = step * batch_size
                end_index = min((step + 1) * batch_size, total_examples)

                batch_examples = examples[start_index:end_index]
                batch_labels = labels[start_index:end_index]

                predictions = []

                for batch_example in batch_examples:
                    predictions.append(self.__call__(inputs=batch_example))

                loss = loss_fn(batch_labels, predictions)
                optimisation_step(root=loss, lr=lr)

                total_loss += loss.data

            total_loss /= steps_per_epoch

            if epoch % log_interval == 0 or epoch == epochs - 1:
                self.log(
                    examples=examples,
                    labels=labels,
                    epoch=epoch,
                    total_loss=total_loss,
                    show_accuracy=show_accuracy,
                )

    def summary(self) -> None:
        print("=" * 10, "Model", "=" * 10)

        for index, layer in enumerate(self._layers):
            layer.summary(index)
