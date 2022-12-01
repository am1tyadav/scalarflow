class Callback:
    def __init__(self, name: str) -> None:
        self._name = name

    def on_training_start(self):
        return

    def on_training_end(self, epoch: int, metrics: dict):
        return

    def on_epoch_start(self, epoch: int):
        return

    def on_epoch_end(self, epoch: int, metrics: dict):
        return


class ConsoleLogger(Callback):
    def __init__(self, log_interval: int) -> None:
        super().__init__(name="console_logger")

        self._log_interval = log_interval

    def log(self, epoch: int, metrics: dict):
        output = f"Epoch: {epoch} "

        for metric, value in metrics.items():
            output += f"{metric}: {value:.4f} "

        print(output)

    def on_epoch_end(self, epoch: int, metrics: dict):
        if epoch % self._log_interval == 0:
            self.log(epoch=epoch, metrics=metrics)

    def on_training_end(self, epoch: int, metrics: dict):
        self.log(epoch=epoch, metrics=metrics)
