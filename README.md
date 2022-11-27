# scalarflow

A Machine Learning library written in pure Python for educational purpose. It's very slow,
and almost completely useless for solving real problems, and should be used just to
understand and explore basic machine learning concepts like gradient descent, automatic
differentiation etc - i.e. start with a Keras-like API but with much simpler, easier to
understand functionality under the hood.

This project is heavily inspired by [Micrograd](https://github.com/karpathy/micrograd).

## Usage

```python
import scalarflow as sf


mlp = sf.MLP(
    layers=(
        sf.Dense(output_dim=2, input_dim=2),
        sf.Dense(output_dim=1, input_dim=2, activation=sf.sigmoid),
    )
)

mlp.fit(
    examples=examples,
    labels=labels,
    epochs=200,
    batch_size=4,
    loss_fn=sf.mean_squared_error,
    lr=0.009,
    log_interval=20,
    show_accuracy=True,
)
```

## Examples

Concrete examples can be found in the [Example Notebooks](/examples/):

- Linear regression with a single Node
- Linear regression with MLP
- Logistic regression with a single Node
- Logistic regression with MLP
