# scalarflow

A Machine Learning library written in pure Python for educational purpose. It's very slow,
and almost completely useless for solving real problems, and should be used just to
understand and explore basic machine learning concepts like gradient descent, automatic
differentiation etc - i.e. start with a Keras-like API but with much simpler, easier to
understand functionality under the hood.

This project is heavily inspired by [Micrograd](https://github.com/karpathy/micrograd).

## Installation

Create a virtual enivironment:

```sh
python -m venv venv

source venv/bin/activate
```

Install `scalarflow` with pip in one of the following ways:

```sh
pip install git+https://github.com/am1tyadav/scalarflow
```

or

```sh
git clone https://github.com/am1tyadav/scalarflow

cd scalarflow

pip install .
```

## Usage

```python
import scalarflow as sf


mlp = sf.models.MLP(
    layers=(
        sf.layers.Dense(output_dim=2, input_dim=2),
        sf.layers.Dense(output_dim=1, input_dim=2, activation=sf.operators.sigmoid),
    )
)

mlp.fit(
    examples=examples,
    labels=labels,
    epochs=200,
    batch_size=4,
    loss_fn=sf.losses.mean_squared_error,
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
