# from typing import Tuple, Callable

# import pydot

# from scalarflow.core import Scalar, ScalarOp
# from scalarflow.mlp import MLP


# def plot_graph(mlp: MLP, example: Tuple[Scalar], label: Scalar, loss_fn: Callable):
#     dot_graph = pydot.Dot("ScalarGraph", graph_type="digraph")

#     prediction = mlp(example)
#     loss = loss_fn(label, prediction)

#     mlp.backward(root=loss)
#     mlp.refresh()

#     for op_or_scalar in mlp.graph:
#         if op_or_scalar is not None:
#             if isinstance(op_or_scalar, Scalar):
#                 node_label = str(op_or_scalar)
#                 shape = "box"
#                 color = "red" if op_or_scalar.trainable else "black"
#             else:
#                 node_label = op_or_scalar.name
#                 shape = "circle"
#                 color = "green"

#             node = pydot.Node(
#                 name=op_or_scalar._id, label=node_label, shape=shape, color=color
#             )
#             dot_graph.add_node(node)

#     for op_or_scalar in mlp.graph:
#         if isinstance(op_or_scalar, ScalarOp):
#             for argument in op_or_scalar.arguments:
#                 edge = pydot.Edge(argument._id, op_or_scalar._id)
#                 dot_graph.add_edge(edge)

#             # if op_or_scalar.arguments[0] != op_or_scalar.result:
#             edge = pydot.Edge(op_or_scalar._id, op_or_scalar.result._id)
#             dot_graph.add_edge(edge)

#     dot_graph.write_png("graph.png")
