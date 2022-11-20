import scalarflow as sf

n = sf.Node(num_inputs=3)

output = n(inputs=(1, 2, 3))

print(output)
