from scalarflow.core.operator import add, multiply, power, subtract

a = add(1, 2)

b = subtract(a, 2)

c = multiply(2, b)

d = power(c, 3)

print(d)
