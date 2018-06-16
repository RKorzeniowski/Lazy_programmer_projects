# for like everything e.g. scalar, tensors, matrixes
import theano.tensor as T
import numpy as np
import theano


c = T.scalar('c')

v = T.vector('v')

A = T.matrix('A')
# those are only symbols(thay have no values) so we can go do algebra on them

# tensors are for dimentionalyty 3 and up (e.g. images that have not been faltend yet)

w = A.dot(v)
# each function creation specifies input and output dimentions
matrix_times_vector = theano.function(inputs=[A, v], outputs=w)
A_val = np.array([[1, 2], [3, 4]])
v_val = np.array([5, 6])

m_val = matrix_times_vector(A_val, v_val)

print(m_val)

# variable in theano are not updateable and we need to make special shared variable to make it updateable
x = theano.shared(20.0, 'x')

cost = x * x + x + 1

x_update = x - 0.3 * T.grad(cost, x)

train = theano.function(inputs=[], outputs=cost, updates=[(x, x_update)])

for i in range(25):
    cost_val = train()
    print(cost_val)

print(x.get_value())
