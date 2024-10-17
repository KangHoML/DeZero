from dezero.core import Variable, as_array

def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(as_array(x.data - eps)) 
    x1 = Variable(as_array(x.data + eps))
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)