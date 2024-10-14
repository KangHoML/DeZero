import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None        # 미분값 저장

class Function:
    def __call__(self, input: Variable):
        x = input.data
        y = self.forward(x)
        self.input = input      # 입력값 저장 (역전파에서 사용)
        output = Variable(y)
        return output

    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()
    
class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx
    
class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

def numerical_diff(f, x: Variable, eps=1e-4):
    x0 = Variable(x.data - eps) 
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / 2 * eps
