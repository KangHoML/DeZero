import numpy as np

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)} is not supported')
            
        self.data = data
        self.grad = None        # 미분값 저장
        self.creator = None     # 해당 변수를 만든 함수

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:            
            f = funcs.pop()
            
            if f is not None:
                x, y = f.input, f.output    
                x.grad = f.backward(y.grad) 

            if x.creator is not None:
                funcs.append(x.creator)       

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        output.set_creator(self)        # output 변수를 만든 함수
        self.input = input              # 입력값 저장 (역전파에서 사용)
        self.output = output
        return output

    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x