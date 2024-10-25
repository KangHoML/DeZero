import weakref
import numpy as np
import dezero.functions as F
from dezero.core import Parameter

class Layer:
    def __init__(self):
        self._params = set()

    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs, )
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, inputs):
        raise NotImplementedError()
    
    def params(self):
        for name in self._params:
            obj = self.__dict__[name]

            if isinstance(obj, Layer):
                yield from obj.params()
            else:
                yield obj
    
    def cleargrads(self):
        for param in self.params():
            param.cleargrad()

class Linear(Layer):
    def __init__(self, out_features, bias=False, dtype=np.float32, in_features=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        
        self.W = Parameter(None, name='W')
        if self.in_features is not None:
            self._init_W()
        
        if bias:
            self.b = Parameter(np.zeros(self.out_features), name='b')
        else:
            self.b = None
            
    def _init_W(self):
        # xavier initialization
        in_feats, out_feats = self.in_features, self.out_features
        weight = np.random.randn(in_feats, out_feats).astype(self.dtype) * np.sqrt(1 / in_feats)
        self.W.data = weight

    def forward(self, x):
        if self.W.data is None:
            self.in_features = x.shape[1]
            self._init_W()
        
        y = F.linear(x, self.W, self.b)
        return y