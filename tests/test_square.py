import unittest
import numpy as np
import dezero.functions as F

from dezero.core import Variable
from dezero.utils import numerical_diff

class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = F.square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)
    
    def test_backward(self):
        x = Variable(np.array(3.0))
        y = F.square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = F.square(x)
        y.backward()
        num_grad = numerical_diff(F.square, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)