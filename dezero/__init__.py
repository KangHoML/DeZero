from dezero.core import Variable
from dezero.core import Parameter
from dezero.core import Function
from dezero.core import Config
from dezero.core import using_config
from dezero.core import no_grad
from dezero.core import as_array
from dezero.core import as_variable
from dezero.core import setup_variable
from dezero.layers import Layer
from dezero.models import Model

import dezero.utils
import dezero.functions
import dezero.layers


__version__ = '0.0.1'
setup_variable()