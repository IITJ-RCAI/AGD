from custom_metric import *
from operations import *

import os


def _set_maestro(v):
    os.environ["USE_MAESTRO"] = "1" if v is True else "0"


c = Conv(3, 10)
_set_maestro(True)
print(c.forward_flops((3, 32, 32)))
_set_maestro(False)
print(c.forward_flops((3, 32, 32)))

_set_maestro(True)
print(c.forward_flops((3, 32, 32)))
_set_maestro(False)
print(c.forward_flops((3, 32, 32)))