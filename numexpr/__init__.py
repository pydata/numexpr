import os.path

from numexpr.info import __doc__
from numexpr.expressions import E
from numexpr.necompiler import numexpr, disassemble, evaluate
from numexpr.tests import test

dirname = os.path.dirname(__file__)

__version__ = open(os.path.join(dirname,'VERSION')).read().strip()
