from numpy.linalg import norm

# Local imports 
from .computer import Computer

class ElasticNet(Computer):
    def __init__(self, *args, **kwargs):
        super(ElasticNet, self).__init__(*args, **kwargs)

        self.name       = type(self).__name__ 
        self.desc       = """EN:

        EN(delta) = ||delta||_1 + ||delta||_2^2
        Lower values means less changed.

        """

    def score(self, i, x, xcf, y, yhat, ycf, yhatcf):
        delta   = x.reshape(-1) - xcf.reshape(-1)
        l1      = norm( delta, ord=1 )
        l2      = norm( delta, ord=2 )
        return l1 + l2


