from numpy.linalg import norm

# Local imports 
from .computer import Computer
from .utils import load_aes, fix_shape

class IM1(Computer):
    def __init__(self, *args, epsilon=1e-5, **kwargs):
        super(IM1, self).__init__(*args, **kwargs)
        self.epsilon    = epsilon
        self.aes        = load_aes(self.cfg)

        self.name       = type(self).__name__
        self.desc       = """IM1:

              ||c - ae_q(c)||^2_2 
        IM1 = ----------------------- 
              ||c - ae_p(c)||^2_2 + e
        Less than one means that CF is more representative in cf class than original class

        """

    def score(self, i, x, xcf, y, yhat, ycf, yhatcf):
        x           = fix_shape(x)
        xcf         = fix_shape(xcf)

        ae_from     = self.aes[yhat]
        ae_to       = self.aes[ycf]

        to_rec      = ae_to.predict(xcf)
        fr_rec      = ae_from.predict(xcf)
        
        cf_diff     = norm( xcf.flatten() - to_rec.flatten() )**2
        fr_diff     = norm( xcf.flatten() - fr_rec.flatten() )**2

        return cf_diff / (fr_diff + self.epsilon)

