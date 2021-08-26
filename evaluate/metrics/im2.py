from numpy.linalg import norm

# Local imports 
from .computer import Computer
from .utils import load_aes, fix_shape

class IM2(Computer):
    def __init__(self, *args, epsilon=1e-5, aes=None, **kwargs):
        super(IM2, self).__init__(*args, **kwargs)
        self.epsilon = epsilon

        if aes is not None: self.aes = aes
        else:               self.aes = load_aes(self.cfg)

        self.name       = type(self).__name__
        self.desc       = """IM2:
             
              ||ae_q(c) - ae(c)||^2_2 
        IM2 = ----------------------- 
                    ||c||_1
        Closer to zero means more interpretable counterfactual.

        """

    def score(self, i, x, xcf, y, yhat, ycf, yhatcf, return_tracks=False):
        x           = fix_shape(x)
        xcf         = fix_shape(xcf)

        ae          = self.aes[-1]
        ae_to       = self.aes[ycf]

        rec         = ae.predict(xcf)
        to_rec      = ae_to.predict(xcf)
        
        rec_diff    = norm( to_rec.flatten() - rec.flatten() )**2
        cf_norm     = norm( xcf.flatten(), ord=1 )

        if return_tracks:   return rec_diff / (cf_norm + self.epsilon), (rec_diff, cf_norm)
        else:               return rec_diff / (cf_norm + self.epsilon)

