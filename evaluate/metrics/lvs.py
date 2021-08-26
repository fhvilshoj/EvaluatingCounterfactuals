import os
import numpy as np
from tensorflow.keras.models import load_model

# Local imports
from .utils import fix_shape, get_model_dir
from .computer import Computer

MODEL_NAMES = {
    'fakemnist':   ['independent.h5', 'independent_mnist.h5'],
    'celeba':       ['independent_makeup.h5', 'independent_cheekbones.h5', 'independent_attractive.h5', 'independent_lipstick.h5', 'independent_smile.h5'],
}

LABEL_NAMES = { 
    'fakemnist':   ['FakeMNIST', 'MNIST'],
    'celeba':       [ 'Heavy_Makeup', 'High_Cheekbones', 'Attractive', 'Wearing_Lipstick', 'Smiling' ],
}


def KL(P, Q, e=1e-3):
    return P * (np.log(P + (e * P==0)) - np.log(Q+(e*Q==0)))

def JS(P, Q):
    if len(P.shape) == 1: 
        P_ = np.concatenate([1-P.reshape(-1, 1), P.reshape(-1, 1)], -1)
        Q_ = np.concatenate([1-Q.reshape(-1, 1), Q.reshape(-1, 1)], -1)
    else:
        P_ = P
        Q_ = Q 
    
    M   = 0.5 * ( P_ + Q_ )
    js  = 0.5 *( KL(P_, M) + KL(Q_, M) ) 
    js  = js.sum(-1) 
    return js, np.mean(js), np.std(js), P.shape[0]


class LabelVariationScore(Computer):
    def __init__(self, *args, model=None, **kwargs):
        super(LabelVariationScore, self).__init__(*args, **kwargs)

        self.load_models()

        self.name       = type(self).__name__ 
        self.desc       = """LVS:
              
        E_{x~Px} [D_JS( o(x) || o( cf(x) ) )]
        
        """

    def load_models(self):
        dataset = self.cfg.get('data', 'dataset').lower().replace("_", "")
        try: 
            model_names         = MODEL_NAMES[dataset]
            self.label_names    = LABEL_NAMES[dataset]
        except KeyError:
            raise ValueError("LVS can only be computed for FakeMNIST and Celeba-HQ")
        self.class_label_models = [load_model(get_model_dir(self.cfg, mn)) for mn in model_names]

    def score_fn(self, model): 
        def score(i, x, xcf, y, yhat, ycf, yhatcf):
            x   = fix_shape(x)
            xcf = fix_shape(xcf) 

            p   = model.predict(x).squeeze()
            q   = model.predict(xcf).squeeze()
            return p, q 

        return lambda i, args: score(i, *args)

    def compute_divergences(self, s): 
        P = np.array(list( map( lambda x: x[0], s) ) )
        Q = np.array(list( map( lambda x: x[1], s) ) )

        _, mu, std, n   = JS(P, Q)
        ci              = 1.96*std/np.sqrt(n)

        return mu.astype(float), float(ci), len(s)

    def compute(self, *args): 
        aggr_fn = self.compute_divergences
        res     = {}

        for (label_name, model) in zip(self.label_names, self.class_label_models): 
            elt_fn          = self.score_fn(model) 
            res[label_name] = self._iterate(elt_fn, aggr_fn, *args)

        return res

