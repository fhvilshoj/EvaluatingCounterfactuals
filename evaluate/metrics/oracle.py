from tensorflow.keras.models import load_model
import os

# Local imports
from .computer import Computer
from .utils import fix_shape, get_model_dir

class Oracle(Computer):
    def __init__(self, *args, **kwargs):
        super(Oracle, self).__init__(*args, **kwargs)
        dataset = self.cfg.get('data', 'dataset').lower().replace("_", "")

        model_name = 'independent.h5' if 'mnist' in dataset else 'independent_makeup.h5'
        pth = get_model_dir(self.cfg, model_name)
        self.classifier = load_model(pth)

        self.name       = type(self).__name__ 
        self.desc       = """Oracle:

        Oracle = 1_{f(c) == o(c)}
        Number of agreements with external classifier. Closer to 1 is better 
        (classifier also convinced that the class changed)

        """

    def score(self, i, x, xcf, y, yhat, ycf, yhatcf):
        pred = self.classifier.predict(fix_shape(xcf))
        sc = ycf == pred.argmax(1).squeeze()
        return sc


