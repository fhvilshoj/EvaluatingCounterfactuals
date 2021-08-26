from .computer import Computer

class TargetClassValidity(Computer):
    def __init__(self, *args, **kwargs):
        super(TargetClassValidity, self).__init__(*args, **kwargs)
        self.name       = type(self).__name__ 
        self.desc       = """TargetClassValidity:

        % of successful counterfactuals

        """
        self.toprint = []

    def skip(self): return False # Avoid skipping unsuccessful counterfactuals. 

    def score(self, i, x, xcf, y, yhat, ycf, yhatcf):
        if yhatcf == -1: return 0   # Unsuccessful counterfactuals are stored with  -1 yhatcf label for Alibi
        return int(yhat != yhatcf)  # ECINN might not change the predicted class.

