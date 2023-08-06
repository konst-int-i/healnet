

def calc_reg_loss(model, l1: float, model_topo: str):
    if model_topo == "fcnn": # don't regularise FCNN
        reg_loss = 0
    else:
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        reg_loss = float(l1) * l1_norm
    return reg_loss

class EarlyStopping:
    def __init__(self, patience=5, delta=0, maximize=True):
        """

        Args:
            patience: How long to wait after last time validation metric improved.
            delta: Minimum change in the monitored quantity to qualify as an improvement.
            maximize: Indicates if the objective is to maximize the monitored metric.
        """
        self.patience = patience
        self.delta = delta
        self.maximize = maximize
        self.best_score = None
        self.counter = 0

    def __call__(self, score):
        stop_training = False

        if self.best_score is None:
            self.best_score = score
        elif ((score < self.best_score + self.delta) if self.maximize else (score > self.best_score - self.delta)):
            self.counter += 1
            if self.counter >= self.patience:
                stop_training = True
        else:
            self.best_score = score
            self.counter = 0

        return stop_training