class EarlyStopping:
    def __init__(self, gaman_count, verbose = 0):

        """
        verbose: 1ならimmplement early stoppingが実行される

        """

        self.step = 0
        self.loss = float("inf")
        self.gaman_count = gaman_count
        self.verbose = verbose

    def __call__(self, val_loss):
        if self.loss < val_loss:
            self.step += 1
            if self.step > self.gaman_count:
                if self.verbose:
                    print("immplement early stopping")
                return True

        else:
            self.loss = val_loss
            self.step = 0
        return False

    