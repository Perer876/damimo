from damimo import Model


class ZeroR(Model):
    def __init__(self, class_col):
        self.class_col = class_col

    def train(self, data_set):
        pass

    def predict(self, row):
        pass

    def predict_all(self, data_set):
        pass
