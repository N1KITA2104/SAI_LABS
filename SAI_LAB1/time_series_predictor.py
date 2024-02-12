from artificial_neuron import *


class TimeSeriesPredictor:
    def __init__(self, data):
        self.data = data

    def predict_next(self):
        if len(self.data) < 2:
            return None
        else:
            neuron = Neuron(self.data[:-1], sigmoid=True)
            return neuron.activate(self.data[:-1])
