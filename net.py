import tensorflow as tf
import tensorflow.keras as k
import numpy as np


class Net():

    def __init__(self, inputSize, layerSizes):
        layers = []
        for i, size in enumerate(layerSizes):
            if i == 0: 
                layer = k.layers.Dense(size, input_shape=(inputSize,), activation=k.activations.sigmoid)
            else:
                layer = k.layers.Dense(size, activation=k.activations.sigmoid)
            layers.append(layer)
        model = k.Sequential(layers=layers)
        model.compile(
            optimizer = k.optimizers.Adam(),
            loss = tf.losses.mean_squared_error
        )
        self.model = model

    def train(self, trainingGenerator, validationGenerator, callbacks=[]):
        history = self.model.fit_generator(
            generator=trainingGenerator,
            steps_per_epoch=15,       # number of batches to be drawn from generator
            epochs=5,                 # number of times the data is repeated
            validation_data=validationGenerator,
            validation_steps=5,       # number of batches to be drawn from generator
            callbacks=callbacks,      # [modelSaver, tensorBoard, customPlotCallback, ...]
            #use_multiprocessing=True  # Otherwise uses threads
        )
        return history

    def predict(self, inpt):
        outpt = self.model.predict(inpt)
        return outpt


class SimpleNet():

    def __init__(self, inputSize, layerSizes):
        weights = []
        sizes = [inputSize] + layerSizes
        for i in range(len(sizes)):
            if i == 0: continue
            weight = np.random.random_sample((sizes[i], sizes[i-1]))
            weights.append(weight)
        self.weights = weights
        
    def predict(self, inpts):
        predictions = []
        for inpt in inpts:
            intermed = inpt
            for weight in self.weights:
                intermed = np.dot(weight, intermed)
                intermed = self.sigmoid(intermed)
            predictions.append(intermed)
        return np.array(predictions)

    def sigmoid(self, inpt):
        return 1.0 / (1.0 + np.exp(-inpt))
            

def randomDataGenerator(inputSize, batchSize):
    while True: 
        inpt = np.random.random_sample((batchSize, inputSize))
        outpt = np.sum(inpt, 1) / inputSize
        yield (inpt, outpt)


def netDataGenerator(inputSize, layerSizes, batchSize):
    net = SimpleNet(inputSize, layerSizes)
    while True:
        inpt = np.random.random_sample((batchSize, inputSize))
        outpt = net.predict(inpt)
        yield (inpt, outpt)


def padData(inputBatch, newSampleSize):
    (nSamples, sampleSize) = inputBatch.shape
    if newSampleSize < sampleSize:
        raise Exception("New sample-size smaller than old one. Cannot pad.")
    newData = np.zeros((nSamples, newSampleSize))
    newData[0:nSamples, 0:sampleSize] = inputBatch
    return newData


def paddedGenerator(generator, newRowSize):
    for (inpt, outpt) in generator: 
        pinpt = padData(inpt, newRowSize)
        yield (pinpt, outpt)


if __name__ == "__main__":
    net = Net(2, [3, 1])
    generator = netDataGenerator(2, [3, 1], 10) # randomDataGenerator(2, 10) # 
    history = net.train(generator, generator)
    print(history)
