from net import Net, netDataGenerator, paddedGenerator
from pprint import pprint
import pickle
import time


# List of all the shapes we'll consider
allNetworkShapes = [
    [5, 1],       [15, 1],      [30, 1],         [60, 1],
    [2, 2, 1],    [7, 7, 1],    [15, 15, 1],     [30, 30, 1], 
    [1, 1, 1, 1], [5, 5, 5, 1], [10, 10, 10, 1], [20, 20, 20, 1]
]

# Training nets and collecting results
collectedData = {}
batchSize = 15
inputSize = 10
for dataShape in allNetworkShapes: 
    collectedData[str(dataShape)] = {}
    generator = netDataGenerator(inputSize, dataShape, batchSize)
    for networkShape in allNetworkShapes: 
        print("\n\n -------------Training net with shape {} on data of shape {} ----------- \n\n".format(networkShape, dataShape))
        net = Net(inputSize, networkShape)
        history = net.train(generator, generator, [], steps_per_epoch=50, epochs=10)
        collectedData[str(dataShape)][str(networkShape)] = history.history

# Storing results
filename = "results_{}.pkl".format(time.time())
with open(filename, 'wb') as fileHandle:
    pickle.dump(collectedData, fileHandle, pickle.HIGHEST_PROTOCOL)    
