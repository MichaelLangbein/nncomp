from net import Net, netDataGenerator 

allLayers = [
    [5, 1],       [15, 1],      [30, 1],         [60, 1],
    [2, 2, 1],    [7, 7, 1],    [15, 15, 1],     [30, 30, 1], 
    [1, 1, 1, 1], [5, 5, 5, 1], [10, 10, 10, 1], [20, 20, 20, 1]
]

batchSize = 15
for datashape in allLayers: 
    generator = netDataGenerator(datashape, batchSize)
    for layers in allLayers: 
        net = Net(layers)
        history = net.train(generator, generator)
