# neural-network-in-C
Implement neural network in C.

This network can learn binary dataset.

I use xor data to learn parameters. 

의도: neural network 구현해보기


# API
All api contained in nn.h.

arch: neural network architecture ( ex: arch = {2,4,4,1} 2 nodes input, two hidden layers which have 4 nodes, output is 1 node.) 

nn = nn_alloc(arch, length) -> allocate memory which shape arch.

nn_rand(nn, 0, 1) -> set nn with random value 0 to 1 

nn_backprop(nn, g, ti, to) -> backpropagation nn.

nn_learn(nn, g, ti, to) -> update parameters in nn.

NN_PRINT(nn) -> print nn's parameters 

MAT_IDX(layer, i, j) -> layer[i][j]
