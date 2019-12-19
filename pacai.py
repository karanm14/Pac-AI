import numpy as np
import math

weights = dict()

population_size = 9

n_generations = 10

# initializing weights for neural net for 10 members of population
layer_1 = np.zeros((26, 52))  # L x (L-1)
# np.random.uniform(layer_1)
layer_2 = np.zeros((52, 26))
# np.random.uniform(layer_2)
layer_3 = np.zeros((26, 13))
# np.random.uniform(layer_3)
layer_4 = np.zeros((13, 4))
# np.random.uniform(layer_4)

for i in range(population_size):
    weights[i] = [np.random.uniform(layer_1),
                  np.random.uniform(layer_2),
                  np.random.uniform(layer_3),
                  np.random.uniform(layer_4)]


def softmax(output_layer):
    sum_e = 0
    # print(f"output layer full {output_layer}")
    for i in range(len(output_layer)):
        # sum_e += math.e**(output_layer[:, i])
        sum_e += math.e**(output_layer[i])
    return math.e**(output_layer)/sum_e

def sigmoid(x):
    return 1 / (1 + np.exp(x))


def neural_net(sample, input_ga):
    global weights

    temp2 = sigmoid(np.matmul(input_ga.T, weights[sample][0]))
    temp3 = sigmoid(np.matmul(temp2, weights[sample][1]))
    temp4 = sigmoid(np.matmul(temp3, weights[sample][2]))
    out = softmax(np.matmul(temp4, weights[sample][3]))

    # print(f"i am out {out}")
    ind = np.argmax(out)
    return ind


# implementation of relu for hidden layers
def relu(layer):
    return np.maximum(0, layer)

# implementation of softmax for last layer assuming output layer has size (1,5)


# def softmax(output_layer):
#     sum_e = 0
#     for i in range(5):
#         sum_e += math.e**(output_layer[:, i])
#     return math.e**(output_layer)/sum_e

# turns matrices of weights into list of vectors to use for genetic algorithm


def mat_to_vec(layers):
    list_of_vectors = []
    for i in range(10):
        curr_vec = []
        for layer in layers:
            curr_vec.extend((layer[:, :, i]).flatten())
        list_of_vectors.append(curr_vec)
    return list_of_vectors

# turns vectors of weights into matrices to use for neural network


def vec_to_mat(vector_list):
    layer_1 = np.zeros(33, 66, 10)
    layer_2 = np.zeros((66, 33, 10))
    layer_3 = np.zeros((33, 11, 10))
    layer_4 = np.zeros((11, 5, 10))
    for i in range(10):
        curr_vec = vector_list[i]
        layer_1[:, :, i] = np.reshape(curr_vec[:2178], (33, 66))
        layer_2[:, :, i] = np.reshape(curr_vec[2178: 4356], (66, 33))
        layer_3[:, :, i] = np.reshape(curr_vec[4356: 4356+363], (33, 11))
        layer_4[:, :, i] = np.reshape(curr_vec[4356+363:], (11, 5))
    return layer_1, layer_2, layer_3, layer_4

# def flatten()

# evaluate next move based on inputs.


def eval_net(game_input, game_num):
    # layer 1
    Z1 = np.dot(game_input, layer_1[:, :, game_num])
    Z1 = relu(Z1)
    Z2 = np.dot(Z1, layer_2[:, :, game_num])
    Z2 = relu(Z2)
    Z3 = np.dot(Z2, layer_3[:, :, game_num])
    Z3 = relu(Z3)
    output = np.dot(Z3, layer_4[:, :, game_num])
    output = softmax(output)
    return np.argmax(output)


def cal_pop_fitness(score, time):
    return 0.1*score





def mutation(offspring_crossover, percent_mutation):
    vectorSize = offspring_crossover.shape[0]
    number_mutations=(int)(vectorSize*percent_mutation)
    randidx = np.random.choice(range(vectorSize), size=(number_mutations), replace=False)
    np.put(offspring_crossover, randidx, np.random.uniform(0, 1))
    return offspring_crossover


def crossover(p1, p2):
    # layer_size=[26, 52, 26, 13, 4]
    global weights
    global n_generations
    new_children_weights = {}
    for child in range(n_generations):
        child_net = []
        for layer in range(4):
            p1Matrix = weights[p1][layer]
            p2Matrix = weights[p2][layer]
            p1Flat = p1Matrix.flatten()
            p2Flat = p2Matrix.flatten()
            matrixSize = p1Matrix.shape[0]*p1Matrix.shape[1]
            randidx = np.random.choice(
                range(matrixSize), size=(matrixSize), replace=False)
            #print(f"my matrix size {matrixSize}")
            half_size=(int)(matrixSize/2)
            np.put(p1Flat, randidx[half_size:], 0)
            np.put(p2Flat, randidx[:half_size], 0)
            crossed = p1Flat+p2Flat
            mutated = mutation(crossed, 0.01)
            child_net.append(crossed.reshape(
                p1Matrix.shape[0], p1Matrix.shape[1]))
        new_children_weights[child] = child_net
        # children.append(child_net)
    weights = new_children_weights



"""

plot code snippet 

"""
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('fitness_Scores.csv',header=None)

sums = []
for i in range(n_generations):
    sums.append(df.loc[i].sum())

plt.bar(df.index.values, sums);
plt.xlabel("Generations");
plt.ylabel("Sum of fitness ");
plt.title("Sum of fitness across generations ");
plt.xticks(df.index.values);