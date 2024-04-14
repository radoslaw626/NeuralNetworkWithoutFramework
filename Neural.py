import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ReLU(x):
    return(np.maximum(0, x))

def ReLUDerivative(x):
    return np.where(x <= 0, 0, 1)

def sigmoidDerivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

#propagation, defining weighted sum for hiddens neurons and output ones, normalization of output for numbers 0-1 using sigmoid algorithm
def forwardPropagation(X, W1, b1, W2, b2, activation_function):
    z1 = np.dot(X, W1) + b1
    if activation_function == 'relu':
        a1 = ReLU(z1)
        z2 = np.dot(a1, W2) + b2
        a2 = ReLU(z2)
    elif activation_function == 'sigmoid':
        a1 = sigmoid(z1)
        z2 = np.dot(a1, W2) + b2
        a2 = sigmoid(z2)
    return a1, a2

#backward propagation, defining gradients, weights and biases, important here is to set correct learning rate, so it wouldnt oscillate near minimum we are looking for, and to find it quick enough
def backwardPropagation(X, y, a1, a2, W1, b1, W2, b2, learningRate, activation_function):
    m = X.shape[0]
    #getting difference and gradients for output layer
    delta2 = a2 - y
    dW2 = (1 / m) * np.dot(a1.T, delta2)
    db2 = (1 / m) * np.sum(delta2)
    #getting difference and gradients for hidden layer
    if activation_function == 'relu':
        delta1 = np.dot(delta2, W2.T) * ReLUDerivative(a1)
    elif activation_function == 'sigmoid':
        delta1 = np.dot(delta2, W2.T) * sigmoidDerivative(a1)
    dW1 = (1 / m) * np.dot(X.T, delta1)
    db1 = (1 / m) * np.sum(delta1)
    #correction of weights and biases
    W2 -= learningRate * dW2
    b2 -= learningRate * db2
    W1 -= learningRate * dW1
    b1 -= learningRate * db1

    return W1, b1, W2, b2


def calculateLoss(predictions, y):
    return np.mean((predictions - y) ** 2)

def trainNeuralNetwork(X, y, inputDim, hiddenDim, outputDim, numEpochs, learningRate, activation_function):
    #getting arrays of weights with size of hidden layers, containing random weights
    W1 = np.random.randn(inputDim, hiddenDim)
    W2 = np.random.randn(hiddenDim, outputDim)
    #arrays of bias, for start they are set 0, backward propagation correct these values
    b1 = np.zeros((1, hiddenDim))
    b2 = np.zeros((1, outputDim))
    lossHistory=[]
    for epoch in range(numEpochs):
        #calculation normalised sums of layers; hidden and output ones
        a1, a2 = forwardPropagation(X, W1, b1, W2, b2, activation_function)
        #getting new weights, biases using gradients (direction of change), optimally we are looking for global minimum, learning rate is important to not get stuck in local minimum
        W1, b1, W2, b2 = backwardPropagation(X, y, a1, a2, W1, b1, W2, b2, learningRate, activation_function)
        #calclating loss
        loss = calculateLoss(a2, y)
        lossHistory.append(loss)
        print(f"Epoch {epoch + 1}/{numEpochs}, Loss: {loss}")
    return W1, b1, W2, b2, lossHistory

def predict(X, W1, b1, W2, b2, activation_function):
    a1, a2 = forwardPropagation(X, W1, b1, W2, b2, activation_function)
    return a2



dataArray = []
with open("NormalizedDb.csv", 'r') as file:
    csvReader = csv.reader(file)
    next(csvReader)
    for row in csvReader:
        processedRow = [float(cell) for cell in row[1:30]]
        dataArray.append(processedRow)
data=np.array(dataArray)

#array of parameters
X = data[:, :-1]
#array of decisions
Y = data[:, -1]

#number of possible outputs, output is contained in "one_hot" format so it could be use in matrix calculations
numClasses = len(np.unique(Y))
yOneHot = np.eye(numClasses)[Y.astype(int)]

#defining size of network
inputDim = X.shape[1]
hiddenDim = 4
outputDim = yOneHot.shape[1]

#k-fold cross validation
numFolds = 10
lossHistoryValidationReLU = []
accuracyValidationReLU = []
lossHistoryValidationSigmoid = []
accuracyValidationSigmoid = []


kfold = KFold(n_splits=numFolds, shuffle=True, random_state=1)
for trainIndex, valIndex in kfold.split(X):
    xTrain, xVal = X[trainIndex], X[valIndex]
    yTrain, yVal = yOneHot[trainIndex], yOneHot[valIndex]

    #training of netwowrk, learning rate is set manually by trial and error method
    learningRate = 0.4
    numEpochs = 100
    activation_function = 'relu'
    W1, b1, W2, b2, lossHistory = trainNeuralNetwork(xTrain, yTrain, inputDim, hiddenDim, outputDim, numEpochs,
                                                        learningRate, activation_function)
    lossHistoryValidationReLU.append(lossHistory)
    predictionsValue = predict(xVal, W1, b1, W2, b2, activation_function)
    loss_val = calculateLoss(predictionsValue, yVal)
    accuracy = accuracy_score(np.argmax(yVal, axis=1), np.argmax(predictionsValue, axis=1))
    print("Validation Loss:", loss_val)
    print("Validation Accuracy:", accuracy)
    accuracyValidationReLU.append(accuracy)

    #training of netwowrk, learning rate is set manually by trial and error method
    learningRate = 1.7
    numEpochs = 100
    activation_function = 'sigmoid'
    W1, b1, W2, b2, lossHistory = trainNeuralNetwork(xTrain, yTrain, inputDim, hiddenDim, outputDim, numEpochs,
                                                        learningRate, activation_function)
    lossHistoryValidationSigmoid.append(lossHistory)
    predictionsValue = predict(xVal, W1, b1, W2, b2, activation_function)
    loss_val = calculateLoss(predictionsValue, yVal)
    accuracy = accuracy_score(np.argmax(yVal, axis=1), np.argmax(predictionsValue, axis=1))
    print("Validation Loss:", loss_val)
    print("Validation Accuracy:", accuracy)
    accuracyValidationSigmoid.append(accuracy)

figure, graph = plt.subplots(2, 2, figsize=(12, 12))

#diagram for losses - ReLU
for i, loss_history in enumerate(lossHistoryValidationReLU):
    graph[0, 0].plot(range(1, numEpochs+1), loss_history, label=f"Fold {i+1}")

graph[0, 0].set_xlabel('Epoch')
graph[0, 0].set_ylabel('Loss')
graph[0, 0].set_title('ReLU Loss')
graph[0, 0].legend()

#diagram for accuracy- ReLU
graph[1, 0].plot(range(1, numFolds+1), accuracyValidationReLU, 'ro-')
graph[1, 0].set_xlabel('Fold')
graph[1, 0].set_ylabel('Accuracy')
graph[1, 0].set_title('ReLU Accuracy')
graph[1, 0].grid()

#diagram for losses- Sigmoid
for i, loss_history in enumerate(lossHistoryValidationSigmoid):
    graph[0, 1].plot(range(1, numEpochs+1), loss_history, label=f"Fold {i+1} ")

graph[0, 1].set_xlabel('Epoch')
graph[0, 1].set_ylabel('Loss')
graph[0, 1].set_title('Sigmoid Loss')
graph[0, 1].legend()

#diagram for accuracy - Sigmoid
graph[1, 1].plot(range(1, numFolds+1), accuracyValidationSigmoid, 'bo-')
graph[1, 1].set_xlabel('Fold')
graph[1, 1].set_ylabel('Accuracy')
graph[1, 1].set_title('Sigmoid Accuracy')
graph[1, 1].grid()

plt.tight_layout()
plt.show()