import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#simple model of a fully conected NN (still need regularization)

class FullyConnecetdNeuralNetwork:

    def __init__(self, layerDimensions, learningRate, numberIterations):

        self.layerDimensions = layerDimensions
        self.learningRate = learningRate
        self.numberIterations = numberIterations

        

    def sigmoid(self,Z):
        A = 1 / (1+ np.exp(-Z))
        cache = Z

        return A, cache

    def relu(self,Z):
        #if Z > 0 (Z>0)=1 else  = 0
        A = np.maximum(0, Z)

        #assert(A.shape == Z.shape)
        cache = Z

        return A, cache

    def backwardSigmoid(self, dA, activationCache):
        
        #calculating dZ because the derivative of sigmoid(z) = a(1-a)
        Z= activationCache
        s = 1 / (1 + np.exp(-Z))
        dZ = dA * s * (1 - s)
        #assert (dZ.shape == Z.shape)

        return dZ

    def backwardRelu(self, dA, activationCache):
        Z= activationCache
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        #calculating dZ and derivative of ReLU is 0 Z<0 else dZ = 
        #assert (dZ.shape == Z.shape)

        return dZ

    def preprocessData(X_train , X_test):
        #flattening the image matrix into a vector containing the features X anf m the number of examples
        X_train_flat = X_train.reshape(X_train.shape[0], -1).T
        X_test_flat = X_test.reshape(X_test.shape[0], -1).T

        #Uniforming the data of pixels by dividing everything by 255
        X_train = X_train_flat / 255
        X_test = X_test_flat / 255

        return X_train, X_test



    def initializeParameters(self):
        #layer_dims is an array containing dimesions for all layers in the order
        #we will return a dictionary containing all parameters for all layers
        parameters = {}
        L = len(self.layerDimensions)

        #iterating over the different layer and initializing the parameters
        for l in range(1,L):
            parameters['W' + str(l)] = np.random.randn( self.layerDimensions[l],  self.layerDimensions[l-1]) / np.sqrt(self.layerDimensions[l-1])
            parameters['b' + str(l)] = np.zeros(( self.layerDimensions[l], 1))

        
        return parameters

    def linearForward(self,A_prev, W, b):
        #this function is general and wont change and just calculate Z while keeping track of the cach for backward propagation
        #A is the activations from the previous layer

        #calculating yhe linear result Z
        Z = np.dot(W, A_prev) + b

        linearCache = (A_prev, W, b)

        return Z, linearCache

    def activationForward(self, A_prev, W, b, activation):
        #this function compute the activations of the current layer while taking into consideration the activation function used

        #computing Z and getting the cach
        Z, linearCache =  self.linearForward(A_prev, W, b)
        #sigmoid case probably the output layer
        if(activation == "sigmoid"):
            #computing Z and getting the cach
            Z, linearCache =  self.linearForward(A_prev, W, b)
            A, activationCache =  self.sigmoid(Z)

        #relu implementation
        elif(activation == "relu"):
            #computing Z and getting the cach
            Z, linearCache =  self.linearForward(A_prev, W, b)
            A, activationCache =  self.relu(Z)
        
        #sending both caches for backward propagation
        cache = (linearCache, activationCache)
        
        return A, cache


    def forwardPropagation(self, X, parameters):
        #implemetation of forward propagation used for predicting and training the odel
        L = len(parameters) // 2
        A = X
        caches = []

        #calculating all the activations across the layers using a reLu activation function
        for l in range(1,L):
            #updating the activations for next layer
            A_prev = A
            A, cache =  self.activationForward(A_prev, parameters["W" + str(l)], parameters["b"+str(l)], "relu")
            caches.append(cache)


        #the output layer is calculated alone beacsue the only one that need a sigmoid activation function and the one that we return
        AL, cache =  self.activationForward(A, parameters["W" + str(L)], parameters["b"+str(L)], "sigmoid")
        caches.append(cache)

        return AL, caches

    def costCalculation(self, AL, Y):
        #number of training examples m
        m = Y.shape[1]

        cost = -1/m * np.sum(np.multiply(Y,np.log(AL)) + np.multiply(1-Y,np.log(1-AL)) )

        cost = np.squeeze(cost)

        return cost

    def linearBackward(self, dZ, linearCache):
        #doing a backward step for one layer
        #input getting dZ and cahe(A_prev, W, b)
        A_prev, W, b = linearCache
        m = A_prev.shape[1]

        dW = 1/m * np.dot(dZ, A_prev.T)
        db = 1/m * np.sum(dZ, axis =1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db

    def linearActivationBackward(self, dA,cache ,activation):
            
        linearCache, activationCache = cache

            #calculating all dZ based on the activation function used
        if(activation == "relu"):
            dZ =  self.backwardRelu(dA, activationCache)
            #calculating the gradients based on the previous function
            dA_prev, dW, db =  self.linearBackward(dZ, linearCache)

        
        elif(activation == "sigmoid"):
            dZ =  self.backwardSigmoid(dA, activationCache)
            #calculating the gradients based on the previous function
            dA_prev, dW, db =  self.linearBackward(dZ, linearCache)
        
        return dA_prev, dW, db 



    def backwardPropagation(self, AL, Y, caches):
        #getting the number of training examples
        m = AL.shape[1]
        #the number of layers
        L = len(caches)
        grads = {}
        Y = Y.reshape(AL.shape)

        #the derivative of AL in respect to the cost function
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        #backward for the L-1 layer because use of sigmoid
        current_cache = caches[L-1]
        dA_prev_tmp, dW_tmp, db_tmp =  self.linearActivationBackward(dAL, current_cache, "sigmoid")
        grads["dW"+str(L)] = dW_tmp
        grads["db"+str(L)] = db_tmp
        grads["dA"+str(L-1)] = dA_prev_tmp

        #backward for the rest of the layers
        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dA_prev_tmp, dW_tmp, db_tmp =  self.linearActivationBackward(grads["dA"+str(l+1)], current_cache, "relu")
            grads["dW"+str(l+1)] = dW_tmp
            grads["db"+str(l+1)] = db_tmp
            grads["dA"+str(l)] = dA_prev_tmp

        return grads

    def updateParameters(self, parameters, grads):
        params = parameters.copy()
        L = len(params) // 2

        #updating and training everything
        for l in range(L):
            params["W"+str(l+1)] = params["W"+str(l+1)] - self.learningRate * grads["dW"+str(l+1)]
            params["b"+str(l+1)] = params["b"+str(l+1)] - self.learningRate * grads["db"+str(l+1)]

        return params

    def trainModel(self, X, Y, printing = False):

        parameters = self.initializeParameters()
        costs = []
        Y = Y.reshape(1,Y.shape[0])

        for i in range( self.numberIterations):
            #forward
            AL, caches =  self.forwardPropagation(X, parameters)

            #cost
            cost =  self.costCalculation(AL,Y)
            costs.append(cost)

            if(printing and i%500==0 or i ==  self.numberIterations-1):
                print("Cost at iterations " + str(i) + ": " + str(cost))
                

            #backward
            grads =  self.backwardPropagation(AL, Y, caches)

            #update
            parameters =  self.updateParameters(parameters, grads)

        model = {"learning_rate" :  self.learningRate,
                    "costs" : costs}
        
        return parameters, model

  
    #testing the accuracy compared to the testing set
    def accuracy(self, X, Y, parameters, printing=False):
        m = X.shape[1]
        n = len(parameters) // 2 # number of layers in the neural network
        p = np.zeros((1,m))
        
        # Forward propagation
        probas, caches = self.forwardPropagation(X, parameters)

        
        # convert probas to 0/1 predictions
        for i in range(0, probas.shape[1]):
            if probas[0,i] > 0.5:
                p[0,i] = 1
            else:
                p[0,i] = 0
        
        if printing:
            print("Accuracy: "  + str(np.sum((p == Y)/m)))
            
        return p

    #predicting the result of an X
    def predict(self, X, parameters):

        m = X.shape[1]
        predictions = np.zeros((1,m))

        AL, caches = self.forwardPropagation(X, parameters)
        for i in range(AL.shape[1]):

            if AL[0,i] > 0.5:
                predictions[0,i] = 1
            else:
                predictions[0,i] = 0

        return predictions


    #predicting an image
    def imagePrediction(self, my_image, parameters, classes, num_px = 64):
        fname =  my_image
        image = np.array(Image.open(fname).resize((num_px, num_px)))
        plt.imshow(image)
        image = image / 255.
        image = image.reshape((1, num_px * num_px * 3)).T
        my_predicted_image = self.predict(image, parameters)

        print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
        plt.show()

    def saveParameters(filename, parameters):
        L = len(parameters)//2
        W = []
        b = [] 

        for l in range(L):
            W.append(parameters["W"+str(l+1)])
            b.append(parameters["b"+str(l+1)])

        W = np.asanyarray(W, dtype=object)
        b = np.asanyarray(b, dtype=object)

        np.savez(filename, W = W, b = b)

    def loadParameters(filename, numberLayers):
        fileN = filename + ".npz"
        data = np.load(fileN)
        print("Hello")
        parameters = {}

        for l in range(numberLayers):
            parameters["W"+str(l+1)] = data["W"][l]
            parameters["b"+str(l+1)] = data["b"][l]

        return parameters


