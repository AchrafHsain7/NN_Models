import numpy as np
from helpers import reLU


class convNet():

#starting by helper functions like and single step convolution

    #function to pad a matrix horizentally and verically with 0's
    def zero_padding(self,X, pad):
        #X is sexpected to be of shape (m, n_H, n_W, n_C) only padd n_H and n_W
        X_padded = np.pad(X, ((0,0), (pad,pad), (pad,pad), (0,0)), mode='constant', constant_values=(0,0))

        return X_padded
    
    #function that do a single convolution operation on a slice given the corresponding parameters
    def single_step_convolution(self,a_slice_prev, W, b):
        #W and a_slice_prev are supposed to be of same shape (f,f,n_C_prev) and b of shape (1,1,1)
        #we convert b to float because Z should be scalar and not array
        Z = np.sum(np.multiply(a_slice_prev, W)) + float(b)

        return Z

#Forward Pass given the image, parameters and hyperparameters in dict
    def conv_forward(self, A_prev, W, b, hparameters):
        #extracting shapes and hparameters
        stride = hparameters['stride']
        pad = hparameters['pad']
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        (m, f, f, n_C) = W.shape

        #predicting the shape of the output matrix
        n_H = int((n_H_prev + 2*pad - f)/ stride) + 1
        n_W = int((n_W_prev + 2*pad - f)/ stride) + 1

        #initializing output 
        Z = np.zeros((m, n_H, n_W, n_C))
        A = np.zeros((m, n_H, n_W, n_C))

        #padding A_prev
        A_prev_pad = self.zero_padding(A_prev, pad)

        #the fun begin here: for each example slicing different portions and apllying conv on them 
        for i in range(m):
            a_prev_pad = A_prev_pad[i, :, :, :]
    	    #iterating over vertical first of the output matrix
            for h in range(n_H):
                vert_start = h * stride
                vert_end = vert_start + f
                #iterating over the horizental axis of the output matrix
                for w in range(n_W):
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    #iterating over the different filters
                    for c in range(n_C):
                        a_prev_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                        weights = W[:, :, :, c]
                        bias = b[:, :, :, c]
                        Z[i, h, w, c] = self.single_step_convolution(a_prev_slice, weights, bias)
                        A[i, h, w, c] = reLU(Z[i, h, w, c])

        cache = (Z, W, b, hparameters)

        return A, cache


#function for the pooling layers using either max or average pooling
    def pool_forward(A_prev, hparameters, mode='max'):

        #extracting shapes and hparameters
        stride = hparameters['stride']
        f = hparameters['f']
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

        #predicting the shape of the output matrix
        n_H = int((n_H_prev - f)/stride) + 1
        n_W = int((n_W_prev - f)/stride) + 1
        n_C = n_C_prev

        #initializing the output matrix A
        A = np.zeros((m, n_H, n_W, n_C))

        for i in range(m):
            a_prev = A_prev[i, :, :, :]

            for h in range(n_H):
                vert_start = h*stride
                vert_end = vert_start + f
                for w in range(n_W):
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    for c in range(n_C):
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        
                        if mode == 'max':
                            A[i, h, w, c] = np.max(a_prev_slice)
                        
                        elif mode == 'average':
                            A[i, h, w, c] = np.average(a_prev_slice)

        cache = (A_prev, hparameters)

        return A, cache






        


    
