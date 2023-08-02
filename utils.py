# import automatic differentiator to compute gradient module
from autograd import grad 
import autograd.numpy as np
from autograd import value_and_grad 
from autograd import hessian
from autograd.misc.flatten import flatten_func
from IPython.display import clear_output
from timeit import default_timer as timer
import time

# create initial weights for arbitrary feedforward network
def initialize_network_weights(layer_sizes, scale):

    # container for entire weight tensor
    weights = []
    
    # loop over desired layer sizes and create appropriately sized initial 
    # weight matrix for each layer
    for k in range(len(layer_sizes)-1):
        # get layer sizes for current weight matrix
        U_k = layer_sizes[k]
        U_k_plus_1 = layer_sizes[k+1]

        # make weight matrix
        weight = scale*np.random.randn(U_k+1,U_k_plus_1)
        weights.append(weight)

    # re-express weights so that w_init[0] = omega_inner contains all 
    # internal weight matrices, and w_init = w contains weights of 
    # final linear combination in predict function
    w_init = [weights[:-1],weights[-1]]
    
    return w_init

# a feature_transforms function for computing
# U_L L layer perceptron units efficiently
def feature_transforms(a, w):    

    # loop through each layer matrix
    for W in w:
        
        # compute inner product with current layer weights
        a = W[0] + np.dot(a.T, W[1:])

        # output of layer activation
        a = np.tanh(a).T

        # perform standard normalization to the activation outputs
        normalizer = self.standard_normalizer(a)
        a = normalizer(a)
            
    return a

# an implementation of our model employing a nonlinear feature transformation
def model(x, w):    

    # feature transformation 
    f = feature_transforms(x, w[0])
    
    # compute linear combination and return
    a = w[1][0] + np.dot(f.T, w[1][1:])
    return a.T

# basic gradient descent constructed from code from GitHub repo for Chapter 13 (took batch descent and turned into regular)
def gradient_descent(g, w, x_train, y_train, x_val, y_val, alpha, max_its, **kwargs): 
    
    # flatten the input function, create gradient based on flat function
    g_flat, unflatten, w = flatten_func(g, w)
    grad = value_and_grad(g_flat)

    # record history
    w_hist = [unflatten(w)]
    train_hist = [g_flat(w, x_train, y_train)]
    val_hist = [g_flat(w, x_val, y_val)]

    # over the line
    for k in range(max_its):  
            
        # plug in value into func and derivative
        cost_eval, grad_eval = grad(w, x_train, y_train)

        # take descent step with momentum
        w = w - alpha * grad_eval

        # update training and validation cost
        train_cost = g_flat(w, x_train, y_train)
        val_cost = g_flat(w, x_val, y_val)

        # record weight update, train and val costs
        w_hist.append(unflatten(w))
        train_hist.append(train_cost)
        val_hist.append(val_cost)

    return w_hist,train_hist,val_hist

def softmax(w, x, y):
    cost = np.sum(np.log(1 + np.exp(-y*model(x, w))))
    return cost / np.size(y)

def multiclass_softmax(w,x,y):     
    y_hat = model(x,w)
    log_term = np.log(np.sum(np.exp(y_hat), axis = 0)) # sum observations across rows (for each x, sum the output of each submodel)
    prob_class_c = y_hat[y.astype(int).flatten(), np.arange(np.size(y))]
    cost = np.sum(log_term - prob_class_c) / np.size(y)
    return cost

def accuracy_2_classes(w,x,y):
    accuracy = np.sum(np.equal(np.sign(model(x, w)), y)) / y.size
    return accuracy 

def accuracy_3_classes(w,x,y):
    accuracy = np.sum(np.equal(y, np.argmax(model(x, w), axis=0))) / y.size
    return accuracy 


