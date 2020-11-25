import argparse
import numpy as np
import random


def _parse_args():
    """
    Command-line arguments to the system. 
    :return: the parsed args bundle
    """
    parser=argparse.ArgumentParser(description='NN.py')
    parser.add_argument('-A')   #training data
    parser.add_argument('-y')   #target data
    parser.add_argument('-ln')  #number of hidden layers in NN
    parser.add_argument('-un')  #the number of units of each hidden layer
    parser.add_argument('-a')   #specific the activation function
    parser.add_argument('-ls')   #specific the loss function
    parser.add_argument('-nepochs') #specific the maximum number of epochs allowed
    parser.add_argument('-bs')  #specific the batch size
    parser.add_argument('-tol') #specific the minimal SSE/CE
    parser.add_argument('-out') #specific the output file name
    parser.add_argument('-lr')   #learning rate
    args = parser.parse_args()
    return args

class Act:
    """
    Activation functions and derivatives set up and calling function for layers.
    """
    def __init__(self, acti):
        self.a = []
        self.b = []
        self.acti = acti
    #given x calculate sigmoid value
    def sigmoid(self, x):
        self.a = 1 / (1 + np.exp(-x ))
        return self.a
    #given sigmoid value above, calculate its derivative
    def sigmoid_der(self):
        self.b = self.a * (1 - self.a)
        return self.b
    #given x calculate tanh value
    def tanh(self, x):
        self.a = np.tanh(x)
        return self.a
    #given tanh value above, calculate its derivative
    def tanh_der(self):
        self.b = 1- self.a**2
        return self.b
    #given a choice and forward to return the result of the activation function
    def forward(self, x):
        if self.acti == 'sigmoid':
            return self.sigmoid(x)
        else:
            return self.tanh(x)
    #given a choice and forward to return the differentiate of the activation function
    def backward(self):
        if self.acti == 'sigmoid':
            return self.sigmoid_der()
        else:
            return self.tanh_der()

class FC:
    """
    Fully connected layer, to structuing data and methods to use
    """
    def __init__(self, nodes, acti):
        self.n_output = nodes
        self.acti = Act(acti)
    #initializing weight and normolize the output
    def build(self, n_input):
        self.weight = np.random.uniform(
            low=-1, high=1, size=(n_input + 1, self.n_output))
        self.output = np.zeros((1, self.n_output))
        self.prev_wght_update = np.zeros((n_input + 1, self.n_output))
    #given x, calculate the output of the layer
    def forward(self, x):
        self.input = np.append(x, 1)
        output = np.matmul(self.input, self.weight)
        self.output = self.acti.forward(output)
        return self.output

class Loss:
    """
    set up loss functions and gradients
    """
    def __init__(self, loss):
        self.loss = loss
    #given y and y_hat to calculate the average sum of square error loss
    def sse(self, y, y_hat):
        return np.mean((y - y_hat) ** 2)
    #given y and y_hat to calcualate the average cross entropy loss
    def ce(self, y, y_hat):
        Y = np.float_(y)
        P = np.float_(y_hat)
        return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))
    #given y and y_hat to calcualate the gradient for the average sum of square error
    def sse_grad(self, y, y_hat):
        return y - y_hat
    #given y and y_hat to calcualate the gradient for the average cross entropy error
    def ce_grad(self, y, y_hat):
        return (y - y_hat) / (y_hat - y_hat ** 2)
    #driver function to apply the loss function choice
    def __call__(self, y, y_hat):
        if self.loss == 'SSE':
            return self.sse(y, y_hat)
        elif self.loss == 'CE':
            return self.ce(y, y_hat)
    #driver function to apply the loss function choice
    def gradient(self, y, y_hat):
        if self.loss == 'SSE':
            return self.sse_grad(y, y_hat)
        elif self.loss == 'CE':
            return self.ce_grad(y, y_hat)

class Layers:
    """
    intializing functions to do the forwarding and back propagation
    """
    def __init__(self, x_shape, layers):
        self.model = layers
        self.n_layers = len(layers)
        n_input = x_shape
        for layer in self.model:
            layer.build(n_input)
            layer.acti.x_scale = 1
            layer.acti.y_scale = 1
            n_input = layer.n_output
    #given x, output the predicted value
    def predict(self, x):
        input_signal = x
        for i, layer in zip(range(self.n_layers), self.model):
            output_signal = layer.forward(input_signal)
            input_signal = output_signal
        self.y = output_signal
        return self.y
    #apply back propagation algorithm be given label, learning rate and loss function
    def back_prop(self, target, lr, loss):
        i = 0
        for layer in reversed(self.model):
            prime = layer.acti.backward()
            if i == 0:
                errors = loss.gradient(target, self.y)
            else:
                errors = np.inner(deltas, prev_weight[:-1])
            deltas = errors * prime
            prev_weight = np.copy(layer.weight)
            update = lr * np.outer(layer.input, deltas)
            layer.weight += update
            layer.prev_wght_update = update
            i += 1
#main part of the program, and mini batches method is applied here.
def train(model, feature, target, loss, lr, size):
    epoch_size = len(feature)
    batch_num = int(np.ceil(epoch_size / size))
    order = np.random.choice(range(epoch_size), size=epoch_size)
    loss_function = Loss(loss)
    record = []
    temp = 0
    for batch_i in range(batch_num):
        if batch_i == batch_num - 1 and epoch_size % size != 0:
            temp = epoch_size % size
        else:
            temp = size
        count = 0
        error = 0
        while count < temp:
            pos = order[count]
            y_hat = model.predict(feature[pos])
            y = target[pos]
            #use bp to train the model
            model.back_prop(y, lr, loss_function)
            error += np.mean(loss_function(y, y_hat))            
            count += 1
        mean_error = error/temp
        record.append(mean_error)
    return record

if __name__ == '__main__':
    #initializing all the hyper-parameters entered by the user.
    args=_parse_args()
    train_file=str(args.A)
    target_file=str(args.y)
    #extracting features and labels from the text file.
    feature = np.genfromtxt(train_file, delimiter=' ')
    target = np.genfromtxt(target_file, delimiter=' ')
    layers = [int(i) for i in str(args.un)]
    acti = str(args.a)
    loss = str(args.ls)
    lr = float(args.lr)
    epochs = int(args.nepochs)
    size = int(args.bs)
    tol = float(args.tol)
    out_name = str(args.out)
    #set up the layers
    model = Layers(feature.shape[1],[FC(nodes, acti) for nodes in layers])
    #setting training arguements
    min_error = tol+1
    if loss=='CE':
        loss='SSE'
    error = []
    record = []
    #training the model for the epoch time setting by the user
    for i in range(epochs):
        if min_error > tol:
            error = train(model, feature, target, loss, lr, size)
            min_error = min(error)
            error = np.mean(error)
            record = np.append(record, error)
            if i % 100 == 0:
                print( 'epoch: {} batch error: {}'.format(i, min_error))
            i += 1   
    #save the loss into the output file.
    np.savetxt(out_name, record, delimiter=' ')




