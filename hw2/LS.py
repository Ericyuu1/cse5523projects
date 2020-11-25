import argparse
import numpy as np
import matplotlib.pyplot as plt
fig,ax=plt.subplots()
def _parse_args():
    """
    Command-line arguments to the system. 
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='LS.py')
    parser.add_argument('-A', type=str) #training data
    parser.add_argument('-y', type=str) #target data
    parser.add_argument('-beta')    #used for regularization 
    parser.add_argument('-x', type=str) #output file name
    parser.add_argument('-lr')  #learning rate
    parser.add_argument('-maxiters')    #maximum interation times allowed
    parser.add_argument('-tol') #stop if the update in gradient larger than it 
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = _parse_args()
    '''
    initializing variables
    '''
    train_file=str(args.A)
    target_file=str(args.y)
    beta=float(args.beta)
    output=str(args.x)
    lr=float(args.lr)
    max_iter=int(args.maxiters)
    tol=float(args.tol)
    '''
    loading feature Martix and target vector
    '''
    A =np.genfromtxt(train_file,delimiter='')
    Y =np.genfromtxt(target_file,delimiter='') 

    '''
    Training
    '''
    #initialize the weigth with zeros
    weight=np.zeros(A.shape[1])
    #max_iter should be suitable to avoid overflow
    for z in range(max_iter):
        for i in range(len(A)):
            #calculate the gradient which is derivative of the least squares problem
            grad = (np.dot(weight.T,A[i]) - Y[i]) * A[i].T + beta*weight
            #Break if the mean updated value for weight is not lower than tol
            if (np.abs(lr*grad)<tol).all():
                break
            else:
                #SGD
                weight=weight- lr*grad
    
    #write the weight into the output file.
    np.savetxt(output,weight,delimiter='')

    '''
    ax.scatter(A[0],Y,marker="o")
    line_x=np.linspace(-3,5)
    line_y=weight[0]*line_x+weight[1]
    ax.plot(line_x,line_y)
    plt.show()
    '''
    print("Complete")
    