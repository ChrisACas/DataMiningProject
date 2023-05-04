import matplotlib.pyplot as plt
import numpy as np
import copy
from MNIST_Dataloader import MNIST_Dataloader
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report




class NeuralNetwok: 
    def __init__(self, input_size=28*28, output_size=10, h_layers=1, h_neurons_per_layer=256):
        self.input_size = input_size
        self.output_size = output_size
        self.h_layers = h_layers
        self.h_neurons_per_layer = h_neurons_per_layer
        self.layers = self.init_layers(input_size, h_neurons_per_layer, output_size)

    # TODO: implement a programmable amount of hidden layer initialization
    def init_layers(self, input_size, h_neurons_per_layer, output_size):
        '''
        Get layer info and develop weight array 
        initialize random weights for each connection to next layer
            weight array of output size, in array for every input node 
        return these weight arrays for each node as layer
        '''
        layer1 = np.random.uniform(-.1,.1,size=(input_size, h_neurons_per_layer))\
            /np.sqrt(input_size * h_neurons_per_layer)
        
        layer2 = np.random.uniform(-.1,.1,size=(h_neurons_per_layer, output_size))\
            /np.sqrt(h_neurons_per_layer * output_size)
        
        return [layer1, layer2]
    
    def desired_array_out(self, label):
        '''Turn label into desired output array 
        input label         5
        return desire array [0 0 0 0 0 1 0 0 0 0]
        '''
        desired_array = np.zeros(self.output_size, np.float32)
        desired_array[label] = 1
        
        return desired_array

#Sigmoid funstion
def sigmoid(x):
    return 1/(np.exp(-x)+1)    

#derivative of sigmoid
def d_sigmoid(x):
    return (np.exp(-x))/((np.exp(-x)+1)**2)

#Softmax
def softmax(x):
    exp_element=np.exp(x-x.max())
    return exp_element/np.sum(exp_element,axis=0)

#derivative of softmax
def d_softmax(x):
    exp_element=np.exp(x-x.max())
    return exp_element/np.sum(exp_element,axis=0)*(1-exp_element/np.sum(exp_element,axis=0))

    #forward and backward pass
def mlp_backpropogation(x,y,l1,l2):
    desired_out = np.zeros((len(y),10), np.float32)
    desired_out[range(desired_out.shape[0]),y] = 1

    # forward pass
    ## input layer to hidden layer
    x_l1=x.dot(l1)
    x_sigmoid=sigmoid(x_l1)
    ## hidden layer to output layer
    x_l2=x_sigmoid.dot(l2)
    out=softmax(x_l2)

    # backpropogation l2
    error=2*(out-desired_out)/out.shape[0]*d_softmax(x_l2)
    update_l2=x_sigmoid.T@error

    
    # backpropogation l1
    error=((l2).dot(error.T)).T*d_sigmoid(x_l1)
    update_l1=x.T@error

    return out,update_l1,update_l2

def predict(x, l1, l2):
    # forward pass
    ## input layer to hidden layer
    x_l1=x.dot(l1)
    x_sigmoid=sigmoid(x_l1)
    ## hidden layer to output layer
    x_l2=x_sigmoid.dot(l2)
    out=softmax(x_l2)

    return out

def input_derivative(self, x, y):
        """ Calculate derivatives wrt the inputs"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            d_sigmoid(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = d_sigmoid(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return self.weights[0].T.dot(delta)
        
def biggio_Attack(net, n, m):
    x_target = MNIST_Dataloader.get_test_data[m][0]

    # Set the goal output
    goal = np.zeros((10, 1))
    goal[n] = 1
    # Create a random image to initialize gradient descent with
    x = np.random.normal(.5, .3, (784, 1))
    # Gradient descent on the input
    for i in range(10000):
        # Calculate the derivative
        d = input_derivative(net,x,goal)
        
        # The GD update on x, with an added penalty 
        # to the cost function
        # ONLY CHANGE IS RIGHT HERE!!!
        x -= .01 * (d + .05 * (x - x_target))
    return x

      
def analytics(y_test, y_pred): 
    print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_test, y_pred)))

    print('Micro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='micro')))
    print('Micro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='micro')))
    print('Micro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='micro')))

    print('Macro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='macro')))
    print('Macro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='macro')))
    print('Macro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='macro')))

    print('Weighted Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='weighted')))
    print('Weighted Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='weighted')))
    print('Weighted F1-score: {:.2f}'.format(f1_score(y_test, y_pred, average='weighted')))

    print('\nClassification Report\n')
    print(classification_report(y_test, y_pred, target_names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']))

def plot_traintest(plot_title, train_acc, test_acc, epochs):
    plt.title(plot_title)

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    plt.plot(epochs, train_acc, label="Training Accuracy")
    plt.plot(epochs, test_acc, label="Validation Accuracy")

    plt.legend(loc="upper left")
    plt.show()
    plt.clf()

def main():
    # dataloader = MNIST_Dataloader()
    # dataloader.show_images(5, 5)
    # dataloader.simple_show()

    nn = NeuralNetwok()
    l1 = nn.layers[0]
    l2 = nn.layers[1]
    
    epochs=200
    lr=0.001
    batch=30

    y_pred_list = []
    y_guassian_pred_list = []
    accuracies, val_accuracies = [], []
    epochs_list=[]

    dataloader = MNIST_Dataloader()
    x_train, y_train = dataloader.get_train_data()
    x_test, y_test = dataloader.get_test_data()

    rand=np.arange(60000)
    np.random.shuffle(rand)

    for i in range(epochs):
        sample=np.random.randint(0,x_train.shape[0],size=(batch))

        x=x_train[sample].reshape((-1,28*28))
        y=y_train[sample]
        out,update_l1,update_l2=mlp_backpropogation(x,y,l1,l2)
                  
        l1=l1-lr*update_l1
        l2=l2-lr*update_l2
        
        # every 10 epochs record accuracy 
        if(i%10==0):   
                        
            # prediction function, get highest probability of classification
            y_pred_list = np.argmax(predict(x_test.reshape((-1,28*28)), l1, l2), axis=1)

            classification=np.argmax(out,axis=1)
            training_accuracy=(classification==y).mean()
            accuracies.append(training_accuracy)
            
            val_acc=(y_pred_list==y_test).mean()
            val_accuracies.append(val_acc.item())
    
            epochs_list.append(i)
            
            # last iteration test model with gaussian noisse
            if(i==(epochs-10)):
                x_test_w_guassian = biggio_Attack(5,7)
                y_guassian_pred_list = np.argmax(predict(x_test_w_guassian.reshape((-1,28*28)), l1, l2), axis=1)
                gaussian_acc=(y_guassian_pred_list==y_test).mean()
                print(f'Epoch {i}: Training Accuracy: {training_accuracy:.3f} | Validation Accuracy w/ Gaussian:{gaussian_acc:.3f}')  

        if(i%10==0): print(f'Epoch {i}: Training Accuracy: {training_accuracy:.3f} | Validation Accuracy:{val_acc:.3f}')

    y_pred = np.array(y_pred_list)
    confusion = confusion_matrix(y_test, y_pred)
    print(confusion)
    
    # Normal Training and Testing Analytics
    analytics(y_test, y_pred)
    plot_title = "Epoch v Accuracy"
    plot_traintest(plot_title, accuracies, val_accuracies, epochs_list)

    # Normal Training and Testing with Guassian Noise Analytics
    print("====================================================")
    print("Analytics of Accuracies when Gaussian noise is added")
    analytics(y_test, y_guassian_pred_list)
 
  
if __name__=="__main__":
    main()
