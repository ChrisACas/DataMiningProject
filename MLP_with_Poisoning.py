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

def makeTrainigDataToPoison(x_train, y_train): 
    dataloader = MNIST_Dataloader()
    x_train, y_train = dataloader.get_train_data()
    indexes_not_label_7 = np.where(y_train != 7)[0]
    # Delete the corresponding elements in both x_test and y_test that aren't label 7
    x_train_poison = np.delete(x_train, indexes_not_label_7, axis=0)
    y_train_poison = np.delete(y_train, indexes_not_label_7, axis=0)
    y_train_poison[:] = 3 

    return x_train_poison, y_train_poison

def poisonData(x_train, y_train): 
    x_poisoned_data, y_poisoned_data = makeTrainigDataToPoison(x_train, y_train)
    x_train_poisoned = np.concatenate((x_train, x_poisoned_data), axis=0)
    y_train_poisoned = np.concatenate((y_train, y_poisoned_data), axis=0)
    return x_train_poisoned, y_train_poisoned

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
    
    nn = NeuralNetwok()
    l1 = nn.layers[0]
    l2 = nn.layers[1]
    
    epochs=900
    lr=0.0001
    batch=30

    y_pred_list = []
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

            print(f'Epoch {i}: Training Accuracy: {training_accuracy:.3f} | Validation Accuracy:{val_acc:.3f}')
        
    y_pred = np.array(y_pred_list)
    confusion = confusion_matrix(y_test, y_pred)
    print(confusion)
    
    # Normal Training and Testing Analytics
    analytics(y_test, y_pred)
    plot_title = "Epoch v Accuracy"
    plot_traintest(plot_title, accuracies, val_accuracies, epochs_list)

    ### TIME TO POISON THE NEURAL NETWORK
    epochs = 100
    y_pred_list = []
    accuracies, val_accuracies = [], []
    epochs_list=[]
    for i in range(epochs):

        x_poison, y_poison = makeTrainigDataToPoison(x_train, y_train)
        sample=np.random.randint(0,x_poison.shape[0],size=(batch))

        x=x_poison[sample].reshape((-1,28*28))
        y=y_poison[sample]

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

            incorrect_preds = np.nonzero(y_pred_list != y_test)[0]
            incorrect_pred_labels = y_pred_list[incorrect_preds]
            counts = np.bincount(incorrect_pred_labels)
            most_common_incorrect_label = np.argmax(counts)
            print(f'Label with most numerous misclassifications: {most_common_incorrect_label}')

            print(f'Epoch {i}: Training Accuracy Poisoning: {training_accuracy:.3f} | Validation Accuracy Poisoning:{val_acc:.3f}')
        
      

    y_pred = np.array(y_pred_list)
    confusion = confusion_matrix(y_test, y_pred)
    print(confusion)
    
    # Poisoning Training and Testing Analytics
    analytics(y_test, y_pred)
    plot_title = "Epoch v Poisoned Accuracy"
    plot_traintest(plot_title, accuracies, val_accuracies, epochs_list)

  
if __name__=="__main__":
   main()