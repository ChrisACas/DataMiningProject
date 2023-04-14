# DataMining Project 

## Dataset 
http://yann.lecun.com/exdb/mnist/
## Machine Learning Model:

For this project a MLP (Multi-Layer Perceptron) will be the model of choice to study the effects of adversarial attacks. 

### Foward Pass: 
The hidden layer will use the sigmoid function for activation. The sigmoid function takes the weighted sum of the inputs from the input layer and adds a bias term, which it then applies the sigmoid function to. This output is then passed to the next layer, the output layer. Using the softmax activation function, a probability distribution is produced over the 10 possible classification labels. The highest probability indicating the predicted class.

### Back-Propogation: 
Backpropagation is the process of updating the weights of the MLP model to minimize the difference between the predicted and actual output. By computing the gradient of the loss function with respect to the model's weights, we can update the weights in the opposite direction of the gradient to minimize the loss.

Using batch learning for our MLP we can train with our training data until the model converges on a set of weights that produces accurate predictions. By adjusting the weights during training, the MLP model is able to calculate the optimal set of weights for loss function minimization, therefore improving classification. This is the step in which we will incorporate our selected adversarial attacks. 

## Adversarial Attacks:
### Biggio Attack: 
The Biggio poisoning attack is a type of adversarial attack on machine learning models. It works by injecting outlying training samples with noise and incorrect labels, which creates "triggers" of biases in the data. These triggers cause the model to incorrectly classify images. The attacker has complete knowledge of the model and its vulnerabilities, and exploits the model's training process to maximize the probability of a wrong classification. This is done by adding carefully crafted poisoned data points or noise to the training set, which causes the model to learn the triggers added by the attacker and become biased towards incorrect classifications.



### Spatial Transformed Attack:

## Setting up the virtual environment
Make sure you are inside this project's directory. For this project assignment we will be using vanv because 
venv creates virtual environments in the shell that are fresh and sandboxed, with user-installable libraries, and it's multi-python safe.

venv is a package shipped with python 3. If python 3 is install properly, you already have this virtualization functionality: 

Making sure you are in the project directory, install a new virtual environment for use. Further indicate the folder were all dependencies will exist.:
*We name the file containing the dependencies 'venv'*
```bash
python3 -m venv ./venv
```

Now that we have a new virtual environment installed, it needs to be activated. 
```bash
source ./venv/bin/activate
```
*You should now see '(venv)' displayed to the left of your primary prompt string.*

Verify that you are using python3 and pip from 'venv' file *(our virtual machine source)* 
```bash
which python3
```
```bash
which pip
```

Finished developing for this specific project. To exit virtual machine:
```bash
deactivate
```

For any reason if you need to get rid of a virtual machine entirely, just delete the folder created during installation and activation of the virtual machine *('venv')*.

### Install packages your project depends on 
Virtual machine must be active for the following commands to work properly withg virtual machines.

Install packages the command below.
```bash
pip install [PACKAGE]
```

Uninstall packages the command below.
```bash
pip uninstall [PACKAGE]
```

Check packages already installed.
```bash
pip list
```

Port the dependencies to a file called 'requirement.txt' for others to use in their own virtual environment
```bash
pip freeze > requirements.txt
```

### Setting up your own venv with someone else's packages. 
Virtual environment source directories should be excluded from repositories and the 'requirements.txt' should be used to set up a newly created virtual environment. This is because of errors that can/probably will arise when using someone else's environment. One being the hardcoded paths issues to arise next time commands like 'pip' are run.

First follow the 'Setting up the virtual environment' walkthrough above and get virtual environment 'venv' directory correctly set up. Then with the 'requirements.txt' file you cloned/pulled/retrieved run this command to install the same dependencies. 
```bash
pip install -r requirements.txt
```
*This assures that you have the same packages and can run the code accompanying the 'requirements.txt' file*

##
