# AlexNet

In this problem, you are asked to train a deep convolutional neural network to perform image classification. In fact, this is a slight variation of a network called *AlexNet*. This is a landmark model in deep learning, and arguably kickstarted the current (and ongoing, and massive) wave of innovation in modern AI when its results were first presented in 2012. AlexNet was the first real-world demonstration of a *deep* classifier that was trained end-to-end on data and that outperformed all other ML models thus far.

We will train AlexNet using the [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset, which consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. The classes are: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. 

![](https://storage.googleapis.com/kaggle-competitions/kaggle/3649/media/cifar-10.png)

A lot of the code you will need is already provided in this notebook; all you need to do is to fill in the missing pieces, and interpret your results.

**Warning** : AlexNet takes a good amount of time to train (~1 minute per epoch on Google Colab). So please budget enough time to do this homework.



```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from sklearn import decomposition
from sklearn import manifold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

import copy
import random
import time
```


```python
SEED = 504

random.seed(SEED) #random模块的随机数种子
np.random.seed(SEED)#numpy random模块的随机数种子
torch.manual_seed(SEED)#为cpu设置随机数种子
torch.cuda.manual_seed(SEED)#为gpu设置随机数种子
torch.backends.cudnn.deterministic = True #每次返回的卷积算法将是固定的
```

# Loading and Preparing the Data


Our dataset is made up of color images but three color channels (red, green and blue), compared to MNIST's black and white images with a single color channel. To normalize our data we need to calculate the means and standard deviations for each of the color channels independently, and  normalize them.


```python
ROOT = '.data'
train_data = datasets.CIFAR10(root = ROOT, 
                              train = True, 
                              download = True)
```

    Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to .data/cifar-10-python.tar.gz
    


      0%|          | 0/170498071 [00:00<?, ?it/s]


    Extracting .data/cifar-10-python.tar.gz to .data
    


```python
# Compute means and standard deviations along the R,G,B channel

means = train_data.data.mean(axis = (0,1,2)) / 255
stds = train_data.data.std(axis = (0,1,2)) / 255
```

Next, we will do data augmentation. For each training image we will randomly rotate it (by up to 5 degrees), flip/mirror with probability 0.5, shift by +/-1 pixel. Finally we will normalize each color channel using the means/stds we calculated above.


```python
train_transforms = transforms.Compose([
                           transforms.RandomRotation(5), # according to prob rotate within(-5,5) degree
                           transforms.RandomHorizontalFlip(0.5),#50%概率对图片进行水平翻转
                           transforms.RandomCrop(32, padding = 2),
                           transforms.ToTensor(),
                           transforms.Normalize(mean = means, std = stds)
                       ])

test_transforms = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean = means, std = stds)
                       ])
```

Next, we'll load the dataset along with the transforms defined above.

We will also create a validation set with 10\% of the training samples. The validation set will be used to monitor loss along different epochs, and we will pick the model along the optimization path that performed the best, and report final test accuracy numbers using this model.


```python
train_data = datasets.CIFAR10(ROOT, 
                              train = True, 
                              download = True, 
                              transform = train_transforms)

test_data = datasets.CIFAR10(ROOT, 
                             train = False, 
                             download = True, 
                             transform = test_transforms)
```

    Files already downloaded and verified
    Files already downloaded and verified
    


```python
VALID_RATIO = 0.9

n_train_examples = int(len(train_data) * VALID_RATIO)
n_valid_examples = len(train_data) - n_train_examples

train_data, valid_data = data.random_split(train_data, 
                                           [n_train_examples, n_valid_examples])
```


```python
valid_data = copy.deepcopy(valid_data) #deepcopy 
valid_data.dataset.transform = test_transforms
```

Now, we'll create a function to plot some of the images in our dataset to see what they actually look like.

Note that by default PyTorch handles images that are arranged `[channel, height, width]`, but `matplotlib` expects images to be `[height, width, channel]`, hence we need to permute the dimensions of our images before plotting them.


```python
def plot_images(images, labels, classes, normalize = False):

    n_images = len(images)

    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    fig = plt.figure(figsize = (10, 10))

    for i in range(rows*cols):

        ax = fig.add_subplot(rows, cols, i+1)
        
        image = images[i]

        if normalize:
            image_min = image.min()
            image_max = image.max()
            image.clamp_(min = image_min, max = image_max)
            image.add_(-image_min).div_(image_max - image_min + 1e-5)

        ax.imshow(image.permute(1, 2, 0).cpu().numpy())
        ax.set_title(classes[labels[i]])
        ax.axis('off')
```

One point here: `matplotlib` is expecting the values of every pixel to be between $[0, 1]$, however our normalization will cause them to be outside this range. By default `matplotlib` will then clip these values into the $[0,1]$ range. This clipping causes all of the images to look a bit weird - all of the colors are oversaturated. The solution is to normalize each image between [0,1].


```python
N_IMAGES = 25

images, labels = zip(*[(image, label) for image, label in 
                           [train_data[i] for i in range(N_IMAGES)]])

classes = test_data.classes
```


```python
plot_images(images, labels, classes, normalize = True)
```


    
![png](output_16_0.png)
    


We'll be normalizing our images by default from now on, so we'll write a function that does it for us which we can use whenever we need to renormalize an image.


```python
def normalize_image(image):
    image_min = image.min()
    image_max = image.max()
    image.clamp_(min = image_min, max = image_max)#if < min: = min if > max: = max
    image.add_(-image_min).div_(image_max - image_min + 1e-5)# img - min / max - min + 10^-5
    return image
```

The final bit of the data processing is creating the iterators. We will use a large. Generally, a larger batch size means that our model trains faster but is a bit more susceptible to overfitting.


```python
# Q1: Create data loaders for train_data, valid_data, test_data
# Use batch size 256


BATCH_SIZE = 256

train_iterator = data.DataLoader(train_data,
                                 shuffle = True,
                                 batch_size = BATCH_SIZE) 

valid_iterator = data.DataLoader(valid_data,
                                 batch_size = BATCH_SIZE)

test_iterator = data.DataLoader(test_data,
                                batch_size = BATCH_SIZE)
```

### Defining the Model

Next up is defining the model.

AlexNet will have the following architecture:

* There are 5 2D convolutional layers (which serve as *feature extractors*), followed by 3 linear layers (which serve as the *classifier*).
* All layers (except the last one) have `ReLU` activations. (Use `inplace=True` while defining your ReLUs.)
* All convolutional filter sizes have kernel size 3 x 3 and padding 1. 
* Convolutional layer 1 has stride 2. All others have the default stride (1).
* Convolutional layers 1,2, and 5 are followed by a 2D maxpool of size 2.
* Linear layers 1 and 2 are preceded by Dropouts with Bernoulli parameter 0.5.

* For the convolutional layers, the number of channels is set as follows. We start with 3 channels and then proceed like this:

  - $3 \rightarrow 64 \rightarrow 192 \rightarrow384\rightarrow256\rightarrow 256$

  In the end, if everything is correct you should get a feature map of size $ 2\times2 \times 256 = 1024$.

* For the linear layers, the feature sizes are as follows:

  - $1024 \rightarrow 4096 \rightarrow 4096 \rightarrow 10$.

  (The 10, of course, is because 10 is the number of classes in CIFAR-10).


```python
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.activation import ReLU
from torch.nn.modules.pooling import MaxPool2d
class AlexNet(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3,64,3,2,1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace = True),
            nn.Conv2d(64,192,3,padding = 1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace = True),
            nn.Conv2d(192,384,3,padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(384,256,3,padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(256,256,3,padding = 1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace = True)

            # Define according to the steps described above
        )
        
        self.classifier = nn.Sequential(
            # define according to the steps described above
            nn.Dropout(0.5),
            nn.Linear(1024,4096),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(inplace = True),
            nn.Linear(4096,output_dim)
        )

    def forward(self, x):
        x = self.features(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        return x, h
```

We'll create an instance of our model with the desired amount of classes.


```python
OUTPUT_DIM = 10
model = AlexNet(OUTPUT_DIM)
```

### Training the Model

We first initialize parameters in PyTorch by creating a function that takes in a PyTorch module, checking what type of module it is, and then using the `nn.init` methods to actually initialize the parameters.

For convolutional layers we will initialize using the *Kaiming Normal* scheme, also known as *He Normal*. For the linear layers we initialize using the *Xavier Normal* scheme, also known as *Glorot Normal*. For both types of layer we initialize the bias terms to zeros.




```python
def initialize_parameters(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data, nonlinearity = 'relu')
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain = nn.init.calculate_gain('relu'))
        nn.init.constant_(m.bias.data, 0)
```

We apply the initialization by using the model's `apply` method. If your definitions above are correct you should get the printed output as below.


```python
model.apply(initialize_parameters)
```




    AlexNet(
      (features): Sequential(
        (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (2): ReLU(inplace=True)
        (3): Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (5): ReLU(inplace=True)
        (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (7): ReLU(inplace=True)
        (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (9): ReLU(inplace=True)
        (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (11): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (12): ReLU(inplace=True)
      )
      (classifier): Sequential(
        (0): Dropout(p=0.5, inplace=False)
        (1): Linear(in_features=1024, out_features=4096, bias=True)
        (2): ReLU(inplace=True)
        (3): Dropout(p=0.5, inplace=False)
        (4): Linear(in_features=4096, out_features=4096, bias=True)
        (5): ReLU(inplace=True)
        (6): Linear(in_features=4096, out_features=10, bias=True)
      )
    )





We then define the loss function we want to use, the device we'll use and place our model and criterion on to our device.


```python
optimizer = optim.Adam(model.parameters(), lr = 1e-3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()

model = model.to(device)
criterion = criterion.to(device)

```



```
# This is formatted as code
```

We define a function to calculate accuracy...


```python
def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc
```



As we are using dropout we need to make sure to "turn it on" when training by using `model.train()`.


```python
def train(model, iterator, optimizer, criterion, device):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for (x, y) in iterator:
        
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
                
        y_pred, _ = model(x)
        
        loss = criterion(y_pred, y)
        
        acc = calculate_accuracy(y_pred, y)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
```

We also define an evaluation loop, making sure to "turn off" dropout with `model.eval()`.


```python
def evaluate(model, iterator, criterion, device):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
        
        for (x, y) in iterator:

            x = x.to(device)
            y = y.to(device)

            y_pred, _ = model(x)

            loss = criterion(y_pred, y)

            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
```

Next, we define a function to tell us how long an epoch takes.


```python
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
```

Then, finally, we train our model.

Train it for 25 epochs (using the train dataset). At the end of each epoch, compute the validation loss and keep track of the best model. You might find the command `torch.save` helpful.

At the end you should expect to see validation losses of ~76% accuracy.


```python
# Q3: train your model here for 25 epochs. 
# Print out training and validation loss/accuracy of the model after each epoch
# Keep track of the model that achieved best validation loss thus far.

EPOCHS = 25

# Fill training code here
best_valid_loss = float('inf')
for epoch in range(EPOCHS):
  start_time = time.monotonic()

  train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
  valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, device)

  if valid_loss < best_valid_loss:
    best_valid_loss = valid_loss
    torch.save(model.state_dict(),'tut3-model.pt')
  end_time = time.monotonic()

  epoch_mins, epoch_secs = epoch_time(start_time, end_time)
  print(f'Epoch: {epoch+1:02} || EpochTime: {epoch_mins}m{epoch_secs}s')
  print(f'\tTrain Loss: {train_loss:.3f} || Train Acc: {train_acc*100:.2f}%')
  print(f'\t Val Loss: {valid_loss:.3f} ||  Val Acc: {valid_acc*100:.2f}%')
```

    Epoch: 01 || EpochTime: 0m34s
    	Train Loss: 2.319 || Train Acc: 24.32%
    	 Val Loss: 1.775 ||  Val Acc: 33.31%
    Epoch: 02 || EpochTime: 0m27s
    	Train Loss: 1.531 || Train Acc: 43.58%
    	 Val Loss: 1.329 ||  Val Acc: 51.48%
    Epoch: 03 || EpochTime: 0m28s
    	Train Loss: 1.360 || Train Acc: 50.93%
    	 Val Loss: 1.219 ||  Val Acc: 55.35%
    Epoch: 04 || EpochTime: 0m27s
    	Train Loss: 1.239 || Train Acc: 55.50%
    	 Val Loss: 1.134 ||  Val Acc: 59.05%
    Epoch: 05 || EpochTime: 0m27s
    	Train Loss: 1.165 || Train Acc: 58.62%
    	 Val Loss: 1.067 ||  Val Acc: 62.01%
    Epoch: 06 || EpochTime: 0m27s
    	Train Loss: 1.098 || Train Acc: 61.16%
    	 Val Loss: 0.990 ||  Val Acc: 64.96%
    Epoch: 07 || EpochTime: 0m27s
    	Train Loss: 1.037 || Train Acc: 63.55%
    	 Val Loss: 0.985 ||  Val Acc: 64.96%
    Epoch: 08 || EpochTime: 0m27s
    	Train Loss: 0.992 || Train Acc: 65.20%
    	 Val Loss: 0.887 ||  Val Acc: 68.66%
    Epoch: 09 || EpochTime: 0m27s
    	Train Loss: 0.950 || Train Acc: 66.68%
    	 Val Loss: 0.901 ||  Val Acc: 68.35%
    Epoch: 10 || EpochTime: 0m27s
    	Train Loss: 0.918 || Train Acc: 67.87%
    	 Val Loss: 0.861 ||  Val Acc: 70.39%
    Epoch: 11 || EpochTime: 0m27s
    	Train Loss: 0.871 || Train Acc: 69.69%
    	 Val Loss: 0.822 ||  Val Acc: 71.23%
    Epoch: 12 || EpochTime: 0m27s
    	Train Loss: 0.847 || Train Acc: 70.76%
    	 Val Loss: 0.805 ||  Val Acc: 71.49%
    Epoch: 13 || EpochTime: 0m27s
    	Train Loss: 0.832 || Train Acc: 71.08%
    	 Val Loss: 0.828 ||  Val Acc: 71.34%
    Epoch: 14 || EpochTime: 0m27s
    	Train Loss: 0.805 || Train Acc: 72.26%
    	 Val Loss: 0.774 ||  Val Acc: 73.57%
    Epoch: 15 || EpochTime: 0m27s
    	Train Loss: 0.787 || Train Acc: 72.66%
    	 Val Loss: 0.782 ||  Val Acc: 73.14%
    Epoch: 16 || EpochTime: 0m27s
    	Train Loss: 0.772 || Train Acc: 73.38%
    	 Val Loss: 0.771 ||  Val Acc: 73.19%
    Epoch: 17 || EpochTime: 0m26s
    	Train Loss: 0.745 || Train Acc: 74.45%
    	 Val Loss: 0.765 ||  Val Acc: 73.25%
    Epoch: 18 || EpochTime: 0m27s
    	Train Loss: 0.724 || Train Acc: 74.90%
    	 Val Loss: 0.744 ||  Val Acc: 74.25%
    Epoch: 19 || EpochTime: 0m27s
    	Train Loss: 0.716 || Train Acc: 75.33%
    	 Val Loss: 0.722 ||  Val Acc: 75.14%
    Epoch: 20 || EpochTime: 0m27s
    	Train Loss: 0.698 || Train Acc: 76.10%
    	 Val Loss: 0.704 ||  Val Acc: 75.70%
    Epoch: 21 || EpochTime: 0m27s
    	Train Loss: 0.686 || Train Acc: 76.33%
    	 Val Loss: 0.703 ||  Val Acc: 76.05%
    Epoch: 22 || EpochTime: 0m26s
    	Train Loss: 0.672 || Train Acc: 76.83%
    	 Val Loss: 0.684 ||  Val Acc: 76.64%
    Epoch: 23 || EpochTime: 0m27s
    	Train Loss: 0.650 || Train Acc: 77.69%
    	 Val Loss: 0.689 ||  Val Acc: 76.76%
    Epoch: 24 || EpochTime: 0m26s
    	Train Loss: 0.635 || Train Acc: 77.90%
    	 Val Loss: 0.674 ||  Val Acc: 76.95%
    Epoch: 25 || EpochTime: 0m26s
    	Train Loss: 0.635 || Train Acc: 78.03%
    	 Val Loss: 0.698 ||  Val Acc: 76.14%
    

# Evaluating the model


We then load the parameters of our model that achieved the best validation loss. You should expect to see ~75% accuracy of this model on the test dataset.

Finally, plot the confusion matrix of this model and comment on any interesting patterns you can observe there. For example, which two classes are confused the most?




```python
# Q4: Load the best performing model, evaluate it on the test dataset, and print test accuracy.

# Also, print out the confusion matrox.
model.load_state_dict(torch.load('tut3-model.pt'))

test_loss, test_acc = evaluate(model, test_iterator, criterion, device)

print(f'Test Loss: {test_loss:.3f} || Test Acc: {test_acc*100:.2f}%')

```

    Test Loss: 0.690 || Test Acc: 76.78%
    


```python
def get_predictions(model, iterator, device):

    model.eval() #no drop out

    labels = []
    probs = []

    # Q4: Fill code here.
    with torch.no_grad():
      for(x,y) in iterator:
        x = x.to(device)
        y_pred,_ = model(x)
        y_prob = F.softmax(y_pred, dim = -1)
        labels.append(y.cpu())
        probs.append(y_prob.cpu())
   
    labels = torch.cat(labels, dim = 0)
    probs = torch.cat(probs, dim = 0)

    return labels, probs
```


```python
labels, probs = get_predictions(model, test_iterator, device)
```


```python
pred_labels = torch.argmax(probs, 1)
```


```python
def plot_confusion_matrix(labels, pred_labels, classes):
    
    fig = plt.figure(figsize = (10, 10));
    ax = fig.add_subplot(1, 1, 1);
    cm = confusion_matrix(labels, pred_labels);
    cm = ConfusionMatrixDisplay(cm, display_labels = classes);
    cm.plot(values_format = 'd', cmap = 'Blues', ax = ax)
    plt.xticks(rotation = 20)
```


```python
plot_confusion_matrix(labels, pred_labels, classes) 
```


    
![png](output_47_0.png)
    


### Conclusion

That's it! As a side project (this is not for credit and won't be graded), feel free to play around with different design choices that you made while building this network.

- Whether or not to normalize the color channels in the input.
- The learning rate parameter in Adam.
- The batch size.
- The number of training epochs.
- (and if you are feeling brave -- the AlexNet architecture itself.)
