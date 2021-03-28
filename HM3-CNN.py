#!/usr/bin/env python
# coding: utf-8

# # Home 3: Build a CNN for image recognition.
# 
# ### Name: Prathamesh Dilip Bane (CWID 10457749)
# 

# ## 0. You will do the following:
# 
# 1. Read, complete, and run the code.
# 
# 2. **Make substantial improvements** to maximize the accurcy.
#     
# 3. Convert the .IPYNB file to .HTML file.
# 
#     * The HTML file must contain the code and the output after execution.
#     
#     
# 4. Upload this .HTML file to your Google Drive, Dropbox, or Github repo.
# 
# 4. Submit the link to this .HTML file to Canvas.
# 
#     * Example: https://github.com/wangshusen/CS583-2019F/blob/master/homework/HM3/HM3.html
# 
# 
# ## Requirements:
# 
# 1. You can use whatever CNN architecture, including VGG, Inception, and ResNet. However, you must build the networks layer by layer. You must NOT import the archetectures from ```keras.applications```.
# 
# 2. Make sure ```BatchNormalization``` is between a ```Conv```/```Dense``` layer and an ```activation``` layer.
# 
# 3. If you want to regularize a ```Conv```/```Dense``` layer, you should place a ```Dropout``` layer **before** the ```Conv```/```Dense``` layer.
# 
# 4. An accuracy above 70% is considered reasonable. An accuracy above 80% is considered good. Without data augmentation, achieving 80% accuracy is difficult.
# 
# 
# ## Google Colab
# 
# - If you do not have GPU, the training of a CNN can be slow. Google Colab is a good option.
# 
# - Keep in mind that you must download it as an IPYNB file and then use IPython Notebook to convert it to HTML.
# 
# - Also keep in mind that the IPYNB and HTML files must contain the outputs. (Otherwise, the instructor will not be able to know the correctness and performance.) Do the followings to keep the outputs.
# 
# - In Colab, go to ```Runtime``` --> ```Change runtime type``` --> Do NOT check ```Omit code cell output when saving this notebook```. In this way, the downloaded IPYNB file contains the outputs.

# ## 1. Data preparation

# ### 1.1. Load data
# 

# In[1]:


from keras.datasets import cifar10
import numpy

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print('shape of x_train: ' + str(x_train.shape))
print('shape of y_train: ' + str(y_train.shape))
print('shape of x_test: ' + str(x_test.shape))
print('shape of y_test: ' + str(y_test.shape))
print('number of classes: ' + str(numpy.max(y_train) - numpy.min(y_train) + 1))


# ### 1.2. One-hot encode the labels
# 
# In the input, a label is a scalar in $\{0, 1, \cdots , 9\}$. One-hot encode transform such a scalar to a $10$-dim vector. E.g., a scalar ```y_train[j]=3``` is transformed to the vector ```y_train_vec[j]=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]```.
# 
# 1. Define a function ```to_one_hot``` that transforms an $n\times 1$ array to a $n\times 10$ matrix.
# 
# 2. Apply the function to ```y_train``` and ```y_test```.

# In[2]:



from tensorflow.keras.utils import to_categorical    
y_train_vec = to_categorical(y_train, num_classes=10)
y_test_vec = to_categorical(y_test, num_classes=10)    
#If we need to convert our dataset into categorical format (and hence one-hot encoded format),
# we can do so using Scikit-learn’s OneHotEncoder module. However, TensorFlow also offers its own implementation: tensorflow.keras.utils.to_categorical.
# It’s a utility function which allows us to convert integer targets into categorical and hence one-hot encoded ones.

print('Shape of y_train_vec: ' + str(y_train_vec.shape))
print('Shape of y_test_vec: ' + str(y_test_vec.shape))

print(y_train[0])
print(y_train_vec[0])


# #### Remark: the outputs should be
# * Shape of y_train_vec: (50000, 10)
# * Shape of y_test_vec: (10000, 10)
# * [6]
# * [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]

# ### 1.3. Randomly partition the training set to training and validation sets
# 
# Randomly partition the 50K training samples to 2 sets:
# * a training set containing 40K samples
# * a validation set containing 10K samples
# 

# In[3]:


rand_indices = numpy.random.permutation(50000)
train_indices = rand_indices[0:40000]
valid_indices = rand_indices[40000:50000]

x_val = x_train[valid_indices, :]
y_val = y_train_vec[valid_indices, :]

x_tr = x_train[train_indices, :]
y_tr = y_train_vec[train_indices, :]

print('Shape of x_tr: ' + str(x_tr.shape))
print('Shape of y_tr: ' + str(y_tr.shape))
print('Shape of x_val: ' + str(x_val.shape))
print('Shape of y_val: ' + str(y_val.shape))


# ## 2. Build a CNN and tune its hyper-parameters
# 
# 1. Build a convolutional neural network model
# 2. Use the validation data to tune the hyper-parameters (e.g., network structure, and optimization algorithm)
#     * Do NOT use test data for hyper-parameter tuning!!!
# 3. Try to achieve a validation accuracy as high as possible.

# ### Remark: 
# 
# The following CNN is just an example. You are supposed to make **substantial improvements** such as:
# * Add more layers.
# * Use regularizations, e.g., dropout.
# * Use batch normalization.

# In[8]:


from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.10))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.10))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.10))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(10, activation='softmax'))

model.summary()


# In[9]:


from keras import optimizers

learning_rate = 1E-5 # to be tuned!

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=learning_rate),
              metrics=['acc'])


# In[10]:


history = model.fit(x_tr, y_tr, batch_size=32, epochs=10, validation_data=(x_val, y_val))


# In[11]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

acc = history.history['acc']
val_acc = history.history['val_acc']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# ## 3. Train (again) and evaluate the model
# 
# - To this end, you have found the "best" hyper-parameters. 
# - Now, fix the hyper-parameters and train the network on the entire training set (all the 50K training samples)
# - Evaluate your model on the test set.

# ### 3.1. Train the model on the entire training set
# 
# Why? Previously, you used 40K samples for training; you wasted 10K samples for the sake of hyper-parameter tuning. Now you already know the hyper-parameters, so why not using all the 50K samples for training?

# In[ ]:


<Compile your model again (using the same hyper-parameters)>


# In[ ]:



<Train your model on the entire training set (50K samples)>
<Use (x_train, y_train_vec) instead of (x_tr, y_tr)>
<Do NOT use the validation_data option (because now you do not have validation data)>
...


# ### 3.2. Evaluate the model on the test set
# 
# Do NOT used the test set until now. Make sure that your model parameters and hyper-parameters are independent of the test set.

# In[ ]:


loss_and_acc = model.evaluate(x_test, y_test_vec)
print('loss = ' + str(loss_and_acc[0]))
print('accuracy = ' + str(loss_and_acc[1]))


# In[12]:


rand_indices = numpy.random.permutation(50000)
train_indices = rand_indices[0:50000]


x_train = x_train[train_indices, :]
y_train_vec = y_train_vec[train_indices, :]

print('Shape of x_tr: ' + str(x_train.shape))
print('Shape of y_tr: ' + str(y_train_vec.shape))



# In[13]:


from keras import optimizers

learning_rate = 1E-5 # to be tuned!

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=learning_rate),
              metrics=['acc'])


# In[14]:


history = model.fit(x_train, y_train_vec, batch_size=32, epochs=10)


# In[15]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

acc = history.history['acc']


epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[16]:


loss_and_acc = model.evaluate(x_test, y_test_vec)
print('loss = ' + str(loss_and_acc[0]))
print('accuracy = ' + str(loss_and_acc[1]))


# In[ ]:




