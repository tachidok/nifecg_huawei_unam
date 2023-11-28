#!/usr/bin/env python

# Import pytorch packages
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from torchinfo import summary
#import torchmetrics

# OS packages
import os
import sys
import shutil

# Argparse import
import argparse

# Pandas import
import pandas as pd

# Plotting
import matplotlib.pyplot as plt

# Additional libraries
import glob
import csv
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Timming
import time

# ******************************************************************************
# Custom dataset class
# ******************************************************************************
class FileDataset(Dataset):
    def __init__(self, root_dir, channels, sample_interval, transform=None):
        self.root_dir = root_dir
        
        self.channels = channels
        self.sample_interval = sample_interval
        
        self.transform = transform
        self.file_list = []
        self.labels = []
        
        # The number of classes
        self.n_classes = 0

        # Extract file paths and labels
        self._extract_file_paths()
        
         # Perform label encoding as one-hot encoding
        self._encode_labels()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        label = self.labels[idx]

        # Read the file
        data = self._read_file(file_path)

        # Apply transformation if provided
        if self.transform:
            data = self.transform(data)
            
        return data, label

    def _extract_file_paths(self):
        label_dirs = os.listdir(self.root_dir)

        for label in label_dirs:
            label_dir = os.path.join(self.root_dir, label)
            if os.path.isdir(label_dir):
                files = glob.glob(os.path.join(label_dir, '*.csv'))
                self.file_list.extend(files)
                self.labels.extend([label] * len(files))

    def _read_file(self, file_path):
        
        signal_data = np.loadtxt(file_path, dtype=np.float32, delimiter=",")
        # Transform to torch vector and reshape to column vector
        return torch.from_numpy(signal_data)
        
        # Implement your own file reading logic here
        # For example, if you're working with CSV files, you can use pandas
        # dataframe = pd.read_csv(file_path)
        # return dataframe.values

        # In this example, we assume a simple text file and read its content
        #with open(file_path, 'r') as file:
        #    content = file.read()

        #return content
        
    def _encode_labels(self):
        
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(self.labels)
        
        # Print the original classes names
        #print(label_encoder.classes_)
        
        # Reshape to column vector
        integer_encoded = integer_encoded.reshape(-1, 1)

        # JCPS "sparse" deprecated in version 1.2, use "sparse_output" from version 1.4
        #onehot_encoder = OneHotEncoder(sparse=False)
        onehot_encoder = OneHotEncoder(sparse_output=False)
        
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        
        self.n_classes = onehot_encoded.shape[1]
        
        self.labels = torch.tensor(onehot_encoded, dtype=torch.float32)
        #self.labels = torch.tensor(onehot_encoded, dtype=torch.long)


# ******************************************************************************
# Transformations
# ******************************************************************************
# Reshape the raw data to a row vector
class ToRowVector(object):
    """Transforms the input signal to a row vector"""
    
    def __call__(self, sample):
        #reshaped_sample = sample.view(1, -t1)
        #print(sample.shape[0])
        #print(sample.shape[1])
        
        # Slide the data, use only the first row
        preshaped = sample[0,:]
                        
        #reshaped_sample = sample.view(sample.shape[0]*sample.shape[1])
        reshaped_sample = preshaped.view(preshaped.shape[0])
        
        #return sample
        return reshaped_sample

# Subsample signal extracting only selected channels and/or a given sample interval
class SignalSubSample(object):
    
    def __init__(self, channels, sample_interval):
        """
        Initialize input arguments.
        """
        self.channels = channels
        self.sample_interval = sample_interval
    
    def __call__(self, sample):
            
        # Get the indices of the columns to extract
        indexes_columns = list(range(0, sample.shape[1], sample_interval))  # Select every "sample_interval" column
        
        # Extract selected channels and columns
        subsample_signal = sample[channels, :][:, indexes_columns]
        
        #print(subsample_signal.shape)
                               
        #return sample
        return subsample_signal


# ******************************************************************************
# Neural networks definitions
# ******************************************************************************

# *****************
# Linear Classifier
# *****************

# Define the neural network architecture
class LinearClassifier(nn.Module):
    # n_input_data_dim1: Number of data in dimension one (number of rows/channels)
    # n_input_data_dim2: Number of data in dimension two (number of columns/data_per_channel)
    def __init__(self, n_input_data_dim1, n_input_data_dim2, n_output):
        super(LinearClassifier, self).__init__()
           
        # Define your layers here   
        
        #self.linear1 = nn.Linear(in_dim, 1024, bias=True)
        #self.linear2 = nn.Linear(1024, 256)
        #self.relu = nn.ReLU()
        #self.linear3 = nn.Linear(256, out_dim)
     
        input_features = n_input_data_dim1 * n_input_data_dim2
    
        self.linear1 = nn.Linear(input_features, 128, bias=True)
        self.linear2 = nn.Linear(128, 32)
        self.relu = nn.ReLU()
        self.linear3 = nn.Linear(32, n_output)

    def forward(self, x):
        
        #print("Before the flattening")
        #print(x.shape[0])
        #print(x.shape[1])
        #print(x.shape)
        
        # Perform the flattening of the input data to pass through the linear layers
        #reshaped_sample = sample.view(sample.shape[0]*sample.shape[1])
    #accuracy_metric = torchmetrics.classification.Accuracy(task="multic
        x = x.view(x.shape[0], -1)
        
        #print("After the flattening")
        #print(x.shape[0])
        #print(x.shape[1])
        #print(x.shape)
        
        # Define the forward pass
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
                
        return x

# *****************
# CNN Classifier
# *****************

# Define the neural network architecture
class CNNClassifier(nn.Module):
    # n_input_data_dim1: Number of data in dimension one (number of rows/channels)
    # n_input_data_dim2: Number of data in dimension two (number of columns/data_per_channel)
    def __init__(self, n_input_data_dim1, n_input_data_dim2, n_output):
        super(CNNClassifier, self).__init__()
           
        # Define your layers here   
        
        #self.linear1 = nn.Linear(in_dim, 1024, bias=True)
        #self.linear2 = nn.Linear(1024, 256)
        #self.relu = nn.ReLU()
        #self.linear3 = nn.Linear(256, out_dim)
     
        input_features = n_input_data_dim1 * n_input_data_dim2
        
        self.Conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 1, 
                      out_channels = 64,
                      kernel_size = (3, 3),
                      stride = 1,
                      padding = 1
                     ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.Conv2 = nn.Sequential(
            nn.Conv2d(in_channels = 64, 
                      out_channels = 128,
                      kernel_size = (3, 3),
                      stride = 1,
                      padding = 1
                     ),
            nn.Dropout(),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True)
        )
        
        self.Conv3 = nn.Sequential(
            nn.Conv2d(in_channels = 128, 
                      out_channels = 256,
                      kernel_size = (3, 3),
                      stride = 1,
                      padding = 1
                     ),
            nn.Dropout(),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True)
        )
        
        self.linear1 = nn.Linear(256 * 3 * 3, n_output, bias=True)

    def forward(self, x):
        
        print("Before the flattening")
        print(x.shape[0])
        print(x.shape[1])
        print(x.shape)
        
        # Define the forward pass
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        
        # Perform the flattening of the input data to pass through the linear layers
        #reshaped_sample = sample.view(sample.shape[0]*sample.shape[1])
        x = x.view(x.shape[0], -1)
        #x = nn.Flatten(x)
        
        #print("After the flattening")
        #print(x.shape[0])
        #print(x.shape[1])
        #print(x.shape)
        
        x = self.linear1(x)
        
        #return F.log_softmax(x, dim=1)
                
        return x


# ******************************************************************************
# Helper functions for performace evaluation
# ******************************************************************************

# Define a helper function to generate a one-hot encoding at the position of the maximum value
def generate_max_indices_tensor(input_tensor):
    # Compute the maximum along each row
    max_values, max_indices = torch.max(input_tensor, dim=1)
    
    # Create a tensor of zeros with the same shape as the input tensor
    output_tensor = torch.zeros_like(input_tensor)
    
    # Set ones at the indices of the maximum values
    output_tensor.scatter_(1, max_indices.unsqueeze(1), 1)
    
    return output_tensor

# Define a helper function that returns a one tensor if the input tensors are equal
def compare_tensors(tensor1, tensor2):
    
    # Ensure both tensors have the same shape
    assert tensor1.shape == tensor2.shape, "Both tensors should have the same dimensions."

    # Calculate element-wise equality and count equal rows
    row_equality = torch.all(tensor1 == tensor2, dim=1)
    equal_rows = torch.sum(row_equality).item()

    # Count different rows
    different_rows = tensor1.shape[0] - equal_rows
    
    return torch.tensor([equal_rows, different_rows])

# ******************************************************************************
# Train the model
# ******************************************************************************
# Define the training method
def train(model=net,
    #accuracy_metric = torchmetrics.classification.Accuracy(task="multic
          optimizer=opt,
          n_epochs=n_epochs,
          loss_fn=criterion,
          lr=learning_rate):
    
    #accuracy_metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes = n_output_classes)
    
    # Indicate the Pytorch backend we are on training mode
    model.train()
    loss_lt = []
    accuracy_lt = []
    
    # Training loop
    for epoch in range(n_epochs):
        running_loss = 0.0
        accuracy_epoch = 0.0
        batch_counter = 0
        for batch_data, batch_labels in train_loader:
        
            # Prior any operation clear the gradient
            optimizer.zero_grad()
        
            # Move data and labels to the device
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
        
            # Forward pass
            outputs = model(batch_data)

            #print("Outputs\n")
            #print(outputs)
        
            #print("Batch labels\n")
            #print(batch_labels)
        
            # Apply the loss function
            loss = loss_fn(outputs, batch_labels)
            
            # Backpropagation and optimization
            loss.backward()
        
            # Perform an optimization step (this updates the weights and bias on the network)
            optimizer.step()
                    
            # Keep track of sum of loss of each batch
            running_loss+=loss.item()
            
             # Compute the one-hot enconding version
            one_hot_output = generate_max_indices_tensor(outputs)
                   
            # Compute accuracy
            batch_accuracy = compare_tensors(one_hot_output, batch_labels)[0]
            
            # Keep track of accuracy
            accuracy_epoch+=batch_accuracy
            
            # Metrics
            #acc = accuracy_metric(outputs, batch_labels)
            
            # Print the loss for monitoring
            print('Epoch [{}/{}], batch [{}/{}], lr={:.6f}, batch Loss: {:.4f}, batch accuracy: {:.4f}'.format(epoch+1, n_epochs,
                                                                                                                batch_counter + 1, len(train_loader),
                                                                                                                lr, loss.item(),
                                                                                                                batch_accuracy*100.0/batch_size))
            #print("Torch Metrics: {}".format(acc))
            
            # Increase the batch counter
            batch_counter += 1
        
        # Add the cumulative loss to a list
        loss_lt.append(running_loss/len(train_loader))
        
        # Add the cumulative accuracy to a list
        accuracy_lt.append(accuracy_epoch/len(train_loader)) 
        
        # Print the total loss of the epoch
        print('Epoch: {} training loss: {:.4f}, training accuracy: {:.4f}'.format(epoch+1, running_loss/len(train_loader), accuracy_epoch/len(train_loader)))
        
        #acc = accuracy_metric.compute()
        #print("All batches torch metric: {}".format(acc))
    
    #accuracy_metric.reset()
    
    fig = plt.figure(figsize=(17, 5))
    ax1 = plt.subplot(121)
    ax1.plot([i for i in range(1, n_epochs+1)], loss_lt, label="Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-entropy")
    ax1.set_title("Training loss: optimiser {}, lr {:.6f}".format("Adam", lr))
    ax1.legend()
    
    ax2 = plt.subplot(122)
    ax2.plot([i for i in range(1, n_epochs+1)], accuracy_lt, label="Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training accuracy: optimiser {}, lr {:.6f}".format("Adam", lr))
    ax2.legend()
    
    plt.show()

    # Save the trained model
    #torch.save(net.state_dict(), "./trained_model/model.pt')

# ******************************************************************************
# Test the model
# ******************************************************************************
# Define the testing method
def test(model=net,
        loss_fn=criterion,
        lr=learning_rate):
    
    # Indicate the Pytorch backend we are on testing mode
    model.eval()
    accuracy = 0.0
    total_loss = 0.0 
    
    # Use no grad to reduce memory and computation cost
    with torch.no_grad():
        
        batch_counter = 0
        for batch_data, batch_labels in test_loader:
            
            # Move data and labels to the device
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            # Forward pass
            outputs = model(batch_data)
            
            #print("These are the outputs")
            #print(outputs)
            
            #print("These are the batch labels")
            #print(batch_labels)
            
            # Compute the loss
            batch_loss = loss_fn(outputs, batch_labels)
            
            # Add up the loss
            total_loss+=batch_loss.item()
            
            # Compute the one-hot enconding version
            one_hot_output = generate_max_indices_tensor(outputs)
            
            #print("These are the one-hot outputs")
            #print(one_hot_output)
            
            # Compute accuracy
            batch_accuracy = compare_tensors(one_hot_output, batch_labels)[0]
            #batch_accuracy = torch.sum(compare_tensors(one_hot_output, batch_labels))
            #print(compare_tensors(one_hot_output, batch_labels))
            
            accuracy+=batch_accuracy
            
            print("Test batch [{}/{}], lr={:.6f},  batch loss: {:.4f}, test batch accuracy: {:.4f}".format(
                batch_counter + 1, len(test_loader), lr, 
                batch_loss/batch_size,
                batch_accuracy*100.0/batch_size))
            
            # Increase the batch counter
            batch_counter += 1
            
        print("Test loss: {:.4f}, test accuracy: {:.4f}".format(
            total_loss/(len(test_loader)*batch_size),
            accuracy*100.0/(len(test_loader)*batch_size)))
    
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def main():
            
    # Create the parser to deal with the arguments
    parser = argparse.ArgumentParser(description="A deep neural network to identify the heart rate of the fetus on a NI-FECG (non-invasive fetal electrocardiography) signal")

    # Set positional arguments
    parser.add_argument("--n_output_classes", dest="n_output_classes", type=int, help="The number of classes that the neural network identifies", default=7)
    parser.add_argument("--channels", dest="channels", type=int, nargs='+', help="The channels' indexes from the input signal used to train and test the neural network", required=True)
    parser.add_argument("--sample_interval", dest="sample_interval", type=int, help="We have 1 minute signals at 60Hz, that is 60,000 data points. The size of the signal would be: 60,000/sample_interval)", default=10)
    parser.add_argument("--batch_size", dest="batch_size", type=int, help="The batch size for training and testing)", default=10)
    parser.add_argument("--n_epochs", dest="n_epochs", type=int, help="The number of epocs to train the neural network)", default=1)
    parser.add_argument("--learning_rate", dest="learning_rate", type=float, help="The learning rate for the neural network", default=1e-2)

    parser.add_argument("--training_folder", dest="training_folder", type=str, help="The training folder", default="./training_set")
    parser.add_argument("--testing_folder", dest="testing_folder", type=str, help="The testing folder", default="./testing_set")
    
    parser.add_argument("--use_device", dest="use_device", type=str, help="Use 'cpu', 'gpu' or 'npu'", required=True, choices=['cpu', 'gpu', 'npu'])

    parser.add_argument("--root_output_folder", dest="root_output_folder", help="The root output folder", default="RESLT")
    
    # parse args
    args = parser.parse_args()

    print("\n")
    
    # *********************************
    # Read and validate input arguments
    # *********************************

    # Hyper-parameters
    n_output_classes = args.n_output_classes
    channels = args.channels
    sample_interval = args.sample_interval
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    learning_rate = args.learning_rate
    use_device = args.use_device.lower()
    
    # Are there enough channels in the input signal?
    if len(args.channels) > 32:
        parser.error('The number of channels is out of range [1 to 32]')
        return 1
    
    # Training and testing folders
    if not os.path.exists(args.training_folder):
        parser.error("The training folder does not exist: {}".format(args.training_folder))
        return 1

    if not os.path.exists(args.testing_folder):
        parser.error("The testing folder does not exist: {}".format(args.testing_folder))
        return 1

    training_folder_name = args.training_folder
    testing_folder_name = args.testing_folder
        
    # Output folder
    root_output_folder = args.root_output_folder
    if os.path.exists(args.root_output_folder):
        parser.error("The root output folder already exists!: {}".format(args.root_output_folder))
        return 1

    # Generate output folder
    os.mkdir(root_output_folder)

    # Add the slash to the folder name
    root_output_folder = root_output_folder + "/"
    
    # Print input arguments for NN
    print("Running experiment with the following arguments values:")
    
    # ********************************
    # Print and save parameters values
    # ********************************
    print("n_output_classes:{:d}".format(n_output_classes))
    print("channels:{}".format(channels))
    print("sample_interval:{:d}".format(sample_interval))
    print("batch_size:{:d}".format(batch_size))
    print("n_epochs:{:d}".format(n_epochs))
    print("learning_rage:{:.6f}".format(learning_rate))
    print("use_device:{:s}".format(use_device))
    print("training_folder_name:{:s}".format(training_folder_name))
    print("testing_folder_name:{:s}".format(testing_folder_name))
    print("root_output_folder:{:s}".format(root_output_folder))

    # Save to disk
    output_parameters_file = root_output_folder + "parameters.txt"

    with open(output_parameters_file, "w") as f:

        # Add the line arguments as a comment to the file
        n = len(sys.argv)
        f.write("#")
        for i in range(0, n):
            f.write(sys.argv[i])
            f.write(" ")
            
        f.write("\n")
        
        output_lines = []
        output_lines.append("n_output_classes:{:d}\n".format(n_output_classes))
        output_lines.append("channels:{}\n".format(channels))
        output_lines.append("sample_interval:{:d}\n".format(sample_interval))
        output_lines.append("batch_size:{:d}\n".format(batch_size))
        output_lines.append("n_epochs:{:d}\n".format(n_epochs))
        output_lines.append("learning_rage:{:.6f}\n".format(learning_rate))
        output_lines.append("use_device:{:s}\n".format(use_device))
        output_lines.append("training_folder_name:{:s}\n".format(training_folder_name))
        output_lines.append("testing_folder_name:{:s}\n".format(testing_folder_name))
        output_lines.append("root_output_folder:{:s}\n".format(root_output_folder))
            
        f.writelines(output_lines)

    # *********************************
    # Select the device (CPU, GPU, NPU)
    # *********************************
    device = "cpu"
    
    # Determine the device (NPU, CPU or GPU)
    device = "cpu"
    if use_device == "npu" and torch.npu.is_available():
        device = "npu:0"
    elif use_device == "gpu" and torch.cuda.is_available():
        device = "cuda"
    
    print("\nThe using device is:{:s}\n".format(device))

    # The folder with the dataset
    #training_folder_name = "../../data/sorted_by_mhr/training_set"
    #testing_folder_name = "../../data/sorted_by_mhr/testing_set"

    # The folder with the dataset
    #training_folder_name = "../02_python_signal_folder_sorting/signals_by_range_and_mhr/training_set"
    #testing_folder_name = "../02_python_signal_folder_sorting/signals_by_range_and_mhr/testing_set"
    #training_folder_name = "../../data/sorted_by_fhr/training_set"
    #testing_folder_name = "../../data/sorted_by_fhr/testing_set"

    #batch_size = 10
    #batch_size = 25

    # Specify the channels to work with
    #channels = [0] # one channel
    #channels = [0, 4, 8, 12, 16, 20, 24, 28, 32] # eight channels
    #channels = [0, 4] # eight channels

    # Specify the sample interval (we have 1 minute signal with 60,000 points *60Hz*)
    #sample_interval = 10 # reduce the number of samples to 6,000
    #sample_interval = 1 # use the whole signal samples

    # Define the number of epochs
    #n_epochs = 1
    
    # Define the optimiser (with its corresponding learning rate)
    #learning_rate = 1e-2

    # The number of output clasess
    #n_output_classes = 4 # output dimension
    #n_output_classes = 11 # output dimension
    #n_output_classes = 6 # output dimension (this is incorrect but works because in the laptop we have only six groups)
    #n_output_classes = 7 # output dimension uncomment this line when move to UNAM server

    # ***********************
    # Instantiate data loader
    # ***********************

    # Create training datasets
    training_dataset = FileDataset(training_folder_name,
                                   channels, 
                                   sample_interval, 
                                   transform=transforms.Compose([
                                       SignalSubSample(channels, sample_interval)#,
                                       #transforms.ToTensor() this does not apply because our input data are not images
                                       #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                       #transforms.Normalize((0.1), (0.1)) # We need a method to normalize a tensor, not an image
                                   ]))

    # Create testing datasets
    testing_dataset = FileDataset(testing_folder_name,
                                  channels, 
                                  sample_interval, 
                                  transform=transforms.Compose([
                                      SignalSubSample(channels, sample_interval)#,
                                      #transforms.ToTensor() this does not apply because our input data are not images
                                      #transforms.Normalize(mean=[0.1], std=[0.01])
                                  ]))

    # The training/testing data loaders
    train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False)

    print("Training/testing data loaders info:\n")
    print("Train loader: Total number of batches {} using {} items per batch. Total samples {}".format(len(train_loader), batch_size, len(train_loader) * batch_size))
    print("Test loader: Total number of batches {} using {} items per batch. Total samples {}".format(len(test_loader), batch_size, len(test_loader) * batch_size))

    # ***********************************
    # Instantiate the optimiser and model
    # ***********************************
    
    # Get the number of channels on the processing signal
    n_channels = len(channels)
    
    # Original number of data per channel
    n_original_data_per_channel = 60000
    
    # Get the number of data per channel
    n_data_per_channel = n_original_data_per_channel // sample_interval # Integer division
    
    #n_input_features = 2040000
    n_input_features = n_channels * n_data_per_channel # input dimension
    print("Number of input features: {}\n".format(n_input_features))

    # Create an instance of the neural network and move it to the device
    #net = LinearClassifier(n_channels, n_data_per_channel, n_output_classes).to(device)
    net = CNNClassifier(n_channels, n_data_per_channel, n_output_classes).to(device)

    #net.cuda()
    #net.cpu()

    # Define the loss function
    # Cross-entropy
    criterion = nn.CrossEntropyLoss()

    # Mean Square Error
    #criterion = nn.MSELoss()
    
    # Use Adam optimiser
    opt = optim.Adam(net.parameters(), lr=learning_rate)

    # Stochastic Gradient Descent
    #opt = optim.SGD(net.parameters(), lr=learning_rate)

    # Summary of the model
    print("Manual summary of the model")
    for p in net.parameters():
        print(p.shape)

    # Summary of the model
    print("Automatic summary of the model")
    #summary(net, input_size = (batch_size, 2040000, 4))
    summary(net, input_size = (batch_size, n_input_features))
    #summary(net)

    # ************************
    # Call the training method
    # ************************
    train(net, opt, n_epochs, criterion)

    # ********************
    # Call the test method
    # ********************
    test(net, criterion)
    
    return 0

    tic = time.clock()
    toc = time.clock()
    print(toc - tic)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
if __name__ == '__main__':    
    # Run the main function
    main()



'''

# # Code testing

# In[ ]:


for batch_data, batch_labels in train_loader:
    print(batch_data.shape)
    print(batch_labels)
    print(batch_labels.shape)


# In[ ]:


signal_data = np.loadtxt("../02_python_signal_folder_sorting/sorted_signals_by_mhr/70_74/nifecg.0003.fs_1000_mhr_72_fhr_132.csv", dtype=np.float32, delimiter=",")
signal_data_torch = torch.from_numpy(signal_data)
signal_data_torch


# In[ ]:


## One-hot enconding
encoder = OneHotEncoder


# In[ ]:


x = torch.rand(3,4)
print(x)
idx, x_max = x.max(dim=1)
print(x_max)
x_arg_max = torch.argmax(x, 1)
print(x_arg_max)


# In[ ]:


# Determine the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


# In[ ]:

'''



