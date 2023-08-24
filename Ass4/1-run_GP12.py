from py_interface import *
from ctypes import *

import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

import helper

class Env(Structure):
    _pack_ = 1
    _fields_ = [
        ('client_num', c_int),
        ('clientUpdateFlag',c_bool),
        ('isRoundFinished', c_bool)
    ]

class Act(Structure):
    _pack_ = 1
    _fields_ = [
        ('client_accuracy', c_float),
        ('server_accuracy', c_float)
    ]



ns3Settings = { }

mempool_key = 1234                                          
mem_size = 4096                                             
memblock_key = 2333  
print("PYTHON:: Starting!!")

exp = Experiment(mempool_key, mem_size, 'Tutorial_81', '../../')

num_packet = 0
fl = Ns3AIRL(memblock_key, Env, Act)
flag = True

Rounds = 10
local_epochs = 20
batch_size = 128

accuracy = 0

# SET SEED
seedNum = 3
torch.manual_seed(seedNum)
np.random.seed(seedNum)

DATA_SIZE = 30000
VALIDATION_SPLIT = 0.6
DATA_ID_DICT = {1:"mnist"}
IMAGE_DIMENSION = [36,36]


# CREATING DATASET AND CLIENT MODELS & SERVER MODEL
data_name = DATA_ID_DICT[1]
client_num = 20
validation_split = 0.6

classes, x_train, y_train,x_test, y_test = helper.get_mnist_data()
x_train, y_train = x_train[:DATA_SIZE], y_train[:DATA_SIZE]
x_test, y_test = x_test[:int(len(x_test)*VALIDATION_SPLIT)], y_test[:int(len(y_test)*VALIDATION_SPLIT)]

num_class = len(classes)
client_list, client_data_split = helper.split_data(x_train,y_train,client_num,True,data_name,IMAGE_DIMENSION)

print("Creating Model...")
models = [helper.Net(num_class=num_class,dim=IMAGE_DIMENSION) for i in range(len(client_list))]
if torch.cuda.is_available():
    models = [model.cuda() for model in models]
client_model_split = {client:model for client,model in zip(client_list,models)}

global_model = helper.Net(num_class=num_class,dim=IMAGE_DIMENSION)


s=0
for c in client_list:
    # print("client_name:{}  data_size:{}  label_size:{}".format(c,client_data_split[c][0].shape,client_data_split[c][1].shape))
    s = s+len(client_data_split[c][0])
print("Total Data Size:{}".format(s))

ns3Settings = {"client_num":client_num}

######################################
#1- Finds each Client’s data size and print them for proper client.
client_data_sizes = {client_name: len(client_data_split[client_name][0]) for client_name in client_list}
# Calculate summation of all client data sizes (30000)
summation_of_all_client_data_size = sum(client_data_sizes.values())
for client_name, data_size in client_data_sizes.items():
    print("Client {}: Data size = {}".format(client_name, data_size))

#2- Calculate each client’s participation rate and print them
#Tip 4- Participation rate for each client can be calculated by: (clients_data_size)/(summation_of_all_client_data_size)
participation_rates = {client_name: data_size / summation_of_all_client_data_size for client_name, data_size in client_data_sizes.items()}
print("Participation Rates:")
for client_name, participation_rate in participation_rates.items():
    print("{}: {:.4f}".format(client_name, participation_rate))

#Tip 3- You can write assessment code that make sure all clients participation rate summation is 1.
sum_participation_rate = sum(participation_rates.values())
print("Sum of Participation Rates: {:.4f}".format(sum_participation_rate))
#########################################

try:    
    for round in range(Rounds):
        
        print("*****************************************************")
        print("PYTHON:: round {}".format(round))
        ## AI Part
        client_validation_split = {}
        for client_name in client_list:
            # Syncronize with server's global model
            helper.syncronize_with_server(global_model,client_model_split[client_name])

            data = client_data_split[client_name][0]
            label = client_data_split[client_name][1]
            model = client_model_split[client_name]

            client_model_split[client_name] = helper.train_local(model,data,label,client_name,local_epochs,batch_size)
            
            client_validation_split[client_name] = helper.validation(client_model_split[client_name],x_test,y_test,IMAGE_DIMENSION,data_name)[0]

        ## Simulation Part
        received_clients = {}

        exp.reset()
        pro = exp.run(setting=ns3Settings,show_output=True)
        while not fl.isFinish():
            
            with fl as data:
                if data == None:
                    break

                
                if data.env.clientUpdateFlag:
                    ReceivedClient = data.env.client_num
                    
                    print("PYTHON:: Received Client is: {}".format(ReceivedClient))
                    
                    client_name = "client_"+str(ReceivedClient)
                    received_clients[client_name] = client_model_split[client_name]
                    received_client_score = client_validation_split[client_name]

                    data.act.client_accuracy = received_client_score

                if data.env.isRoundFinished:
                    print("PYTHON:: All Clients are Received - Aggregation Starts!!!")
                    
#Tip 1- For task 1, you need to change federated_averaging(global_model,received_clients) function
#and write your own federated averaging function. It can be done by adding additional
#parameter that holds clients data size.
#Tip 2- New data structure must be created for weighted aggregation. This structure holds data size
#for each client. It can be a dictionary that has key indicates “client name” and value indicates
#“data size”. You can use “client_data_split” dictionary create this new data structure.
                    helper.federated_averaging(global_model, received_clients, client_data_sizes)
                    

                    score = helper.validation(global_model,x_test,y_test,IMAGE_DIMENSION,data_name)[0]

                    data.act.server_accuracy = score 

                    print("PYTHON:: Aggregation is finished - Model Downloading!!!")

                    pickle.dump(global_model,open("global_model.pickle","wb"))

                    break

        
        
except Exception as e:
    print('Something wrong')
    print(e)
finally:
    del exp
