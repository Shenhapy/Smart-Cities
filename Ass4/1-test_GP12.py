import pickle
import matplotlib.pyplot as plt
import numpy as np
import helper

downloaded_model = pickle.load(open("global_model.pickle",'rb'))
classes, x_train, y_train,x_test, y_test = helper.get_mnist_data()
IMAGE_DIMENSION = [36,36]
data_name = "mnist"

accuracy,outputs,labels = helper.validation(downloaded_model,x_test,y_test,IMAGE_DIMENSION,data_name)
print("validation accuracy is:{} \t data size is: {}".format(accuracy,len(x_test)))
outputs = outputs[1].numpy()

initial = 56
cnt = 0

f, axarr = plt.subplots(4,3)
for i in range(4):
    for j in range(3):
        axarr[i,j].set_title("real: {}  predicted: {}".format(classes[y_test[initial+cnt]][0],classes[outputs[initial+cnt]][0]))
        axarr[i,j].imshow(np.transpose(x_test[initial+cnt], (1, 2, 0)))
        axarr[i, j].axis('off')
        cnt = cnt+1

plt.show()
