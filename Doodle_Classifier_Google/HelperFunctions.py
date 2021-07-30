import numpy as np


def random_initialization(rows, cols):
    return np.random.randn(rows, cols)

def zeros(rows,cols):
    return np.zeros((rows,cols)) 


def sigmoid(data):
    return 1/(1 + np.exp(-data))


def normalize(data):
    return data/255.0


 

def shuffle_dataset(data):
    temp = data
    np.random.shuffle(temp)
    return temp
def return_full_dataset(tuple_):

    concatenated = np.concatenate(tuple_)
    concatenated = shuffle_dataset(concatenated)
    print(concatenated.shape)
    return concatenated

# coudnt find any other way :< for loop F
def add_to_end(data,index):
    final = []
    for i in range(len(data)):
        temp = (np.append(data[i],index))
        final.append(temp)
    
    return np.array(final)


def split_dataset(data):
    index = np.floor(data.shape[0] * 0.9)
    train_data = data[:int(index)]
    test_data = data[int(index):]
    return [train_data,test_data]

def get_X_Y(data): 
    data_X = np.delete(data,-1,1)
    data_Y = (data[:,[-1]])
    return [data_X,data_Y]

def  get_tanh_derivative(F):
    return 1 - np.power(F,2)

def get_max_index(data):
    return np.argmax(data)
    