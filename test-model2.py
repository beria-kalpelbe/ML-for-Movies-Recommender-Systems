from Model import Model2,Model
import numpy as np
# from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
from multiprocessing import cpu_count
import pickle

print('---------------Reading datasets---------------------------')
data = np.genfromtxt('DATA/ratings.dat', delimiter='::')[:,:3]
data_movie = np.genfromtxt('DATA/movies.dat', delimiter='::',dtype="str")

print("----------------Splitting datasets--------------------------")
np.random.seed(555)
n_test = round(0.2*len(data))
test_index = np.random.choice(range(len(data)),size=n_test,replace=False)
data_test = data[test_index]
train_index = [i for i in range(len(data)) if i not in test_index]
data_train = data[train_index]
# data_train, data_test = train_test_split( data, test_size=0.2, random_state=42)

Ks = [0,2,7,16]
Lambdas = [0.01,0.22,0.5]
Taus = [0.01,0.22,0.5]

count = 0
indexes = []

parameters = []
for k in Ks:
    for Lambda in Lambdas:
        for tau in Taus:
            parameters.append((k,Lambda,tau))

def compute(params):
    results = []
    for param in params:
        model = Model.Model(tau=param[2], Lambda=param[1], train=data_train, test=data_test, K=param[0],data_movies=data_movie)
        model.fit(10)
        results.append(model.test_RMSE + model.train_RMSE)
    return results

if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
        final = executor.submit(compute, parameters)

    with open('finalData','wb') as file:
        pickle.dump(final, file)

    print(final.result())

# model = Model.Model(tau=0.01, Lambda=0.05, train=data_train, test=data_test, K=5,data_movies=data_movie)
# model.fit(10)

# model.plot_errors()