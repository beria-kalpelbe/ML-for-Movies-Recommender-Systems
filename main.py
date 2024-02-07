import numpy as np
#import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from sklearn.model_selection import train_test_split
import pickle
from Model import Model,ModelFeatures

# data = np.loadtxt('ml-100k/u.data', delimiter = '\t', dtype='int')[:,:3]
# data = np.loadtxt('ml-100k/ratings.csv', delimiter = ',', skiprows=1)[:,:3]
# data = np.loadtxt('ml-25m/ratings.csv',delimiter=',',skiprows=1)[:,:3]
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

print("---------------For k=0---------------")
model1 = Model.Model(tau=0.01, Lambda=0.05, train=data_train, test=data_test, K=0, data_movies=data_movie)
model1.fit(30)
with open('ModelK0','wb') as file:
    pickle.dump(model1, file)


print("---------------For k=2---------------")
model2 = Model.Model(tau=0.01, Lambda=0.05, train=data_train, test=data_test, K=2, data_movies=data_movie)
model2.fit(30)
with open('ModelK2','wb') as file:
    pickle.dump(model2, file)


print("---------------For k=4---------------")
model3 = Model.Model(tau=0.01, Lambda=0.05, train=data_train, test=data_test, K=4, data_movies=data_movie)
model3.fit(30)
with open('ModelK4','wb') as file:
    pickle.dump(model1, file)


print("---------------For k=8---------------")
model4 = Model.Model(tau=0.01, Lambda=0.05, train=data_train, test=data_test, K=8, data_movies=data_movie)
model4.fit(30)
with open('ModelK8','wb') as file:
    pickle.dump(model1, file)


print("---------------For k=16---------------")
model5 = Model.Model(tau=0.01, Lambda=0.05, train=data_train, test=data_test, K=16, data_movies=data_movie)
model5.fit(30)
with open('ModelK16','wb') as file:
    pickle.dump(model5, file)


sns.lineplot(model1.loss, label="K=0")
sns.lineplot(model2.loss, label="K=2")
sns.lineplot(model3.loss, label="K=4")
sns.lineplot(model4.loss, label="K=8")
sns.lineplot(model5.loss, label="K=16")
plt.legend()
plt.save('DATA/compare-plots.pdf',format='pdf')
plt.show()









# Movie_model = Model.Model(tau=0.5, Lambda=0.05, train=data_train, test=data_test, K=5, data_movies=data_movie)
# Movie_model = ModelFeatures.ModelFeatures(tau=0.5, Lambda=0.05, train=data_train, test=data_test, K=5, data_movies=data_movie)
# Movie_model.fit(30)

# with open('DataModelWithFeatures','wb') as file:
#     pickle.dump(Movie_model, file)

# Movie_model.plot_loss()
# Movie_model.plot_errors()
# Movie_model.plot_frequency_dist()