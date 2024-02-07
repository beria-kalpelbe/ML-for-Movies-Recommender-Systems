import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import multiprocessing



class Model:
    def __init__(self, tau, Lambda, train, test, K,data_movies):
        self.train = train
        self.test = test
        self.map_user_to_index = self.map_index_to_user = self.map_movie_to_index = self.map_index_to_movie = self.data_by_user = self.data_by_movie = np.array([])
        self.map_user_to_index_test = self.map_index_to_user_test = self.map_movie_to_index_test = self.map_index_to_movie_test = self.data_by_user_test = self.data_by_movie_test = np.array([])
        self.Lambda = Lambda
        self.tau = tau
        self.K = K
        self.M = self.N = self.U = self.V = self.user_biases = self.movie_biases = self.loss = self.data_movies = self.features_parameters = None
        self.data_movies = data_movies
        
    
    def _sparse_data(self,data):
        map_user_to_index = {}
        map_index_to_user = []
        map_movie_to_index = {}
        map_index_to_movie = []
        data_by_user = []
        data_by_movie = []
        for user_id, movie_id, rating in data:
            if user_id not in map_user_to_index.keys():
                index_user = len(map_index_to_user)
                map_index_to_user.append(user_id)
                data_by_user.append([])
                map_user_to_index[user_id] = index_user
            else:
                index_user = map_user_to_index[user_id]
            if movie_id not in map_movie_to_index.keys():
                index_movie = len(map_index_to_movie)
                map_index_to_movie.append(movie_id)
                data_by_movie.append([])
                map_movie_to_index[movie_id] = index_movie
            else:
                index_movie = map_movie_to_index[movie_id]
            data_by_user[index_user].append((index_movie, rating))
            data_by_movie[index_movie].append((index_user, rating))
        return map_user_to_index, map_index_to_user, map_movie_to_index, map_index_to_movie, data_by_user, data_by_movie

    
    def plot_frequency_dist(self, savePDF = '', saveSVG = ''):
        if len(self.data_by_user) == 0:
            raise ValueError('Fit your model first !')
        user_count = []
        for element in self.data_by_user:
            user_count.append(len(element))

        user_count_unique = np.unique(user_count)
        user_count_count = []
        for element in user_count_unique:
            user_count_count.append(sum(user_count==element))
        
        movie_count = []
        for element in self.data_by_movie:
            movie_count.append(len(element))

        movie_count_unique = np.unique(movie_count)
        movie_count_count = []
        for element in movie_count_unique:
            movie_count_count.append(sum(movie_count==element))

        sns.scatterplot(x=user_count_unique, y=user_count_count, label="user", color="blue")
        sns.scatterplot(x=movie_count_unique, y=movie_count_count, label="movie", color="red")
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True)
        plt.xlabel('degree')
        plt.ylabel('frequency')
        if savePDF:
            plt.savefig(savePDF+'.pdf', format='pdf')
        if saveSVG:
            plt.savefig(saveSVG+'.svg', format='svg')
        plt.show()

    def _update_biases(self,biases, biases2, data, param1, param2):#args):
        # biases, data, param1, param2, i = args
        for i in range(len(data)):
            bias = 0
            for (j,r) in data[i]:
                bias += self.Lambda * (float(r) - np.dot(param1[i], param2[j]) -  biases[j])
            biases2[i] = bias / (self.Lambda * len(data[i]) + self.tau)
        return biases2

    def _update_parameters(self,param1, param2,data, biases1, biases2):# args):
        # param1, param2,data, biases1, biases2, i = args
        for i in range(len(param1)):
            s1 = np.zeros((self.K,self.K))
            s2 = np.zeros((self.K))
            for (j,r) in data[i]:
                s1 += np.outer(param2[j], param2[j])
                s2 += param2[j] * (r - biases1[i] - biases2[j])
            s1 = self.Lambda * s1 + self.tau*np.eye(self.K)
            s2 = self.Lambda * s2
            param1[i] = np.linalg.solve(s1, s2)
        return param1

    # Calculate loss without features
    def _calculate_loss(self):
        s1 = s2 = s3 = count = 0
        for user in range(len(self.data_by_user)):
            for (movie,r) in self.data_by_user[user]:
                rmn = np.dot(self.U[user,:], self.V[movie,:])
                s1 += (r - (rmn + self.user_biases[user] + self.movie_biases[movie]))**2
                count += 1
        #     s2 += np.dot(self.U[user], self.U[user])
        # for movie in range(self.V.shape[0]):
        #     s3 += np.dot(self.V[movie,:], self.V[movie,:])
        # L = self.Lambda*s1/2 + self.tau*s2/2 + self.tau*s3/2 + self.tau*(sum(self.user_biases**2) + sum(self.movie_biases**2))/2
        rmse = math.sqrt(s1/count)
        return rmse#L, rmse

    def _calculate_test_errors(self):
        s1 = s2 = s3 = count = 0
        for user in range(len(self.data_by_user_test)): 
            for (movie, r) in self.data_by_user_test[user]:
                if self.map_index_to_user_test[user] in self.map_index_to_user:
                    user_to_train = self.map_user_to_index[self.map_index_to_user_test[user]]                
                    if sum(self.map_index_to_movie == self.map_index_to_movie_test[movie]):
                        movie_to_train = self.map_movie_to_index[self.map_index_to_movie_test[movie]]
                        rmn = np.dot(self.U[user_to_train,:], self.V[movie_to_train,:])
                        mbias = self.movie_biases[movie]
                    else:
                        rmn = mbias = um_um = 0
                    ubias = self.user_biases[user_to_train]
                else:
                    rmn = ubias = um_um = 0
                if sum(self.map_index_to_movie == self.map_index_to_movie_test[movie]):
                    movie_to_train = self.map_movie_to_index[self.map_index_to_movie_test[movie]]
                    mbias = self.movie_biases[movie_to_train]
                s1 += (r-(rmn + ubias + mbias)) ** 2
                count += 1

        return math.sqrt(s1/count)
    
    def fit(self, epochs=10):
        self.map_user_to_index, self.map_index_to_user, self.map_movie_to_index, self.map_index_to_movie,self.data_by_user, self.data_by_movie = self._sparse_data(self.train)
        self.map_user_to_index_test, self.map_index_to_user_test, self.map_movie_to_index_test, self.map_index_to_movie_test,self.data_by_user_test, self.data_by_movie_test = self._sparse_data(self.test)

        
        self.M = len(np.concatenate((self.train[:,0], self.test[:,0])))
        self.N = len(self.data_movies)
        self.U = np.random.rand(len(self.data_by_user), self.K)
        self.V = np.random.rand(len(self.data_by_movie), self.K)

        self.user_biases = np.zeros((self.M))
        self.movie_biases = np.zeros((self.N))

        self.loss = []
        self.train_RMSE = []
        self.test_RMSE = []
        self.test_loss = []        
        for _ in tqdm(range(epochs), desc="Fitting model..."):
            # Update parameters
            U = self._update_parameters(self.U, self.V,self.data_by_user, self.user_biases, self.movie_biases)
            V = self._update_parameters(self.V,self.U,self.data_by_movie, self.movie_biases, self.user_biases)
            # Update biases
            self.user_biases = self._update_biases(self.movie_biases,self.user_biases, self.data_by_user,self.U,self.V)
            self.movie_biases = self._update_biases(self.user_biases, self.movie_biases, self.data_by_movie,self.V,self.U)
            # Calculate loss and error
            # lost,error = self._calculate_loss()
            error = self._calculate_loss()
            # self.loss.append(lost)
            self.train_RMSE.append(error)
            # err = self._calculate_test_errors()
            # self.test_RMSE.append(err)
            
    def plot_loss(self, savePDF = '', saveSVG = ''):
        plt.figure(figsize=(12, 6))
        sns.lineplot(self.loss)
        plt.xlim(-1, len(self.loss))
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.grid(True)
        if savePDF != '':
            plt.savefig(savePDF+'.pdf', format='pdf')
        if saveSVG != '':
            plt.savefig(saveSVG+'.svg', format='svg')
        plt.show()

    def plot_errors(self, savePDF = '', saveSVG = ''):
        sns.lineplot(self.train_RMSE, label="Train set")
        sns.lineplot(self.test_RMSE, label="Test set")
        plt.xlim(-1, len(self.train_RMSE))
        plt.xlabel("Epochs")
        plt.ylabel("RMSE")
        plt.grid(True)
        plt.legend()
        if savePDF != '':
            plt.savefig(savePDF+'.pdf', format='pdf')
        if saveSVG != '':
            plt.savefig(saveSVG+'.svg', format='svg')
        plt.show()

    def parameters(self):
        return {'user_parameters':self.U, 'mitem_parameters':self.V, 'user_biases':self.user_biases, 'item_biases':self.movie_biases}
    
    def predict(self, Id_user=None):
        prediction = {}
        if Id_user==None:
            return 0
        elif Id_user in self.map_index_to_user:
            local_user = self.map_user_to_index[Id_user]
            for movie in self.map_index_to_movie:
                local_movie = self.map_movie_to_index[movie]
                rmn = np.dot(self.U[local_user], self.V[local_movie]) + self.movie_biases[local_movie]
                name_movie = self.data_movies[self.data_movies[:,0] == str(int(movie)),1][0]
                prediction[name_movie] = rmn
        return prediction
