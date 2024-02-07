import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import multiprocessing

class ModelFeatures:
    def __init__(self, tau, Lambda, train, test, K,data_movies):
        self.train = train
        self.test = test
        self.map_user_to_index = self.map_index_to_user = self.map_movie_to_index = self.map_index_to_movie = self.data_by_user = self.data_by_movie = np.array([])
        self.map_user_to_index_test = self.map_index_to_user_test = self.map_movie_to_index_test = self.map_index_to_movie_test = self.data_by_user_test = self.data_by_movie_test = np.array([])
        self.f_map_movie_to_index = self.f_map_index_to_movie = self.f_map_genres_to_index = self.f_map_index_to_genres = self.f_data_genres_by_movie = self.f_data_movie_by_genres = None
        self.Lambda = Lambda
        self.tau = tau
        self.K = K
        self.M = self.N = self.U = self.V = self.user_biases = self.movie_biases = self.loss = self.data_movies = self.features_parameters = None
        self.data_movies = data_movies
        self.F = None
        self.features = np.unique(('|'.join(self.data_movies[:,2].astype(str).tolist())).split("|")).tolist()
        self.map_features = {}

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
    

    def _update_biases(self,biases, data, param1, param2):#args):
        # biases, data, param1, param2, i = args
        for i in range(len(data)):
            biases2 = np.zeros((len(data)))
            bias = 0
            for (j,r) in data[i]:
                bias += self.Lambda * (float(r) - np.dot(param1[i], param2[j]) -  biases[j])
            biases2[i] = bias / (self.Lambda * len(data[i]) + self.tau)
        return biases2

    def _updateU(self):
        for i in range(len(self.data_by_user)):
            s1 = np.zeros((self.K,self.K))
            s2 = np.zeros((self.K))
            for (j,r) in self.data_by_user[i]:
                s1 += np.outer(self.V[j], self.V[j])
                s2 += self.V[j] * (r - self.user_biases[i] - self.movie_biases[j])
            s1 = self.Lambda * s1 + self.tau*np.eye(self.K)
            s2 = self.Lambda * s2
            self.U[i] = np.linalg.solve(s1, s2)
    
    def _updateV(self):
        for i in range(len(self.data_by_movie)):
            s1 = np.zeros((self.K,self.K))
            s2 = s3 = np.zeros((self.K))
            for (j,r) in self.data_by_movie[i]:
                s1 += np.outer(self.U[j], self.U[j])
                s2 += self.U[j] * (r - self.user_biases[j] - self.movie_biases[i])
            count = 0
            features = self.f_data_genres_by_movie[self.f_map_movie_to_index[str(int(self.map_index_to_movie[i]))]][0][1].split("|")
            for f in features:
                s3 += self.F[self.map_features[f]]
                count += 1
            s1 = self.Lambda * s1 + self.tau*np.eye(self.K)
            s2 = self.Lambda * s2
            s3 = self.tau * s3 /np.sqrt(count)
            self.V[i] = np.linalg.solve(s1, (s2 + s3))
    
    def _updateF(self):
        for i in range(len(self.features)):
            s1 = s2 = 0
            for movie in range(len(self.data_by_movie)):
                features = self.f_data_genres_by_movie[self.f_map_movie_to_index[str(int(self.map_index_to_movie[movie]))]][0][1].split("|")
                s3 = 0
                if self.features[i] in features:
                    for f in self.features:
                        if i != self.map_features[f]:
                            s3 += self.F[self.map_features[f]]
                    s1 += self.V[movie]/np.sqrt(len(features)) - (s3/len(features))
                    s2 += 1/len(features)
            self.F[i] = s1/(1+s2)
    
    def _calculate_loss(self):
        s1 = s2 = s3 = count = s5 = 0
        for user in range(len(self.data_by_user)):
            for (movie,r) in self.data_by_user[user]:
                rmn = np.dot(self.U[user,:], self.V[movie,:])
                s1 += (r - (rmn + self.user_biases[user] + self.movie_biases[movie]))**2
                count += 1
            s2 += np.dot(self.U[user], self.U[user])
        for movie in range(self.V.shape[0]):
            features_movie = self.f_data_genres_by_movie[self.f_map_movie_to_index[str(int(self.map_index_to_movie[movie]))]][0][1].split("|")
            s4 = 0
            for f in features_movie:
                s4 += self.F[self.map_features[f]]
            s3 += np.dot((self.V[movie,:] - s4/np.sqrt(len(features_movie))), (self.V[movie,:] - s4/np.sqrt(len(features_movie))))
        
        for f in self.features:
            s5 += np.dot(self.F[self.map_features[f]], self.F[self.map_features[f]])

        L = self.Lambda*s1/2 + self.tau*s2/2 + self.tau*s3/2 + self.tau*s5/2 + self.tau*(sum(self.user_biases**2) + sum(self.movie_biases**2))/2
        rmse = math.sqrt(s1/count)
        return L, rmse

    def _calculate_test_errors(self):
        s1 = count = 0
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
                    # um_um = np.dot(self.U[user_to_train,:], self.U[user_to_train,:])
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
        self.f_map_movie_to_index, self.f_map_index_to_movie, self.f_map_genres_to_index, self.f_map_index_to_genres, self.f_data_genres_by_movie, self.f_data_movie_by_genres = self._sparse_data(self.data_movies)
        
        for i in range(len(self.features)):
            self.map_features[self.features[i]] = i

        self.M = len(np.unique(np.vstack((self.train, self.test))[:,0]))
        self.N = len(np.unique(np.vstack((self.train, self.test))[:,1]))
        self.U = np.random.rand(len(self.data_by_user), self.K)
        self.V = np.random.rand(len(self.data_by_movie), self.K)
        self.F = np.random.rand(len(self.features), self.K)

        self.user_biases = np.zeros((self.M))
        self.movie_biases = np.zeros((self.N))

        self.loss = []
        self.train_RMSE = []
        self.test_RMSE = []
        self.test_loss = []        
        for _ in tqdm(range(epochs), desc="Fitting model..."):
            # Update parameters
            self.user_biases = self._update_biases(self.movie_biases, self.data_by_user,self.U,self.V)
            self._updateU()
            self.movie_biases = self._update_biases(self.user_biases, self.data_by_movie,self.V,self.U)
            self._updateV()
            self._updateF()
            # Update biases
            # Calculate loss and error
            lost,error = self._calculate_loss()
            self.loss.append(lost)
            self.train_RMSE.append(error)
            err = self._calculate_test_errors()
            self.test_RMSE.append(err)
            

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
        return {'user_parameters':self.U, 'movie_parameters':self.V,'features_parameters':self.F, 'user_biases':self.user_biases, 'item_biases':self.movie_biases}
    
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

