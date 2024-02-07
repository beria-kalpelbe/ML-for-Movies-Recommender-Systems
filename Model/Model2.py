import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
from multiprocessing import cpu_count



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

    def _update_user_params(self,list_users):
        for i in list_users:
            bias = 0
            s1 = np.zeros((self.K,self.K))
            s2 = np.zeros((self.K))
            for (j,r) in self.data_by_user[i]:
                bias += self.Lambda * (float(r) - np.dot(self.U[i], self.V[j]) -  self.movie_biases[j])
                s1 += np.outer(self.V[j], self.V[j])
                s2 += self.V[j] * (r - self.user_biases[i] - self.movie_biases[j])
            self.user_biases[i] = bias / (self.Lambda * len(self.data_by_user[i]) + self.tau)
            s1 = self.Lambda * s1 + self.tau*np.eye(self.K)
            s2 = self.Lambda * s2
            self.U[i] = np.linalg.solve(s1, s2)
    
    def _update_movie_params(self,list_movies):
        for j in list_movies:
            bias = 0
            s1 = np.zeros((self.K,self.K))
            s2 = np.zeros((self.K))
            for (i,r) in self.data_by_movie[j]:
                bias += self.Lambda * (float(r) - np.dot(self.U[i], self.U[i]) -  self.user_biases[i])
                s1 += np.outer(self.U[i], self.U[i])
                s2 += self.U[i] * (r - self.user_biases[i] - self.movie_biases[j])
            self.movie_biases[j] = bias / (self.Lambda * len(self.data_by_movie[j]) + self.tau)
            s1 = self.Lambda * s1 + self.tau*np.eye(self.K)
            s2 = self.Lambda * s2
            self.V[j] = np.linalg.solve(s1, s2)        

    # Calculate loss without features
    def _calculate_loss(self, list_users):
        s1 = s2 = s3 = count = 0
        for user in list_users:
            for (movie,r) in self.data_by_user[user]:
                rmn = np.dot(self.U[user,:], self.V[movie,:])
                s1 += (r - (rmn + self.user_biases[user] + self.movie_biases[movie]))**2
                count += 1
            # s2 += np.dot(self.U[user], self.U[user])
        # for movie in range(self.V.shape[0]):
        #     s3 += np.dot(self.V[movie,:], self.V[movie,:])
        # L = self.Lambda*s1/2 + self.tau*s2/2 + self.tau*s3/2 + self.tau*(sum(self.user_biases**2) + sum(self.movie_biases**2))/2
        rmse = math.sqrt(s1/count)
        return rmse# L, rmse

    def _calculate_test_errors(self, list_users_tests):
        s1 = s2 = s3 = count = 0
        for user in list_users_tests: 
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
        rmse = math.sqrt(s1/count)
        return rmse
    
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

            
            list_users = list(range(len(self.data_by_user)))
            list_movies = list(range(len(self.data_by_movie)))
            # self._update_user_params(list_users)
            # self._update_movie_params(list_movies)
                

            # with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
            #     executor.submit(self._update_user_params, list_users)
            #     executor.submit(self._update_movie_params, list_movies)
            # print("-----end of parallelism-----")
            # Calculate loss and error
            # lost,
            # executor = ThreadPoolExecutor(max_workers=cpu_count())
            # errors = executor.submit(self._calculate_loss, list_users)
            # with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
            #     results = executor.submit(self._calculate_loss, list_users)
            # error = errors.result()
            
            # print("-------End of train error----------")
            list_users_test = list(range(len(self.data_by_user_test)))
            # error = self._calculate_loss(list_users_test)
            # self.train_RMSE.append(error)
            # self.loss.append(lost)
            list_users_test = list(range(len(self.data_by_user_test)))
            with ThreadPoolExecutor(max_workers=20) as executor:
                results = executor.submit(self._calculate_test_errors, list_users_test)
            err = results.result()
            # err = self._calculate_test_errors()
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
        # sns.lineplot(self.test_RMSE, label="Test set")
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
