import os
# import random
# import threading
# import multiprocessing
import numpy as np
import pandas as pd
from tqdm import tqdm
# from functools import partial
# from threading import Thread

# import time

# from pandarallel import pandarallel
# pandarallel.initialize(progress_bar=True)

import networkx as nx

class FeatureExtractor:
    def __init__(self, graph_type, type: str = None, n_jobs: int = None) -> None:
        self.type = type
        self.n_jobs = n_jobs
        self.temp_path = os.getcwd()+"/temp"
        self.graph_type = graph_type

        self.graph = None

    def fit(self, X, y):
        data = pd.concat((X, y), axis=1)
        data = data[data['label'] == 1]
        data.drop('label', axis=1, inplace=True)

        if "Unnamed: 0" in data.columns:
            data.drop("Unnamed: 0", axis=1, inplace=True)

        if data.shape[1] > 2:
            raise ValueError

        data = data.rename(columns=data.iloc[0].astype(int))\
                .drop(data.index[0]).reset_index(drop=True)

        if not os.path.isdir(self.temp_path):
            os.makedirs(self.temp_path)

        if not os.path.isfile(self.temp_path+'/tmp_train_data.csv'):
            data.to_csv(self.temp_path+'/tmp_train_data.csv', index=False)

        if self.graph_type == 'directed':
            self.graph = nx.read_edgelist(
                self.temp_path+'/tmp_train_data.csv',
                delimiter=',',
                create_using=nx.DiGraph()
                )

        else:
            self.graph = nx.read_edgelist(
                self.temp_path+'/tmp_train_data.csv',
                delimiter=',',
                create_using=nx.Graph()
                )

    # def apply_parallel(self, data, fun):
    #     total_res = []

    #     def partial(data):
    #         try:
    #             if data.shape[1] == 2:
    #                 result = data.apply(lambda x: fun(x['source'], x['destination']), axis=1)
    #                 total_res.extend(list(result))

    #         except IndexError:
    #             result = data.apply(fun)
    #             total_res.extend(list(result))

    #     threads = []
    #     for k in range(4):
    #         data_frac = len(data)//4
    #         if k == 4 - 1:
    #             arg = [data[data_frac*k : len(data)]]
    #         else:
    #             arg = [data[data_frac*k : data_frac*(k+1)]]

    #         t = Thread(target=partial, args= arg)

    #         t.start()
    #         threads.append(t)

    #     for thread in threads:
    #         thread.join()

    #     return total_res

    def transform(self, X):
        X = X.astype(str)

        tqdm.pandas(desc='creating num_successors_source column')
        # X['num_successors_source'] = self.apply_parallel(X['source'], self.num_successors)
        X['num_successors_source'] = X.progress_apply(lambda row: self.num_successors(row['source']), axis=1)
        tqdm.pandas(desc='creating num_predecessors_source column')
        X['num_predecessors_source'] = X.progress_apply(lambda row: self.num_predecessors(row['source']),axis=1)
        tqdm.pandas(desc='creating num_successors_destination column')
        X['num_successors_destination'] = X.progress_apply(lambda row: self.num_successors(row['destination']),axis=1)
        tqdm.pandas(desc='creating num_predecessors_destination column')
        X['num_predecessors_destination'] = X.progress_apply(lambda row: self.num_predecessors(row['destination']),axis=1)                                     
        # data['follows'] = data.apply(lambda row: self.follows(row['source'],row['destination']),axis=1)
        tqdm.pandas(desc='creating back_link column')
        X['back_link'] = X.progress_apply(lambda row: self.back_link(row['source'], row['destination']),axis=1)

        tqdm.pandas(desc='creating jaccard_successors column')
        X['jaccard_successors'] = X.progress_apply(lambda row: self.\
            jaccard_successors(row['source'],row['destination']),axis=1)

        tqdm.pandas(desc='creating jaccard_predecessors column')
        X['jaccard_predecessors'] = X.progress_apply(lambda row: self.\
            jaccard_predecessors(row['source'],row['destination']),axis=1)

        return X.astype(int)

    def jaccard_successors(self,x,y):
        try:
            x_successors = set(self.graph.successors(x))
            y_successors = set(self.graph.successors(y))

            if len(x_successors) == 0 | len(y_successors) == 0:
                return 0
            sim = (len(x_successors.intersection(y_successors)))\
                    /(len(x_successors.union(y_successors)))
            return sim
        except:
            return 0
        

    def jaccard_predecessors(self,x,y):
        try:
            x_predecessors = set(self.graph.predecessors(x))
            y_predecessors = set(self.graph.predecessors(y))

            if len(x_predecessors) == 0 | len(y_predecessors) == 0:
                return 0
            sim = (len(x_predecessors.intersection(y_predecessors)))\
                /(len(x_predecessors.union(y_predecessors)))
            return sim

        except:
            return 0

    def num_successors(self,x):
        try:
            successors = len(set(self.graph.successors(x)))
            if successors == 0:
                return 0
            return successors
        except:
            return 0


    def num_predecessors(self,x):
        try:
            predecessors = len(set(self.graph.predecessors(x)))
            if predecessors == 0:
                return 0
            return predecessors
        except:
            return 0
        
    def back_link(self,x,y):
        if self.graph.has_edge(y,x):
            return 1
        else:
            return 0
