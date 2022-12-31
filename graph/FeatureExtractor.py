import csv
import os
import random
import threading
import multiprocessing
import numpy as np
import pandas as pd
import tqdm

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

        data = data.rename(columns=data.iloc[0].astype(int)).drop(data.index[0]).reset_index(drop=True)

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


    def transform(self, X):
        X['source'] = X['source'].astype(str)
        X['destination'] = X['destination'].astype(str)

        X['num_successors_source'] = X.apply(lambda row: self.num_successors((row['source'])),axis=1)
        X['num_predecessors_source'] = X.apply(lambda row: self.num_predecessors(row['source']),axis=1)
        X['num_successors_destination'] = X.apply(lambda row: self.num_successors(row['destination']),axis=1)
        X['num_predecessors_destination'] = X.apply(lambda row: self.num_predecessors(row['destination']),axis=1)                                     
        # data['follows'] = data.apply(lambda row: self.follows(row['source'],row['destination']),axis=1)
        X['back_link'] = X.apply(lambda row: self.back_link(row['source'],row['destination']),axis=1)

        X['jaccard_successors'] = X.apply(lambda row: self.\
            jaccard_successors(row['source'],row['destination']),axis=1)

        X['jaccard_predecessors'] = X.apply(lambda row: self.\
            jaccard_predecessors(row['source'],row['destination']),axis=1)

        return X

    def jaccard_successors(self,x,y):
        try:
            if len(set(self.graph.successors(x))) == 0 | len(set(self.graph.successors(y))) == 0:
                return 0
            sim = (len(set(self.graph.successors(x)).intersection(set(self.graph.successors(y)))))\
            /(len(set(self.graph.successors(x)).union(set(self.graph.successors(y)))))
            return sim
        except:
            return 0
        

    def jaccard_predecessors(self,x,y):
        try:
            if len(set(self.graph.predecessors(x))) == 0 | len(set(self.graph.predecessors(y))) == 0:
                return 0
            sim = (len(set(self.graph.predecessors(x)).intersection(set(self.graph.predecessors(y)))))\
            /(len(set(self.graph.predecessors(x)).union(set(self.graph.predecessors(y)))))
            return sim

        except:
            return 0

    def num_successors(self,x):
        try:
            if len(set(self.graph.successors(x))) == 0:
                return 0
            return len(set(self.graph.successors(x)))
        except:
            return 0


    def num_predecessors(self,x):
        try:
            if len(set(self.graph.predecessors(x))) == 0:
                return 0
            return len(set(self.graph.predecessors(x)))
        except:
            return 0
        
    def back_link(self,x,y):
        if self.graph.has_edge(y,x):
            return 1
        else:
            return 0
