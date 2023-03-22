import csv
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import numpy as np
import pandas as pd
import tqdm

import networkx as nx
from sklearn.model_selection import train_test_split

class PrepareData:
    def __init__(
            self, path: str, 
            graph_type:str, 
            test_size: float = 0.2, 
            n_jobs: int = 1,
        ) -> None:

        self.path = path
        self.test_size = test_size
        self.n_jobs = n_jobs
        self.graph_type = graph_type
        self.temp_path = os.getcwd()+"/temp"
        self.data = pd.read_csv(self.path)
        # self.graph = self.create_graph()
    
    def read_graph(self):
        # data = pd.read_csv(self.path)
        if "Unnamed: 0" in self.data.columns:
            self.data.drop("Unnamed: 0", axis=1, inplace=True)
        data_noheader = self.data.rename(columns=self.data.iloc[0].astype(int))\
                        .drop(self.data.index[0]).reset_index(drop=True)

        return data_noheader

    def create_graph(self):
        print("Creating graph, It may take some time.")
        data = self.read_graph()

        if not os.path.isdir(self.temp_path):
            os.makedirs(self.temp_path)

        # if not os.path.isfile(self.temp_path+'/train_data.csv'):
        data.to_csv(self.temp_path+'/train_data.csv', index=False)

        if self.graph_type == 'directed':
            graph = nx.read_edgelist(
                self.temp_path+'/train_data.csv',
                delimiter=',',
                create_using=nx.DiGraph()
                )

            return graph

        else:
            graph = nx.read_edgelist(
                self.temp_path+'/train_data.csv',
                delimiter=',',
                create_using=nx.Graph()
                )

            return graph

    def prepare(self):
        # data = pd.read_csv(self.path)
        data = self.data
        data_len = len(data)
        if "Unnamed: 0" in data.columns:
            data.drop("Unnamed: 0", axis=1, inplace=True)

        graph = self.create_graph()
        _csv = csv.reader(open(self.temp_path+"/train_data.csv", "r"))

        _edges = {}

        for node in tqdm.tqdm(_csv, total=data_len):
            _edges[(node[0], node[1])] = 1

        missing_edges = set([])
        max_node = max(max(data[data.columns[0]]), 
                        max(data[data.columns[1]]))

        def utility_func(data):
            while (len(missing_edges)) < data_len:
                
                a = random.randint(1, max_node)
                b = random.randint(1, max_node)

                tmp = _edges.get((a, b), -1)

                if tmp == -1 and a!=b:
                    try:
                        if nx.shortest_path(x = a,y = b, graph = graph) > 2:
                            missing_edges.add((a,b))
                        else:
                            continue
                    except:
                        missing_edges.add((a,b))
                else:
                    continue

            return None
            
        threads = []

        for k in range(self.n_jobs):
            data_frac = data_len//self.n_jobs
            if k == self.n_jobs - 1:
                arg = [data[data_frac*k : data_len]]
            else:
                arg = [data[data_frac*k : data_frac*(k+1)]]

            p = threading.Thread(
                    target=utility_func, 
                    args= arg
                )
            
            p.start()
            threads.append(p)

        for thread in threads:
            thread.join()

        df_pos = data
        df_neg = pd.DataFrame(missing_edges, columns = ['source_node', 'destination_node'])

        y_pos = np.ones(len(df_pos), dtype=int)
        y_neg = np.zeros(len(df_neg), dtype=int)
        
        X_train_pos, X_test_pos, y_train_pos, y_test_pos = \
            train_test_split(df_pos, y_pos , test_size=self.test_size)
        X_train_neg, X_test_neg, y_train_neg, y_test_neg = \
            train_test_split(df_neg, y_neg , test_size=self.test_size)

        X_train = pd.concat([X_train_pos, X_train_neg], ignore_index= True).reset_index(drop=True)
        y_train = pd.DataFrame(np.concatenate((y_train_pos, y_train_neg)).astype(int), columns=['label'])
        X_test = pd.concat([X_test_pos,X_test_neg], ignore_index= True).reset_index(drop=True)
        y_test = pd.DataFrame(np.concatenate((y_test_pos, y_test_neg)).astype(int), columns=['label'])

        return X_train,X_test, y_train, y_test
        # return missing_edges

                


