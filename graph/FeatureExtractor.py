import os
# import random
# import threading
# import multiprocessing
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse.linalg import svds
# from functools import partial
# from threading import Thread

# import time

# from pandarallel import pandarallel
# pandarallel.initialize(progress_bar=True)

import networkx as nx

class FeatureExtractor:
    def __init__(self, graph_type, type: str = 'basic', n_jobs: int = None) -> None:
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

        choice = {
            'basic': (1,0),
            'advanced':(0,1),
            'all':(1,1)
        }
        
        try:
            basic, advanced = choice[self.type]

        except KeyError:
            print("Invalid feature type")

        if basic:
            print("Extracting basic features")
            tqdm.pandas(desc='creating num_successors_source column')
            # X['num_successors_source'] = self.apply_parallel(X['source'], self.num_successors)
            X['num_successors_source'] = X.progress_apply(
                lambda row: self.num_successors(row['source']), axis=1)
            tqdm.pandas(desc='creating num_predecessors_source column')
            X['num_predecessors_source'] = X.progress_apply(
                lambda row: self.num_predecessors(row['source']),axis=1)
            tqdm.pandas(desc='creating num_successors_destination column')
            X['num_successors_destination'] = X.progress_apply(
                lambda row: self.num_successors(row['destination']),axis=1)
            tqdm.pandas(desc='creating num_predecessors_destination column')
            X['num_predecessors_destination'] = X.progress_apply(
                lambda row: self.num_predecessors(row['destination']),axis=1)                                     
            # data['follows'] = data.apply(
            #   lambda row: self.follows(row['source'],row['destination']),axis=1)
            
            tqdm.pandas(desc='creating back_link column')
            X['back_link'] = X.progress_apply(
                lambda row: self.back_link(row['source'], row['destination']),axis=1)

            tqdm.pandas(desc='creating jaccard_successors column')
            X['jaccard_successors'] = X.progress_apply(
                lambda row: self.jaccard_successors(row['source'],row['destination']),axis=1)

            tqdm.pandas(desc='creating jaccard_predecessors column')
            X['jaccard_predecessors'] = X.progress_apply(
                lambda row: self.jaccard_predecessors(row['source'],row['destination']),axis=1)

            tqdm.pandas(desc='creating successors_predecessors_ratio_src column')
            X['successors_predecessors_ratio_src'] = X.progress_apply(
                lambda row: self.successors_predecessors_ratio(row['source']),axis=1)

            tqdm.pandas(desc='creating successors_predecessors_ratio_dst column')
            X['successors_predecessors_ratio_dst'] = X.progress_apply(
                lambda row: self.successors_predecessors_ratio(row['destination']),axis=1)

            tqdm.pandas(desc='creating shortest_path_bw_x_and_y column')
            X['shortest_path_bw_x_and_y'] = X.progress_apply(
                lambda row: self.shortest_path(row['source'],row['destination']),axis=1)

        #     tqdm.pandas(desc='creating dice_index_successors column')
        #     X['dice_index_successors'] = X.progress_apply(
        #         lambda row: self.back_link(row['source'],row['destination']),axis=1)

        #     tqdm.pandas(desc='creating dice_index_predecessors column')
        #     X['dice_index_predecessors'] = [self.dice_index_for_predecessors(*a) for a in 
        #                                 tuple(zip(X["source"], X["destination"], self.graph))]

        #     tqdm.pandas(desc='creating hub_promoted_index_successors column')
        #     X['hub_promoted_index_successors'] = X.progress_apply(
        #         lambda row: self.hub_promoted_index_for_successors(
        #                         row['source'],row['destination']),axis=1)

        #     tqdm.pandas(desc='creating hub_promoted_index_predecessors column')
        #     X['hub_promoted_index_predecessors'] = X.progress_apply(
        #         lambda row: self.hub_promoted_index_for_predecessors(
        #             row['source'],row['destination']),axis=1)

        #     tqdm.pandas(desc='creating hub_depressed_index_successors column')
        #     X['hub_depressed_index_successors'] = X.progress_apply(
        #         lambda row: self.hub_depressed_index_for_successors(
        #             row['source'],row['destination']),axis=1)

        #     tqdm.pandas(desc='creating hub_depressed_index_predecessors column')
        #     X['hub_depressed_index_predecessors'] = X.progress_apply(
        #         lambda row: self.hub_depressed_index_for_predecessors(
        #             row['source'],row['destination']),axis=1)

        #     tqdm.pandas(desc='creating leicht_holme_index_for_successors column')
        #     X['leicht_holme_index_for_successors'] = X.progress_apply(
        #         lambda row: self.leicht_holme_index_for_successors(
        #             row['source'],row['destination']),axis=1)

        #     tqdm.pandas(desc='creating leicht_holme_index_for_predecessors column')
        #     X['leicht_holme_index_for_predecessors'] = X.progress_apply(
        #         lambda row: self.leicht_holme_index_for_predecessors(
        #             row['source'],row['destination']),axis=1)

        #     tqdm.pandas(desc='creating param_dependent_index_for_successors column')
        #     X['param_dependent_index_for_successors'] = X.progress_apply(
        #         lambda row: self.param_dependent_index_for_successors(
        #             row['source'],row['destination']),axis=1)

        #     tqdm.pandas(desc='creating param_dependent_index_for_predecessors column')
        #     X['param_dependent_index_for_predecessors'] = X.progress_apply(
        #         lambda row: self.param_dependent_index_for_predecessors(
        #             row['source'],row['destination']),axis=1)

        #     tqdm.pandas(desc='creating cosine_successors column')
        #     X['cosine_successors'] = X.progress_apply(
        #         lambda row: self.cosine_for_successors(row['source'],row['destination']),axis=1)

        #     tqdm.pandas(desc='creating cosine_predecessors column')
        #     X['cosine_predecessors'] = X.progress_apply(
        #         lambda row: self.cosine_for_predecessors(row['source'],row['destination']),axis=1)
            
        # if advanced:
        #     print("Extracting Advanced features")
        #     katz = self.calculate_katz_centrality()
        #     hits = self.hits_score()
        #     #Page Rank
            
        #     # data['source_node_pagerank'] = data.source.map(lambda x: pr[x])
        #     # data['dest_node_pagerank'] = data.destination.map(lambda x: pr[x])

        #     #Katz Centrality
        #     mean_katz = float(sum(katz.values())) / len(katz)
        #     tqdm.pandas(desc='creating param_dependent_index_for_successors column')
        #     X['katz_source'] = X.source.progress_apply(lambda x: katz.get(x,mean_katz))

        #     tqdm.pandas(desc='creating param_dependent_index_for_successors column')
        #     X['katz_destination'] = X.destination.progress_apply(lambda x: katz.get(x,mean_katz))

        #     #Weakly Connected Components
        #     tqdm.pandas(desc='creating param_dependent_index_for_successors column')
        #     X['same_component'] = X.progress_apply(lambda row: self.belongs_to_same_wcc(row['source'],row['destination']),axis=1)
            
        #     #Hits Score
        #     tqdm.pandas(desc='creating param_dependent_index_for_successors column')
        #     X['hits_source'] = X.source.progress_apply(lambda x: hits[0].get(x,0))

        #     tqdm.pandas(desc='creating param_dependent_index_for_successors column')
        #     X['hits_destination'] = X.destination.progress_apply(lambda x: hits[0].get(x,0))

        #     tqdm.pandas(desc='creating param_dependent_index_for_successors column')
        #     X['authorities_source'] = X.source.progress_apply(lambda x: hits[1].get(x,0))

        #     tqdm.pandas(desc='creating param_dependent_index_for_successors column')
        #     X['authorities_destination'] = X.destination.progress_apply(lambda x: hits[1].get(x,0))

        #     #Weighted Features
        #     weight_list = self.weighted_features()
        #     tqdm.pandas(desc='creating param_dependent_index_for_successors column')
        #     X['weight_in'] = X.source.progress_apply(lambda x: weight_list[0].get(x,weight_list[2]))

        #     tqdm.pandas(desc='creating param_dependent_index_for_successors column')
        #     X['weight_out'] = X.destination.progress_apply(lambda x: weight_list[1].get(x,weight_list[3]))

        #     #feature engineering on with weighted features for more features
        #     print("\nMaking more features with 'weight_in' and 'weight_out'\n")
        #     X['weight_f1'] = X.weight_in + X.weight_out
        #     X['weight_f2'] = X.weight_in * X.weight_out
        #     X['weight_f3'] = (2*X.weight_in + 1*X.weight_out)
        #     X['weight_f4'] = (1*X.weight_in + 2*X.weight_out)

        #     print("Done with weighted features!\n")

        #     #for svd features to get feature vector creating a dict node val and inedx in svd vector
        #     ### With the help of SVD we can have 24 features to predict link between 2 nodes.
        #     print("Making more features using SVD")
        #     sadj_col = sorted(self.graph.nodes())
        #     sadj_dict = {val : idx for idx, val in enumerate(sadj_col)}
        #     Adj = nx.adjacency_matrix(self.graph,nodelist=sorted(self.graph.nodes())).asfptype()

        #     U, s, V = svds(Adj, k = 6)

        #     tqdm.pandas(desc='creating svd features')
        #     X[['svd_u_s_1', 'svd_u_s_2','svd_u_s_3', 'svd_u_s_4', 'svd_u_s_5', 'svd_u_s_6']] = \
        #     X.source.progress_apply(lambda x: self.svd(x, U, sadj_dict)).progress_apply(pd.Series)
            
        #     X[['svd_u_d_1', 'svd_u_d_2', 'svd_u_d_3', 'svd_u_d_4', 'svd_u_d_5','svd_u_d_6']] = \
        #     X.destination.progress_apply(lambda x: self.svd(x, U, sadj_dict)).progress_apply(pd.Series)
        #     #===================================================================================================
            
        #     X[['svd_v_s_1','svd_v_s_2', 'svd_v_s_3', 'svd_v_s_4', 'svd_v_s_5', 'svd_v_s_6',]] = \
        #     X.source.progress_apply(lambda x: self.svd(x, V.T, sadj_dict)).progress_apply(pd.Series)

        #     X[['svd_v_d_1', 'svd_v_d_2', 'svd_v_d_3', 'svd_v_d_4', 'svd_v_d_5','svd_v_d_6']] = \
        #     X.destination.progress_apply(lambda x: self.svd(x, V.T, sadj_dict)).progress_apply(pd.Series)

        #     # Preferential Attachment
        #     source = np.array(X["num_predecessors_source"])
        #     destination = np.array(X["num_predecessors_destination"])
        #     predecessors_preferential_source = []

        #     for i in tqdm(range(len(source)), 
        #             desc="creating predecessors preferential attachment features"):
        #         predecessors_preferential_source.append(source[i]*destination[i])

        #     X["predecessors_preferential_attach"]  = predecessors_preferential_source 

        #     source = np.array(X["num_successors_source"])
        #     destination = np.array(X["num_successors_destination"])
        #     successors_preferential_destination = []

        #     for i in tqdm(range(len(source)),
        #                 desc="creating successors preferential attachment features"):
        #         successors_preferential_destination.append(source[i]*destination[i])

        #     X["successors_preferential_attach"]  = successors_preferential_destination

        

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

    def successors_predecessors_ratio(self,x):
        try:
            num_successors = len(set(self.graph.successors(x)))
            num_predecessors = len(set(self.graph.predecessors(x)))
            if num_successors == 0 | num_predecessors == 0:
                return 0
            r = num_successors/num_predecessors
            return r
        except:
            return 0

    # def successors_predecessors_ratio_destination(self,x):
    #     try:
    #         num_successors = len(set(self.graph.successors(x)))
    #         num_predecessors = len(set(self.graph.predecessors(x)))
    #         if num_successors == 0 | num_predecessors == 0:
    #             return 0
    #         r = len(set(self.graph.successors(x)))/len(set(self.graph.predecessors(x)))
    #         return r 
    #     except:
    #         return 0
    
    def shortest_path(self,x,y):
        # p = -1
        try:
            if self.graph.has_edge(x,y):
                self.graph.remove_edge(x,y)
                p = nx.shortest_path_length(self.graph, source= x, target= y)
                self.graph.add_edge(x,y)
            else:
                p = nx.shortest_path_length(self.graph, source=x, target=y)
            return p
        except:
            return -1
