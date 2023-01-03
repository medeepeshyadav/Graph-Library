from collections import defaultdict
import math
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
            print("This may take a while for big dataset.")
            # src_successors = self._find_successors(X['source'])
            # dest_successors = self._find_successors(X['destination'])
            # src_predecessors = self._find_predecessors(X['source'])
            # dest_predecessors = self._find_predecessors(X['destination'])
            
            tqdm.pandas(desc='creating num_successors_source column')
            # X['num_successors_source'] = self.apply_parallel(X['source'], self.num_successors)
            X['num_successors_source'] = X.progress_apply(
                lambda row: self.num_successors(
                    row['source'], 
                    # src_successors
                    ), axis=1)

            tqdm.pandas(desc='creating num_predecessors_source column')
            X['num_predecessors_source'] = X.progress_apply(
                lambda row: self.num_predecessors(
                    row['source'], 
                    # src_predecessors
                    ),axis=1)
            tqdm.pandas(desc='creating num_successors_destination column')
            X['num_successors_destination'] = X.progress_apply(
                lambda row: self.num_successors(
                    row['destination'], 
                    # dest_successors
                    ),axis=1)
            tqdm.pandas(desc='creating num_predecessors_destination column')
            X['num_predecessors_destination'] = X.progress_apply(
                lambda row: self.num_predecessors(
                    row['destination'], 
                    # dest_predecessors
                    ),axis=1)                                     
            # data['follows'] = data.apply(
            #   lambda row: self.follows(row['source'],row['destination']),axis=1)
            
            tqdm.pandas(desc='creating back_link column')
            X['back_link'] = X.progress_apply(
                lambda row: self.back_link(row['source'], row['destination']),axis=1)

            tqdm.pandas(desc='creating jaccard_successors column')
            X['jaccard_successors'] = X.progress_apply(
                lambda row: self.jaccard(
                    row['source'],
                    row['destination'], 
                    for_what='successors',
                    # src_successors, 
                    # dest_successors
                    ),axis=1)

            tqdm.pandas(desc='creating jaccard_predecessors column')
            X['jaccard_predecessors'] = X.progress_apply(
                lambda row: self.jaccard(
                    row['source'],
                    row['destination'],
                    for_what='pred'
                    # src_predecessors, 
                    # dest_predecessors
                    ),axis=1)

            tqdm.pandas(desc='creating successors_predecessors_ratio_src column')
            X['successors_predecessors_ratio_src'] = X.progress_apply(
                lambda row: self.successors_predecessors_ratio(row['source']),axis=1)

            tqdm.pandas(desc='creating successors_predecessors_ratio_dst column')
            X['successors_predecessors_ratio_dst'] = X.progress_apply(
                lambda row: self.successors_predecessors_ratio(row['destination']),axis=1)

            tqdm.pandas(desc='creating shortest_path_bw_x_and_y column')
            X['shortest_path_bw_x_and_y'] = X.progress_apply(
                lambda row: self.shortest_path(row['source'],row['destination']),axis=1)

            tqdm.pandas(desc='creating dice_index column')
            X['dice_index_successors'] = X.progress_apply(
                lambda row: self.dice_index(
                    row['source'],row['destination'], for_what='successors'),axis=1)

            tqdm.pandas(desc='creating dice_index_predecessors column')
            X['dice_index_predecessors'] = X.progress_apply(
                lambda row: self.dice_index(
                    row['source'],row['destination'], for_what='pred'), axis=1)

            tqdm.pandas(desc='creating hub_promoted_index column')
            X['hub_promoted_index'] = X.progress_apply(
                lambda row: self.hub_promoted_index(
                                row['source'],row['destination'], for_what='successors'),axis=1)

            tqdm.pandas(desc='creating hub_promoted_index_predecessors column')
            X['hub_promoted_index_predecessors'] = X.progress_apply(
                lambda row: self.hub_promoted_index(
                    row['source'],row['destination'], for_what='pred'),axis=1)

            tqdm.pandas(desc='creating hub_depressed_index column')
            X['hub_depressed_index'] = X.progress_apply(
                lambda row: self.hub_depressed_index(
                    row['source'],row['destination'], for_what='successors'),axis=1)

            tqdm.pandas(desc='creating hub_depressed_index_predecessors column')
            X['hub_depressed_index_predecessors'] = X.progress_apply(
                lambda row: self.hub_depressed_index(
                    row['source'],row['destination'], for_what='pred'),axis=1)

            tqdm.pandas(desc='creating leicht_holme_index_for_successors column')
            X['leicht_holme_index_for_successors'] = X.progress_apply(
                lambda row: self.leicht_holme_index(
                    row['source'],row['destination'], for_what='successors'),axis=1)

            tqdm.pandas(desc='creating leicht_holme_index_for_predecessors column')
            X['leicht_holme_index_for_predecessors'] = X.progress_apply(
                lambda row: self.leicht_holme_index(
                    row['source'],row['destination'], for_what='pred'),axis=1)

            tqdm.pandas(desc='creating param_dependent_index_for_successors column')
            X['param_dependent_index_for_successors'] = X.progress_apply(
                lambda row: self.param_dependent_index(
                    row['source'],row['destination'], for_what='successors'),axis=1)

            tqdm.pandas(desc='creating param_dependent_index_for_predecessors column')
            X['param_dependent_index_for_predecessors'] = X.progress_apply(
                lambda row: self.param_dependent_index(
                    row['source'],row['destination'], for_what='pred'),axis=1)

            tqdm.pandas(desc='creating cosine_successors column')
            X['cosine_successors'] = X.progress_apply(
                lambda row: self.cosine(
                    row['source'],
                    row['destination'],
                    for_what='successors' 
                    # src_successors, 
                    # dest_successors
                    ),axis=1)

            tqdm.pandas(desc='creating cosine_predecessors column')
            X['cosine_predecessors'] = X.progress_apply(
                lambda row: self.cosine(
                    row['source'],
                    row['destination'],
                    for_what='pred' 
                    # src_predecessors, 
                    # dest_predecessors
                    ),axis=1)
            
        if advanced:
            print("Extracting Advanced features")
            katz = self.calculate_katz_centrality()
            hits = self.hits_score()
            #Page Rank
            
            # data['source_node_pagerank'] = data.source.map(lambda x: pr[x])
            # data['dest_node_pagerank'] = data.destination.map(lambda x: pr[x])

            #Katz Centrality
            mean_katz = float(sum(katz.values())) / len(katz)
            tqdm.pandas(desc='creating param_dependent_index_for_successors column')
            X['katz_source'] = X.source.progress_apply(lambda x: katz.get(x,mean_katz))

            tqdm.pandas(desc='creating param_dependent_index_for_successors column')
            X['katz_destination'] = X.destination.progress_apply(lambda x: katz.get(x,mean_katz))

            #Weakly Connected Components
            tqdm.pandas(desc='creating param_dependent_index_for_successors column')
            X['same_component'] = X.progress_apply(lambda row: self.belongs_to_same_wcc(row['source'],row['destination']),axis=1)
            
            #Hits Score
            tqdm.pandas(desc='creating param_dependent_index_for_successors column')
            X['hits_source'] = X.source.progress_apply(lambda x: hits[0].get(x,0))

            tqdm.pandas(desc='creating param_dependent_index_for_successors column')
            X['hits_destination'] = X.destination.progress_apply(lambda x: hits[0].get(x,0))

            tqdm.pandas(desc='creating param_dependent_index_for_successors column')
            X['authorities_source'] = X.source.progress_apply(lambda x: hits[1].get(x,0))

            tqdm.pandas(desc='creating param_dependent_index_for_successors column')
            X['authorities_destination'] = X.destination.progress_apply(lambda x: hits[1].get(x,0))

            #Weighted Features
            weight_list = self.weighted_features()
            tqdm.pandas(desc='creating param_dependent_index_for_successors column')
            X['weight_in'] = X.source.progress_apply(lambda x: weight_list[0].get(x,weight_list[2]))

            tqdm.pandas(desc='creating param_dependent_index_for_successors column')
            X['weight_out'] = X.destination.progress_apply(lambda x: weight_list[1].get(x,weight_list[3]))

            #feature engineering on with weighted features for more features
            print("\nMaking more features with 'weight_in' and 'weight_out'\n")
            X['weight_f1'] = X.weight_in + X.weight_out
            X['weight_f2'] = X.weight_in * X.weight_out
            X['weight_f3'] = (2*X.weight_in + 1*X.weight_out)
            X['weight_f4'] = (1*X.weight_in + 2*X.weight_out)

            print("Done with weighted features!\n")

            #for svd features to get feature vector creating a dict node val and inedx in svd vector
            ### With the help of SVD we can have 24 features to predict link between 2 nodes.
            print("Making more features using SVD")
            sadj_col = sorted(self.graph.nodes())
            sadj_dict = {val : idx for idx, val in enumerate(sadj_col)}
            Adj = nx.adjacency_matrix(self.graph,nodelist=sorted(self.graph.nodes())).asfptype()

            U, s, V = svds(Adj, k = 6)

            tqdm.pandas(desc='creating svd features')
            X[['svd_u_s_1', 'svd_u_s_2','svd_u_s_3', 'svd_u_s_4', 'svd_u_s_5', 'svd_u_s_6']] = \
            X.source.progress_apply(lambda x: self.svd(x, U, sadj_dict)).progress_apply(pd.Series)
            
            X[['svd_u_d_1', 'svd_u_d_2', 'svd_u_d_3', 'svd_u_d_4', 'svd_u_d_5','svd_u_d_6']] = \
            X.destination.progress_apply(lambda x: self.svd(x, U, sadj_dict)).progress_apply(pd.Series)
            #===================================================================================================
            
            X[['svd_v_s_1','svd_v_s_2', 'svd_v_s_3', 'svd_v_s_4', 'svd_v_s_5', 'svd_v_s_6',]] = \
            X.source.progress_apply(lambda x: self.svd(x, V.T, sadj_dict)).progress_apply(pd.Series)

            X[['svd_v_d_1', 'svd_v_d_2', 'svd_v_d_3', 'svd_v_d_4', 'svd_v_d_5','svd_v_d_6']] = \
            X.destination.progress_apply(lambda x: self.svd(x, V.T, sadj_dict)).progress_apply(pd.Series)

            # Preferential Attachment
            source = np.array(X["num_predecessors_source"])
            destination = np.array(X["num_predecessors_destination"])
            predecessors_preferential_source = []

            for i in tqdm(range(len(source)), 
                    desc="creating predecessors preferential attachment features"):
                predecessors_preferential_source.append(source[i]*destination[i])

            X["predecessors_preferential_attach"]  = predecessors_preferential_source 

            source = np.array(X["num_successors_source"])
            destination = np.array(X["num_successors_destination"])
            successors_preferential_destination = []

            for i in tqdm(range(len(source)),
                        desc="creating successors preferential attachment features"):
                successors_preferential_destination.append(source[i]*destination[i])

            X["successors_preferential_attach"]  = successors_preferential_destination

        

        return X.astype(int)

    def _find_successors(self, nodelist: list):
        successor_dict = defaultdict(set)

        for node in nodelist:
            try:
                successor_dict[node].add(self.graph.successors(node))

            except:
                continue

        return successor_dict

    def _find_predecessors(self, nodelist: list):
        pred_dict = defaultdict(set)

        for node in nodelist:
            try:
                pred_dict[node].add(self.graph.predecessors(node))

            except:
                continue

        return pred_dict

    def jaccard(
        self,
        x,
        y,
        for_what,
        # suc_dict_src, 
        # suc_dict_dest
        ):
        try:
            if for_what == 'successors':
                x_ = set(self.graph.successors(x))
                y_ = set(self.graph.successors(y))

            else:
                x_ = set(self.graph.predecessors(x))
                y_ = set(self.graph.predecessors(y))

            if len(x_) == 0 | len(y_) == 0:
                return 0
            sim = (len(x_.intersection(y_)))\
                    /(len(x_.union(y_)))
            # if len(suc_dict_src[x]) == 0 | len(suc_dict_dest[y]) == 0:
            #     return 0
            # sim = (len(suc_dict_src[x].intersection(suc_dict_dest[y])))\
            #         /(len(suc_dict_src[x].union(suc_dict_dest[y])))
            return sim
        except:
            return 0

    def num_successors(
        self,
        x, 
        # suc_dict
        ):
        try:
            successors = len(set(self.graph.successors(x)))
            if successors == 0:
                return 0
            return successors
        except:
            return 0


    def num_predecessors(
        self,
        x, 
        # pred_dict
        ):
        try:
            predecessors = len(set(self.graph.predecessors(x)))
            # predecessors = len(pred_dict[x])
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

    def successors_predecessors_ratio(
        self,
        x, 
        # suc_dict, 
        # pred_dict
        ):
        try:
            num_successors = len(set(self.graph.successors(x)))
            num_predecessors = len(set(self.graph.predecessors(x)))
            if num_successors == 0 | num_predecessors == 0:
                return 0
            r = num_successors/num_predecessors
            return r
        except:
            return 0
    
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

    def cosine(
        self,
        x,
        y,
        for_what, 
        # pred_dict_src: dict, 
        # pred_dict_dest: dict
        ):
        try:            
            if for_what == 'successors':
                x_ = set(self.graph.successors(x))
                y_ = set(self.graph.successors(y))

            else:
                x_ = set(self.graph.predecessors(x))
                y_ = set(self.graph.predecessors(y))

            if len(x_) == 0 | len(x_) == 0:
                return 0
            sim = (len(x_.intersection(y_)))\
                    /(math.sqrt(len(x_)*len(y_)))

            # if len(pred_dict_src[x]) == 0 | len(pred_dict_dest[y]) == 0:
            #     return 0
            # sim = (len(pred_dict_src[x]).intersection(pred_dict_dest[y]))\
            #         /(math.sqrt(len(pred_dict_src[x])*len(pred_dict_dest[y])))
            return sim
        except:
            return 0

    def dice_index(self,x,y, for_what):
        try:
            if for_what == 'successors':
                x_ = set(self.graph.successors(x))
                y_ = set(self.graph.successors(y))

            else:
                x_ = set(self.graph.predecessors(x))
                y_ = set(self.graph.predecessors(y))

            if len(x_) == 0 | len(y_) == 0:
                return 0
            sim = (len(x_.intersection(y_)))\
                    /(len(x_) + len(y_))
            return sim
        except:
            return 0

    def hub_promoted_index(self,x,y, for_what):
        try:
            if for_what == 'successors':
                x_ = set(self.graph.successors(x))
                y_ = set(self.graph.successors(y))

            else:
                x_ = set(self.graph.predecessors(x))
                y_ = set(self.graph.predecessors(y))

            if len(x_) == 0 | len(y_) == 0:
                return 0
            sim = (len(x_.intersection(y_)))\
                    /min(len(x_), len(y_))
            return sim
        except:
            return 0
        
    def hub_depressed_index(self,x,y, for_what):
        try:
            if for_what == 'successors':
                x_ = set(self.graph.successors(x))
                y_ = set(self.graph.successors(y))

            else:
                x_ = set(self.graph.predecessors(x))
                y_ = set(self.graph.predecessors(y))

            if len(x_) == 0 | len(y_) == 0:
                return 0
            sim = len(x_.intersection(y_))\
                    /max(len(x_), len(y_))
            return sim
        except:
            return 0

    def leicht_holme_index(self,x,y, for_what):
        try:
            if for_what == 'successors':
                x_ = set(self.graph.successors(x))
                y_ = set(self.graph.successors(y))

            else:
                x_ = set(self.graph.predecessors(x))
                y_ = set(self.graph.predecessors(y))

            if len(x_) == 0 | len(y_) == 0:
                return 0
            sim = len(x_.intersection(y_))\
                    /(len(x_)*len(y_))
            return sim
        except:
            return 0
    def param_dependent_index(self,x,y, for_what, c = 0.5):
        """ 
        math: 
                PD(x,y) =|Γ(x) ∩ Γ(y)| / |Γ(x)|.|Γ(y)|^λ """
        try:
            if for_what == 'successors':
                x_ = set(self.graph.successors(x))
                y_ = set(self.graph.successors(y))

            else:
                x_ = set(self.graph.predecessors(x))
                y_ = set(self.graph.predecessors(y))

            if len(x_) == 0 | len(y_) == 0:
                return 0
            sim = (len(x_.intersection(y_)))\
                    /(((len(x_)*len(y_)))**c)
            return sim
        except:
            return 0

    # def param_dependent_index_for_predecessors(self,x,y, c = 0.5):
    #     """ 
    #     math: 
    #             PD(x,y) =|Γ(x) ∩ Γ(y)| / |Γ(x)|.|Γ(y)|^λ """
    #     try:
    #         x_ = set(self.graph.predecessors(x))
    #         y_ = set(self.graph.predecessors(y))
    #         if len(x_) == 0 | len(y_) == 0:
    #             return 0
    #         sim = len(x_.intersection(y_))\
    #                 /(len(x_)*len(y_))
    #         return sim
    #     except:
    #         return 0

    def individual_attraction_index(self,x,y):
        """
        math:
                summation(i belonging to {X inter Y}) ||/log(|pred(i)|)"""
        self.x = x
        self.y = y
        # self.self.graph = self.graph
        summ = 0

        pass
        # try:
        #     n = list(set(self.graph.successors(x)).intersection(set(self.graph.successors(y))))
            
        #     if n != 0:
        #         for i in n:
        #             summ = sum( / np.log(len(set(self.graph.predcessors(i)))))
        #         return summ
        #     else:
        #         return 0
        # except:
        #     return 0

    ## Advanced Features

    def adar_index(self,x,y):
        """
        math:
                summation(i belonging to {X inter Y}) 1/log(|pred(i)|)"""
        self.x = x
        self.y = y
        # self.self.graph = self.graph
        summ = 0
        try:
            n = list(set(self.graph.successors(x)).intersection(set(self.graph.successors(y))))
            
            if n != 0:
                for i in n:
                    summ = sum(1/np.log(len(set(self.graph.predcessors(i)))))
                return summ
            else:
                return 0
        except:
            return 0

    # def page_rank(self,self.graph):
        # self.self.graph = self.graph
    #     temp_path = os.getcwd()+"/self.graph/temp"
    #     if not os.path.isdir(temp_path+'/data/fea_sample'):
    #         os.makedirs(temp_path+'/data/fea_sample')
    #     pr = nx.pagerank(self.graph, alpha= 0.85)
    #     # with open(temp_path+'/data/fea_sample/page_rank.pkl', 'wb') as f:
    #     #     pickle.dump(pr,f)
        
    #     # pr = pickle.load(open(temp_path+'/data/fea_sample/page_rank.pkl', 'rb'))

    #     return pr

    #getting weekly connected edges from self.graph  
    ## Give infomation about Community
    def belongs_to_same_wcc(self, x, y):
        self.x = x
        self.y = y
        # self.self.graph = self.graph
        wcc=list(nx.weakly_connected_components(self.graph))

        index = []
        if self.graph.has_edge(x,y):
            return 1
        if self.graph.has_edge(x,y):
                for i in wcc:
                    if x in i:
                        index= i
                        break
                if (y in index):
                    self.graph.remove_edge(x,y)
                    if self.shortest_path(x,y) == -1:
                        self.graph.add_edge(x,y)
                        return 0
                    else:
                        self.graph.add_edge(x,y)
                        return 1
                else:
                    return 0
        else:
                for i in wcc:
                    if x in i:
                        index= i
                        break
                if(y in index):
                    return 1
                else:
                    return 0
    
    ## Katz Centrality of a Node
    def calculate_katz_centrality(self):
        # self.self.graph = self.graph
        temp_path = os.getcwd()+"/self.graph/temp"

        if not os.path.isdir(temp_path+'/data/fea_sample'):
            os.makedirs(temp_path+'/data/fea_sample')
        katz = nx.katz.katz_centrality(self.graph , alpha=0.005, beta=1)
        # with open(temp_path+'/data/fea_sample/katz.pkl', 'wb') as f:
        #     pickle.dump(katz,f)
        
        # katz = pickle.load(open(temp_path+'/data/fea_sample/katz.pkl', 'rb'))
        return katz

    def hits_score(self):
        # self.self.graph = self.graph
        temp_path = os.getcwd()+"/self.graph/temp"

        if not os.path.isdir(temp_path+'/data/fea_sample'):
            os.makedirs(temp_path+'/data/fea_sample')
        hits = nx.hits(self.graph, max_iter=100, tol=1e-08, nstart=None, normalized=True)
        # with open(temp_path+'/data/fea_sample/hits.pkl', 'wb') as f:
        #     pickle.dump(hits,f)
        
        # hits = pickle.load(open(temp_path+'/data/fea_sample/hits.pkl', 'rb'))
        return hits

    #weight for source and destination of each link
    def weighted_features(self):
        # self.self.graph = self.graph
        weight_in = {}
        weight_out = {}
        for i in  tqdm(self.graph.nodes()):
            s1 = len(set(self.graph.predecessors(i)))
            w_in = 1.0/(np.sqrt(1 + s1))
            weight_in[i] = w_in
            
            s2 = len(set(self.graph.successors(i)))
            w_out = 1.0/(np.sqrt(1 + s2))
            weight_out[i] = w_out
            
        #for imputing with mean
        mean_weight_in = np.mean(list(weight_in.values()))
        mean_weight_out = np.mean(list(weight_out.values()))

        return [weight_in, weight_out, mean_weight_in, mean_weight_out]        


    # SVD (Singular Value Decomposition) a Matrix Factorization method to extract features from self.Graph
    def svd(self,x, S,sadj_dict):
        self.x = x
        self.S = S
        self.sadj_dict = sadj_dict
        try:
            z = sadj_dict[x]
            return S[z]
        except:
            return [0,0,0,0,0,0]
