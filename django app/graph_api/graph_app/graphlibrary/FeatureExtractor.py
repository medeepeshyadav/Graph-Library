from collections import defaultdict
import math
import os

from multiprocessing import Pool
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse.linalg import svds

import networkx as nx

class InputShapeError(ValueError):
    def __init__(self, size):
        super.__init__(f"Expected 2 columns ('source_node', 'destination_node') got {size}")

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


    def _parallelize_apply(self, data, func, n_jobs):
        df_split = np.array_split(data, n_jobs)
        pool = Pool(n_jobs)
        df = pd.concat(pool.map(func, df_split))
        pool.close()
        pool.join()
        return df

    def _get_features(self, X):
        try:
            if X.shape[1] != 2:
                raise InputShapeError
        except InputShapeError as i:
            print(f"Input data with 2 columns expected got {X.shape[1]} columns.")

        X.columns = ['source', 'destination']
        X = X.astype(str)
        # ddata = dd.from_pandas(X, npartitions=30)

        features1 = [
                    'numSuccessors_source', 
                    'numSuccessors_destination',
                    'numPredecessors_source',
                    'numPredecessors_destination',
                    ]
            
        for feature in features1:
            if self.graph_type == 'undirected' and 'Predecessor' in feature:
                continue
            
            func, flag = feature.split('_')
            func = getattr(self, func)

            # if flag == 'src':
            tqdm.pandas(desc=f"creating {feature} column")
            X[feature] = X.progress_apply(lambda row: func(row[flag]), axis=1)
        

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

            features1 = [
                    'successorPredecessorRatio_source',
                    'successorPredecessorRatio_destination'
                ]

            for feature in features1:
                if self.graph_type == 'undirected' and 'Predecessor' in feature:
                    continue
                
                func, flag = feature.split('_')
                func = getattr(self, func)

                tqdm.pandas(desc=f"creating {feature} column")
                X[feature] = X.progress_apply(lambda row: func(row[flag]), axis=1)


            features2 = ['back_link','shortest_path_bw_x_and_y', 'adar_index']

            for feature in features2:
                func = getattr(self, feature)
                tqdm.pandas(desc=f"creating {feature} column")
                X[feature] = X.progress_apply(lambda row: func(row['source'], row['destination']), axis=1)

            features3 = [
                'jaccard_successors', 
                'jaccard_predecessors',
                'diceIndex_successors',
                'diceIndex_predecessors',
                'hubPromotedIndex_successors',
                'hubPromotedIndex_predecessors',
                'hubDepressedIndex_successors',    # can be removed
                'hubDepressedIndex_predecessors',  # can be removed
                'leichtHolmeIndex_successors',
                'leichtHolmeIndex_predecessors',
                'paramDependentIndex_successors',  # can be removed
                'paramDependentIndex_predecessors', # can be removed
                'cosine_successors',
                'cosine_predecessors',
            ]
            for feature in features3:
                if self.graph_type == 'undirected' and 'predecessor' in feature:
                    continue
                
                func, flag = feature.split('_')
                func = getattr(self, func)
        
                tqdm.pandas(desc=f"creating {feature} column")
                X[feature] = X.progress_apply(lambda row: func(
                                row['source'],row['destination'], for_what=flag), axis=1)

            
        if advanced:
            print("Extracting Advanced features")
            katz = self.calculate_katz_centrality()
            hits = self.hits_score()
            pr = self._get_page_rank()
            weight_list = self.weighted_features()
            #Page Rank
            # X['source_pagerank'] = X['source'].map(lambda x: pr[x])
            # X['destination_pagerank'] = X['destination'].map(lambda x: pr[x])

            #Katz Centrality
            mean_katz = float(sum(katz.values())) / len(katz)

            features = [
                'katz_source', 
                'katz_destination',
                'hits_source',
                'hits_destination',
                'hitsAuthorities_source',
                'hitsAuthorities_destination',
                'weight_source',
                'weight_destination',
                ]

            for feat in features:
                # try:
                # print(feat)
                feature_type, node = feat.split("_")
                # except ValueError:
                #     print(f"This feature is causing trouble -->{feat}")

                tqdm.pandas(desc=f"creating {feat} column")
                if feature_type == 'katz':
                    X[feat] = X[node].progress_apply(lambda x: katz.get(x,mean_katz))

                if 'hits' in feature_type:
                    i = 1 if 'Authorities' in feature_type else 0
                    X[feat] = X[node].progress_apply(lambda x: hits[i].get(x,0))

                if feature_type == 'weight':
                    i = 0 if node == 'source' else 1
                    j = 2 if node == 'source' else 3

                    X[feat] = X[node].progress_apply(lambda x: weight_list[i].get(x,weight_list[j]))




            # #Weakly Connected Components
            tqdm.pandas(desc='creating same_component column')
            X['same_component'] = X.progress_apply(
                lambda row: self.belongs_to_same_wcc(row['source'],row['destination']),axis=1)
            

            #feature engineering on with weighted features for more features
            print("\nMaking more features with 'weight_source' and 'weight_destination'\n")
            X['weight_f1'] = X.weight_source + X.weight_destination
            X['weight_f2'] = X.weight_source * X.weight_destination
            X['weight_f3'] = (2*X.weight_source + 1*X.weight_destination)
            X['weight_f4'] = (1*X.weight_source + 2*X.weight_destination)

            print("Done with weighted features!\n")

            #for svd features to get feature vector creating a dict node val and index in svd vector
            ### With the help of SVD we can have 24 features to predict link between 2 nodes.
            print("Making more features using SVD")
            sadj_col = sorted(self.graph.nodes())
            sadj_dict = {val : idx for idx, val in enumerate(sadj_col)}

            # creating adjacency matrix
            Adj = nx.adjacency_matrix(self.graph, nodelist=sadj_col).asfptype()

            U, s, V = svds(Adj, k = 6)

            for feat in ['svdu_source', 'svdu_destination', 'svdv_source', 'svdv_destination']:
                feature_type, node = feat.split("_")
                tqdm.pandas(desc='creating svd features')
                component = U if feature_type[-1] == 'u' else V.T

                X[[feat+'_1', feat+'_2',feat+'_3', feat+'_4', feat+'_5', feat+'_6']] = \
                    X[node].progress_apply(lambda x: self.svd(x, component, sadj_dict)).apply(pd.Series)


            # Preferential Attachment
            for feat in ['succ_PrefAttacnement', 'pred_PrefAttacnement']:
                flag = feat.split('_')[0]
                src = np.array(X["numPredecessors_source"]) if flag == 'pred' else np.array(X["numSuccessors_source"])
                dest = np.array(X["numPredecessors_destination"]) if flag == 'pred' else np.array(X['numSuccessors_destination'])
                pref_attachment = []

                for i in range(len(src)):
                    pref_attachment.append(src[i]*dest[i])

                X[feat]  = pref_attachment


        return X

    def transform(self, X):
        return self._parallelize_apply(data=X, func=self._get_features, n_jobs=self.n_jobs)

    def jaccard(
        self,
        x,
        y,
        for_what,
        ):
        try:
            if for_what == 'successors':
                x_ = set(self.graph.successors(x))
                y_ = set(self.graph.successors(y))

            else:
                x_ = set(self.graph.predecessors(x))
                y_ = set(self.graph.predecessors(y))

            if len(x_) == 0 or len(y_) == 0:
                return 0
            sim = (len(x_.intersection(y_)))/(len(x_.union(y_)))
            return sim
        except:
            return 0

    def numSuccessors(
        self,
        x, 
        ):
        try:
            successors = len(set(self.graph.successors(x)))
            if successors == 0:
                return 0
            return successors
        except:
            return 0


    def numPredecessors(
        self,
        x, 
        ):
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

    def successorPredecessorRatio(
        self,
        x, 
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
    
    def shortest_path_bw_x_and_y(self,x,y):
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
        ):
        try:            
            if for_what == 'successors':
                x_ = set(self.graph.successors(x))
                y_ = set(self.graph.successors(y))

            else:
                x_ = set(self.graph.predecessors(x))
                y_ = set(self.graph.predecessors(y))

            if len(x_) == 0 or len(x_) == 0:
                return 0
            sim = (len(x_.intersection(y_)))/(math.sqrt(len(x_)*len(y_)))
            return sim
        except:
            return 0

    def diceIndex(self,x,y, for_what):
        try:
            if for_what == 'successors':
                x_ = set(self.graph.successors(x))
                y_ = set(self.graph.successors(y))

            else:
                x_ = set(self.graph.predecessors(x))
                y_ = set(self.graph.predecessors(y))

            if len(x_) == 0 or len(y_) == 0:
                return 0
            sim = (len(x_.intersection(y_)))/(len(x_) + len(y_))
            return sim
        except:
            return 0

    def hubPromotedIndex(self,x,y, for_what):
        try:
            if for_what == 'successors':
                x_ = set(self.graph.successors(x))
                y_ = set(self.graph.successors(y))

            else:
                x_ = set(self.graph.predecessors(x))
                y_ = set(self.graph.predecessors(y))

            if len(x_) == 0 or len(y_) == 0:
                return 0
            sim = (len(x_.intersection(y_)))/min(len(x_), len(y_))
            return sim
        except:
            return 0
        
    def hubDepressedIndex(self,x,y, for_what):
        try:
            if for_what == 'successors':
                x_ = set(self.graph.successors(x))
                y_ = set(self.graph.successors(y))

            else:
                x_ = set(self.graph.predecessors(x))
                y_ = set(self.graph.predecessors(y))

            if len(x_) == 0 or len(y_) == 0:
                return 0
            sim = len(x_.intersection(y_))/max(len(x_), len(y_))
            return sim
        except:
            return 0

    def leichtHolmeIndex(self,x,y, for_what):
        try:
            if for_what == 'successors':
                x_ = set(self.graph.successors(x))
                y_ = set(self.graph.successors(y))

            else:
                x_ = set(self.graph.predecessors(x))
                y_ = set(self.graph.predecessors(y))

            if len(x_) == 0 or len(y_) == 0:
                return 0
            sim = len(x_.intersection(y_))/(len(x_)*len(y_))
            return sim
        except:
            return 0
    def paramDependentIndex(self,x,y, for_what, c = 0.7):
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

            if len(x_) == 0 or len(y_) == 0:
                return 0
            sim = (len(x_.intersection(y_)))/((len(x_)*len(y_))**c)
            return sim
        except:
            return 0

    ## Advanced Features

    def adar_index(self,x,y):
        """
        math:
                summation(i belonging to {X inter Y}) 1/log(|pred(i)|)
        """
        summ = 0
        try:
            # x_ = self.successors_dict[x]
            # y_ = self.successors_dict[y]
            x_ = set(self.graph.successors(x))
            y_ = set(self.graph.successors(y))
            n = list(x_.intersection(y_))

            if len(n) != 0:
                for i in n:
                    num_pred = len(set(self.graph.predecessors(i)))
                    try:
                        summ += 1/np.log(num_pred)
                    except:
                        summ += 0
                return summ
            else:
                return 0
        except:
            return 0

    def _get_page_rank(self):
        pr = nx.pagerank(self.graph, alpha= 0.85)

        return pr

    #getting weakly connected edges from self.graph  
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
        katz = nx.katz.katz_centrality(self.graph , alpha=0.005, beta=1)
        
        return katz

    def hits_score(self):
        hits = nx.hits(self.graph, max_iter=100, tol=1e-08, nstart=None, normalized=True)
    
        return hits

    #weight for source and destination of each link
    def weighted_features(self):
        weight_source = {}
        weight_destination = {}
        for i in  tqdm(self.graph.nodes()):
            s1 = len(set(self.graph.predecessors(i)))
            w_in = 1.0/(np.sqrt(1 + s1))
            weight_source[i] = w_in
            
            s2 = len(set(self.graph.successors(i)))
            w_out = 1.0/(np.sqrt(1 + s2))
            weight_destination[i] = w_out
            
        #for imputing with mean
        mean_weight_source = np.mean(list(weight_source.values()))
        mean_weight_out = np.mean(list(weight_destination.values()))

        return [weight_source, weight_destination, mean_weight_source, mean_weight_out]        


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
