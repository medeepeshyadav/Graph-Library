from graph_app.models import Order
from graph_app.graphlibrary import PrepareData, FeatureExtractor
import os
import shutil
import pandas as pd

class Processor:
    def __init__(self) -> None:
        self.list_of_jobs = []

    def get_entries(self):
        # get first 10 order
        if len(Order.objects.all()) > 10:
            last_ten = Order.objects.filter(status='ready')[:10]
        last_ten = Order.objects.filter(status='ready')

        entries = []
        for entry in last_ten:
            entries.append(entry)

        return entries

    def my_cron_job(self):
        """ 
        run in every 5 minutes
        look for oldest 'Ready' file
        put it on processing

        """
        print("I was executed!! and Still talking.")
        if not self.list_of_jobs:
            self.list_of_jobs = self.get_entries()
            # [1,2,3,4,5]
            # [4,3,2,1]
            # e = 5
            # 
            for e in reversed(self.list_of_jobs):
                # set status to "in progress"
                o = Order.objects.get(file_id=e.file_id)
                o.status = 'in progress'
                o.save()

                # get filepath
                filepath = "graph_app/uploaded/"+str(e.file_id)+".csv"

                # prepare data
                p = PrepareData(path=filepath, graph_type=e.graph_type, test_size=float(e.test_size), n_jobs=e.n_jobs)
                XTr, XTs, ytr, yts = p.prepare()

                # transform data
                fex = FeatureExtractor(graph_type=e.graph_type, type=e.feature_type, n_jobs=e.n_jobs)
                fex.fit(XTr, ytr)
                
                X_train = fex.transform(XTr)
                Train = pd.concat((X_train, ytr), axis=1)

                X_test = fex.transform(XTs)
                Test = pd.concat((X_test, yts), axis=1)

                # output dir
                out_dir = "graph_app/output/"+str(e.file_id)
                if not os.path.isdir(out_dir):
                    os.makedirs(out_dir)
                
                # putting train and test csv to the user's folder
                Train.to_csv(out_dir+"/train.csv", index=False)
                Test.to_csv(out_dir+"/test.csv", index=False)

                # zipping user's folder
                shutil.make_archive(out_dir, 'zip', root_dir=out_dir)
                
                # deleting folder
                shutil.rmtree(out_dir)

                o = Order.objects.get(file_id=e.file_id)
                o.status = 'done'
                o.out_file_loc = out_dir+'.zip'
                o.save()
                self.list_of_jobs.pop()

        else:
            print("Nothing to do!!")
            