import csv
import numpy as np

def params_to_file(f_name, params):
    lst = []
    lst.append(['param','index','val'])
    for p in params:
        try:
            for i,v in enumerate(params[p]):
                lst.append([p, i, v])
        except: # not a list
            lst.append([p, 0, params[p]])

    with open(f_name,'wb') as f:
        writer = csv.writer(f, delimiter=',')
        for row in lst:
            writer.writerow(row)

def params_from_file(f_name):
    with open(f_name,'r') as f:
        reader = csv.reader(f, delimiter=',')
        reader.next()
        
        d = dict()
        for row in reader:
            p, i, v = row
            d[p,int(i)] = v

        res = dict()
        params = set(p for p,i in d)
        for p in params:
            indices = sorted(i for p_,i in d if p_==p)
            res[p] = np.array([float(d[p,i]) for i in indices]) 

        return res



class LoadGold():
    def __init__(self, gold_f, votes_f, params_f=None):
        with open(votes_f,'r') as f:
            d = dict()
            reader = csv.DictReader(f)
            for r in reader:
                w = r['nel.worker']
                v = int(r['nel.response'])
                t = int(r['nel.duration'])
                item = int(r['nel.itemid'])
                if w not in d:
                    d[w] = dict()
                d[w][item] = {'v': v, 't': t}

            max_len = max(len(d[w]) for w in d)
            w_completed = sorted([w for w in d if len(d[w]) == max_len])
            print w_completed
            q_list = sorted(d[w_completed[0]].keys())

            self.votes = np.array([[d[w][q]['v'] for q in q_list] for
                                   w in w_completed])

            self.times = np.array([[d[w][q]['t'] for q in q_list] for
                                   w in w_completed])


        with open(gold_f,'r') as f_gold:

            gold = [int(x.strip()) for x in f_gold]
            self.gold = np.array([gold[i] for i in q_list]) # filter


    
        if params_f is not None:
            self.params = params_from_file(params_f)
        else:
            self.params = None

    def get_votes(self):
        return self.votes

    def get_times(self):
        return self.times

    def get_gt(self):
        return self.gold

    def get_difficulties(self):
        if self.params is not None:
            return self.params['difficulties']
        else:
            return None
                
    def get_skills(self):
        if self.params is not None:
            return self.params['skills']
        else:
            return None
 
