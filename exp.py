from __future__ import division
from collections import defaultdict
import numpy as np
from scipy import stats
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import random
import pickle
import heapq
import sys
import csv
import json
import os
import copy
import itertools

from control import Controller
from simulator import Platform
import parse

NUM_SAVE = 5

class NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray) and obj.ndim == 1:
                return obj.tolist()
        
        return json.JSONEncoder.default(self, obj)



def gen_thetas(num_questions):
    """Randomly generate thetas"""
    return np.random.random(num_questions)


def est_final_params(gt_observations):

    # use dummy controller to run EM
    num_w, num_q = gt_observations.shape
    c = Controller(method=None, platform=None,
                   num_workers=num_w, num_questions=num_q)

    params,_ = c.run_em(gt_observations)

    
        
    d = params['difficulties']
    s = params['skills']
        
    parse.params_to_file('gold_params.csv',params)
    return d,s
        


def agg_scores(accs, x='observed', y='accuracy'):
    """Takes accuracies and computes averages and standard errors"""
    if len(accs) == 1:
        return [(d[x], d[y], None) for d in accs[0]]

    points = []

    def next_point(scores):
        vals = [scores[i][y] for i in scores]
        max_x = max(scores[i][x] for i in scores)
        return (max_x, np.mean(vals), stats.sem(vals, ddof=0))

    # NOTE: can improve efficiency
    # assume first score is at t=0 and votes=0
    accs = copy.deepcopy(accs)
    scores = dict((i,accs[i].pop(0)) for i in xrange(len(accs)))
    points.append(next_point(scores))

    while sum(len(accs[i]) for i in xrange(len(accs))) > 0:

        next_x = []
        for i in xrange(len(accs)):
            if len(accs[i]) > 0:
                next_x.append(accs[i][0][x])
            else:
                next_x.append(None)

        min_x = min(x for x in next_x if x is not None)
        for i,v in enumerate(next_x):
            if v == min_x:
                scores[i] = accs[i].pop(0)

        points.append(next_point(scores))

    return points

# def save_iteration(res_path, p, i, res):
#     with open(os.path.join(res_path,
#                            'details - {} - {}.json'.format(i,p)), 'w') as f:
#         json.dump(res, f, indent=1, cls=NumpyAwareJSONEncoder)
#


def save_results(res_path, exp_name, res, iter_n):
    hist = dict((p, [x['hist'] for x in res[p]]) for p in res)
    
    # NOTE: turned off to conserve space
#    for p in res:
#        with open(os.path.join(res_path, 'res - {}.json'.format(p)), 'w') as f:
#            json.dump(res[p], f, indent=1)

    with open(os.path.join(res_path, 'when_finished.json'), 'w') as f:
        json.dump(dict((p, [d['when_finished'] for d in res[p]]) for p in res),
                  f, indent=1)

    # save timings and detailed accuracies (for rebuttal)
    with open(os.path.join(res_path, 'hist.csv'), 'wb') as f:
        to_csv = []
        for p in hist:
            for i,h in enumerate(hist[p]):
                for d in h:
                    to_csv.append({'policy': p,
                                   'run': i,
                                   'observed': d['observed'],
                                   'time': d['time'],
                                   'accuracy': d['accuracy'],
                                   'exp_accuracy': d['exp_accuracy'],
                                   'duplicates': d['duplicates'],
                                   'timing': d['timing']})


        writer = csv.DictWriter(f, ['policy',
                                    'run',
                                    'observed',
                                    'time',
                                    'accuracy',
                                    'exp_accuracy',
                                    'duplicates',
                                    'timing'])
        writer.writeheader()
        writer.writerows(to_csv)


    # markers = itertools.cycle('>^+*')   
    markers = itertools.repeat(None)   
    
    for t in ('accuracy','exp_accuracy'):
        policies = hist.keys()
        scores = defaultdict(dict)
        for p in policies:
            #assert a[p].shape[1] == NUM_QUESTIONS + 1
            scores['votes'][p] = agg_scores(hist[p],'observed',t)
            scores['time'][p] = agg_scores(hist[p],'time',t)

        # create figures and save data
        for x_type in scores:
            fname = 'plot_{}'.format(x_type)
            if t == 'exp_accuracy':
                fname = 'exp_' + fname

            plt.close('all')
                
            with open(os.path.join(res_path, fname+'.csv'),'wb') as f:
                writer = csv.writer(f) 
                writer.writerow(['policy','x','y','stderr'])

                for p in sorted(policies):
                    x_val, mean, stderr = zip(*scores[x_type][p])
                    if any(x is None for x in stderr):
                        stderr = None
                    else:
                        stderr = [x * 1.96 for x in stderr]

                    # plot with error bars
                    next_marker = markers.next()

                    plt.figure(0)
                    plt.errorbar(x_val, mean, yerr=stderr,
                                 marker=next_marker, label=p)

                    # plot without 
                    plt.figure(1)
                    plt.plot(x_val, mean, marker=next_marker, label=p)

                    # to file
                    if stderr is None:
                        stderr = itertools.repeat(None)

                    for x,y,s in zip(x_val, mean, stderr):
                        writer.writerow([p,x,y,s])


            for i in xrange(2):
                plt.figure(i)
                plt.ylim(ymin=0.5,ymax=1)
                plt.legend(loc="lower right")
                if x_type == 'votes':
                    xlabel = 'Number of votes observed'
                elif x_type == 'time':
                    xlabel = 'Time elapsed'
                else:
                    raise Exception
                plt.xlabel(xlabel)
                plt.ylabel('Prediction accuracy')

            plt.figure(0)
            with open(os.path.join(res_path, fname+'_err.png'),'wb') as f:
                plt.savefig(f, format="png", dpi=150)

            plt.figure(1)
            with open(os.path.join(res_path, fname+'.png'),'wb') as f:
                plt.savefig(f, format="png", dpi=150)



def mkdir_or_ignore(d):
    try:
        os.mkdir(d)
    except OSError:
        pass # dir already exists


def parse_skill(d):
    """Handle legacy policies"""
    if 'mean' in d:
        return {'type': 'normal_pos',
                'params': [float(d['mean']), float(d['std'])]}
    else:
        return d



class Result():
    """ Class for managing reuslts
    """
    def __init__(self, expdir, expname, maxcount=1):
        self.expdir = expdir
        self.expname = expname
        self.runid = 0
        self.runcounts = defaultdict(int)
        self.maxcount = maxcount
        
        mkdir_or_ignore(os.path.join(expdir,'tables'))
        
        self.tables = {
                  'history': ['run_id',
                              't_',
                              'action_',
                              'observation',
                              'reward',
                              'timing',
                              'state_',
                              'actions',
                              'other'],
                  'runs': ['run_id',
                           'policy',
                           'policy_name',
                           'experiment',
                           'experiment_name',
                           'seed',
                           'info']}
    
    
        for t in self.tables:
            with open(self.tablepath(t),'w') as f:
                writer = csv.DictWriter(f, self.tables[t])
                writer.writeheader()
            
    def tablepath(self, t):
        return os.path.join(self.expdir,
                            'tables',
                            self.expname+'-'+t+'.csv')
    
    
    def update(self, history_rows, run_row):
        
        policy_name = run_row['policy_name']
        d = {'history': history_rows,
             'runs': [run_row]}
    
        for tab in self.tables:
            with open(self.tablepath(tab),'a') as f:
                writer = csv.DictWriter(f, self.tables[tab])
                for row in d[tab]:
                    row['run_id'] = self.runid
                    
                    # convert to json
                    json_encoder = NumpyAwareJSONEncoder()
                    if tab=='history':
                        for k in ['observation', 'state_', 'other']:
                            row[k] = json.dumps(row[k], cls=NumpyAwareJSONEncoder, sort_keys=True)
                    if tab=='runs':
                        for k in ['experiment', 'policy', 'info']:
                            row[k] = json.dumps(row[k], cls=NumpyAwareJSONEncoder, sort_keys=True)

                    # conserve space
                    if tab=='history' and self.runcounts[policy_name] >= self.maxcount:
                        row['state_'] = None
                        row['observation'] = None
                        
                    writer.writerow(row)

        self.runcounts[policy_name] += 1
        self.runid += 1
        

#------------------- MAIN --------------------
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print 'usage: {0} policy_file'.format(sys.argv[0])
        sys.exit()

    # load
    with open(sys.argv[1]) as f:
        s = json.load(f)

    def val_or_none(d, k, f=lambda x: x):
        if k in d:
            return f(d[k])
        else:
            return None

    exp_name = os.path.splitext(os.path.split(sys.argv[1])[-1])[-2]
#    exp_name = s['exp_name'] # BUG: change to input filename?
    n_workers = int(s['n_workers'])
    n_questions = int(s['n_questions'])
    sample_p = bool(s['sample'])
    n_exps = int(s['n_exps'])

    gen_method_in = val_or_none(s, 'gen_method')
    skill_in = val_or_none(s, 'skill', lambda x: parse_skill(x))
    diff_in = val_or_none(s, 'diff')
    time_in = val_or_none(s, 'time')
    votes_in = val_or_none(s, 'votes')
    labels_in = val_or_none(s, 'labels')
    subsample_in = val_or_none(s, 'subsample')
    if subsample_in:
        subsample_in = int(subsample_in)

    # prepare experiment version to store in table
    experiment_json = {
                          'n_workers': n_workers,
                          'n_questions': n_questions,
                          # 'sample': int(sample_p),?
                  
                          'votes': votes_in,
                          'labels': labels_in,
                          'diff': diff_in,
                          'skill': skill_in,
                          'time': time_in,

                          'subsample': subsample_in,
                          'gen_method': gen_method_in
                      }
    experiment_name_json = val_or_none(s, 'name')
                  
    policies = s['policies']

    # load experimental data
    gold = parse.LoadGold(os.path.join('data','gold.txt'),
                          os.path.join('data','db_nel.csv'),
                          os.path.join('data','gold_params.csv'))

    if skill_in == 'gold':
        skill_in = gold.get_skills()
    if diff_in == 'gold':
        diff_in = gold.get_difficulties()
    if time_in == 'gold':
        time_in = gold.get_times()
    elif time_in == 'gold_params':
        # hard-code gold params for 12 workers
        assert n_workers == 12
        with open('time_params.csv', 'r') as f:
            reader = csv.DictReader(f)
            time_in = [{'type': 'lognorm',
                        'params': [d['shape'],d['loc'],d['scale']]} for
                       d in reader]




    if votes_in == 'gold':
        votes_in = gold.get_votes()
    if labels_in == 'gold':
        labels_in = gold.get_gt()

    # prepare result directory
    mkdir_or_ignore('res')
    res_path = os.path.join('res', exp_name)
    mkdir_or_ignore(res_path)
    # mkdir_or_ignore(os.path.join(res_path, 'detailed'))
   
    # copy policy file
    import shutil
    shutil.copy(sys.argv[1], res_path)

    res = Result(res_path, exp_name, maxcount=0)

    # run experiments
    policies_run = set()

    def first_array_or_true(lst):
        for e in lst:
            if isinstance(e, np.ndarray) or e:
                return e
        
        return lst[-1]


    for i in xrange(n_exps):
        # rint = random.randint(0,sys.maxint)
        rint = i
        print '------------'
        print 'iteration: ' + str(i)
        np.random.seed(rint)

        # prepare
        if isinstance(votes_in, np.ndarray) or votes_in:
            platform = Platform(
                    votes = votes_in,
                    gt_labels=first_array_or_true([labels_in, gold.get_gt()]),
                    difficulties=first_array_or_true([diff_in,
                                                      gold.get_difficulties()]),
                    times=first_array_or_true([time_in, gold.get_times()]),
                    skills=first_array_or_true([skill_in, gold.get_skills()]),
                    subsample=subsample_in)
            run_once = platform.is_deterministic
#            print run_once
        else:
            run_once = False
            gt_thetas = gen_thetas(n_questions)
            gt_labels = np.round(gt_thetas)
            
            if gen_method_in != 'kg':
                gt_thetas = None
                
            platform = Platform(
                    num_workers=n_workers,
                    gt_labels=first_array_or_true([labels_in, gt_labels]),
                    difficulties=first_array_or_true([diff_in,
                                                      gold.get_difficulties()]),
                    times=first_array_or_true([time_in, gold.get_times()]),
                    skills=first_array_or_true([skill_in, gold.get_skills()]),
                    thetas=gt_thetas)


        # run
        for p in policies:
            if run_once and p['type'] in ['greedy','greedy_reverse','accgain','accgain_reverse'] and p['name'] in policies_run:
                continue

            for s in (p['known_d'],p['known_s']):
                assert s == 'True' or s == 'False'

            if 'eval' in p:
                post_inf = p['eval']
            else:
                post_inf = 'reg'
                
            if 'maxdup' in p:
                maxdup = p['maxdup']
            else:
                maxdup = float('inf')

            np.random.seed(rint)
            platform.reset()
            controller = Controller(method=p['type'],
                                    platform=platform,
                                    num_workers=platform.num_workers,
                                    num_questions=platform.num_questions,
                                    known_d=eval(p['known_d']),
                                    known_s=eval(p['known_s']),
                                    maxdup=maxdup,
                                    post_inf=post_inf)

                                    
            if 'offline' in p:
                r = controller.run_offline()
            else:
                r = controller.run()

            # store
            # hist_detailed = r.pop('hist_detailed')
            # if i < NUM_SAVE:
            #     save_iteration(os.path.join(res_path,'detailed'),
            #                    p['name'], i, hist_detailed)
            
            
            policies_run.add(p['name'])
            
        

            
            # save platform settings (don't have to do this every run, but we do)
            runinfo = {'gt_difficulties': platform.gt_difficulties.tolist(),
                       'gt_skills': platform.gt_skills.tolist(),
                       'gt_labels': platform.gt_labels.tolist()}
            

            res.update(history_rows=r,
                       run_row={'experiment_name': experiment_name_json,
                                'experiment': experiment_json,
                                'policy_name': p['name'],
                                'policy': dict((k, p[k]) for k in p if k != 'name'),
                                'seed': rint,
                                'info': runinfo})


        # save aggregate policy results (overwrite in each iter)
        # save_results(res_path, exp_name, res, i)
