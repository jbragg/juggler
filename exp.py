#!/bin/env python

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
from platform import Platform
import parse



def gen_labels(num_questions):
    """Randomly generate labels"""
    return np.round(np.random.random(num_questions))


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
        


def agg_scores(accs, x='observed'):
    """Takes accuracies and computes averages and standard errors"""
    if len(accs) == 1:
        return [(d[x], d['score'], None) for d in accs[0]]

    points = []

    def next_point(scores):
        vals = [scores[i]['score'] for i in scores]
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

def save_results(res_path, exp_name, res, accs):
    policies = accs.keys()
    scores = defaultdict(dict)
    for p in policies:
        #assert accs[p].shape[1] == NUM_QUESTIONS + 1
        scores['votes'][p] = agg_scores(accs[p],'observed')
        scores['time'][p] = agg_scores(accs[p],'time')

    # create figures and save data
    for x_type in scores:
        fname = 'plot_{}'.format(x_type)
        plt.close('all')
            
        with open(os.path.join(res_path, fname+'.csv'),'wb') as f:
            writer = csv.writer(f) 
            writer.writerow(['policy','x','y','stderr'])

            for p in policies:
                x_val, mean, stderr = zip(*scores[x_type][p])
                if any(x is None for x in stderr):
                    stderr = None
                else:
                    stderr = [x * 1.96 for x in stderr]

                # to plot
                plt.errorbar(x_val, mean, yerr=stderr, label=p)

                # to file
                if stderr is None:
                    stderr = itertools.repeat(None)

                for x,y,s in zip(x_val, mean, stderr):
                    writer.writerow([p,x,y,s])


        plt.ylim(ymax=1)
        plt.legend(loc="lower right")
        if x_type == 'votes':
            xlabel = 'Number of votes observed'
        elif x_type == 'time':
            xlabel = 'Time elapsed'
        else:
            raise Exception
        plt.xlabel(xlabel)
        plt.ylabel('Prediction accuracy')

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

    real_p = s['real'] == 'True'
    skill_dist = val_or_none(s, 'skill', lambda x: parse_skill(x))
    diff_dist = val_or_none(s, 'diff')
    time_dist = val_or_none(s, 'time')

    policies = s['policies']

    # load experimental data
    gold = parse.LoadGold(os.path.join('data','gold.txt'),
                          os.path.join('data','db_nel.csv'),
                          os.path.join('data','gold_params.csv'))

    mkdir_or_ignore('res')
    res_path = os.path.join('res',exp_name)
    mkdir_or_ignore(res_path)
   
    # copy policy file
    import shutil
    shutil.copy(sys.argv[1], res_path)



    # run experiments
    accs = defaultdict(list)
    res = dict()
    #if not real_p:
    for i in xrange(n_exps):
        rint = random.randint(0,sys.maxint)
        print '------------'
        print 'iteration: ' + str(i)
        np.random.seed(rint)
        #------ simulated data ------
        if not real_p:
            gt_labels = gen_labels(n_questions)
            platform = Platform(gt_labels,
                                num_workers=n_workers,
                                skills=skill_dist,
                                difficulties=diff_dist,
                                times=time_dist)
            greedy_once = False
        #------ cases for real data ------
        elif skill_dist or time_dist or diff_dist:
            if time_dist is None or skill_dist is None:
                assert n_workers == len(gold.get_skills())

            platform = Platform(
                    gt_labels=gold.get_gt(),
                    difficulties=diff_dist or gold.get_difficulties(),
                    times=time_dist or gold.get_times(),
                    skills=skill_dist or gold.get_skills(),
                    num_workers=n_workers)
            greedy_once = False
        else: # use params estimated from gold
            platform = Platform(gt_labels=gold.get_gt(),
                                votes=gold.get_votes(),
                                difficulties=gold.get_difficulties(),
                                skills=gold.get_skills(),
                                times=gold.get_times())
            greedy_once = True



        for p in policies:
            if greedy_once and p['type'] in ['greedy','greedy_reverse'] and p['name'] in accs:
                continue

            # run
            np.random.seed(rint)
            platform.reset()
            controller = Controller(method=p['type'],
                                    platform=platform,
                                    num_workers=n_workers,
                                    num_questions=n_questions,
                                    known_d=p['known_d'],
                                    known_s=p['known_s'])
                                    
            if 'offline' in p:
                r = controller.run_offline()
            else:
                r = controller.run()

            # store
            accs[p['name']].append(r['accuracies'])

            res[p['name'],i] = r

        # overwrite results in each iter
        save_results(res_path, exp_name, res, accs)


