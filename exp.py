#!/bin/env python

from __future__ import division
import numpy as np
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
        


def save_results(res_path, exp_name, res, accs):
    mean = dict()
    stderr = dict()
    for p in policies:
        p = p['name']
        #assert accs[p].shape[1] == NUM_QUESTIONS + 1
        if len(accs[p].shape)==2:
            mean[p] = np.mean(accs[p],0)
            stderr[p] = 1.96 * np.std(accs[p],0) / np.sqrt(accs[p].shape[0])
        else:
            mean[p] = accs[p]
            stderr[p] = None
        #print
        #print p + ':'
        #print np.mean(accs[p],0)


#new_state.update_posteriors()
#print new_state.observations
#print new_state.params
#new_state.infer(new_state.observations, new_state.params)


    # create figure
    plt.close('all')
    for p in policies:
        p = p['name']
        plt.errorbar(xrange(len(mean[p])), mean[p], yerr=stderr[p], label=p)


    plt.ylim(ymax=1)
    plt.legend(loc="lower right")
    plt.xlabel('Number of iterations (batch in each iteration)')
    plt.ylabel('Prediction accuracy')
    with open(os.path.join(res_path,'plot.png'),'wb') as f:
        plt.savefig(f, format="png", dpi=150)


    # save data to file
    d = dict()
    with open(os.path.join(res_path,'plot.csv'),'wb') as f:
        num_iterations = len(mean[policies[0]['name']])
        rows = [['policy','type','iteration','val']]
        for p in policies:
            p = p['name']
            for i in xrange(num_iterations):
                rows.append([p, 'mean', i, mean[p][i]])
            for i in xrange(num_iterations):
                if stderr[p] is not None:
                    rows.append([p, 'stderr', i, stderr[p][i]])
        writer = csv.writer(f) 
        writer.writerows(rows)


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

    exp_name = s['exp_name']
    n_workers = int(s['n_workers'])
    n_questions = int(s['n_questions'])
    sample_p = bool(s['sample'])
    n_exps = int(s['n_exps'])

    real_p = s['real'] == 'True'
    if 'skill' in s:
        skill_dist = parse_skill(s['skill'])
    else:
        skill_dist = None

    # BUG: hard-code difficulty parameters (should move to policy)
    diff_dist = {'type': 'beta',
                 'params': [1,1]}

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
    accs = dict()
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
                                difficulties=diff_dist)
            greedy_once = False
        #------ cases for real data ------
        elif 'skill' in s:
            platform = Platform(gt_labels=gold.get_gt(),
                                difficulties=gold.get_difficulties(),
                                skills=skill_dist,
                                num_workers=n_workers)
            greedy_once = False
        else: # use params estimated from gold
            platform = Platform(gt_labels=gold.get_gt(),
                                votes=gold.get_votes(),
                                difficulties=gold.get_difficulties(),
                                skills=gold.get_skills())
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
            if i == 0:
                accs[p['name']] = np.array(r['accuracies'])
            else:
                accs[p['name']] = np.vstack(
                            (accs[p['name']], np.array(r['accuracies'])))

            res[p['name'],i] = r

        # overwrite results in each iter
        save_results(res_path, exp_name, res, accs)


