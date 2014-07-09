import itertools
from collections import defaultdict, Counter
import os
import csv
import copy
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
import argparse
import json
import pyphd



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


def plot(hist, res_path):
    """
    hist = {policy: [
                     [{'time': t,
                       'observed': n,
                       'accuracy': v,
                       'exp_accuracy': v}, ... {}],
                     ...
                     [] # final iteration
                    ]}
    """
    #markers = itertools.cycle('>^+*')   
    markers = itertools.repeat(None)   
    
    for t in ('accuracy','exp_accuracy','workers_per_task'):
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
            elif t == 'workers_per_task':
                fname = 'dup_' + fname

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
                if t != 'workers_per_task':
                    plt.ylim(ymin=0.5,ymax=1)
                plt.legend(loc="lower right")
                if x_type == 'votes':
                    xlabel = 'Number of votes observed'
                elif x_type == 'time':
                    xlabel = 'Time elapsed'
                else:
                    raise Exception
                plt.xlabel(xlabel)
                if t == 'workers_per_task':
                    plt.ylabel('Average workers per task')
                else:
                    plt.ylabel('Prediction accuracy')

            plt.figure(0)
            with open(os.path.join(res_path, fname+'_err.png'),'wb') as f:
                plt.savefig(f, format="png", dpi=150)

            plt.figure(1)
            with open(os.path.join(res_path, fname+'.png'),'wb') as f:
                plt.savefig(f, format="png", dpi=150)

def load_tables(expname):
    """ Loads tables into memory for plot()
    """
    expdir = os.path.join('res', expname)
    tablesdir = os.path.join(expdir, 'tables')
    def tablepath(s):
        return os.path.join(tablesdir, expname+'-'+s+'.csv')

    runid_to_policy_seed = dict()
    with open(tablepath('runs'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            runid_to_policy_seed[row['run_id']] = (row['policy_name'],
                                                        int(row['seed']))

    res = defaultdict(list)
    with open(tablepath('history'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            policy, seed = runid_to_policy_seed[row['run_id']]

            # NOTE: assumes history written in order (which it was)
            if seed == len(res[policy]):
                res[policy].append([])
            elif seed == len(res[policy]) - 1:
                pass
            else:
                raise Exception

            other = json.loads(row['other'])

            try:
                duplicates_counter = Counter(dict((int(k),
                                                   other['duplicates'][k]) for
                                                  k in other['duplicates']))
                duplicates = np.mean(list(duplicates_counter.elements()))
            except TypeError:
                duplicates = 0
            res[policy][-1].append({
                        'observed': other['observed'],
                        'time': int(row['t_']),
                        'accuracy': other['accuracy'],
                        'exp_accuracy': other['exp_accuracy'],
                        'workers_per_task': duplicates}
                        )

    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('expname', type=str,
                        help='experiment name')
    #parser.add_argument('-p', '--policies', nargs='+', type=int,
    #                    help='list of policies')
    #parser.add_argument('-i', '--maxiter', default=-1, type=int,
    #                    help='maximum iterations/policy for detailed plots')
    args = parser.parse_args()
    res = load_tables(args.expname)
    plotdir = os.path.join('res', args.expname, 'plots')
    pyphd.ensure_dir(plotdir)
    plot(res, plotdir)

    #make_plots(args.expname, args.policies, args.maxiter)
