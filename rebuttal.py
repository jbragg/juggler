""" Code for rebuttal. Analyzes timings. """
from __future__ import division
import sys
import csv
import operator
import numpy as np
from scipy import stats
from collections import defaultdict

def relational_to_matrix(gen, col):
    results = dict()
    results2 = defaultdict(list)
    vals = defaultdict(list)

    for row in gen:
        results[row['policy'],int(row['run']),int(row['observed'])] = row[col]

    for p,r,n in sorted(results, key = operator.itemgetter(2)):
        results2[p,r].append(results[p,r,n])

    for p,r in sorted(results2, key = operator.itemgetter(1)):
        vals[p].append(results2[p,r])

    return vals


def analyze_timings(hist_fname):
    """ Analyzes timings csv with the following columns:
        - policy
        - run
        - time
        - timing
    """

    with open(hist_fname, 'r') as f:
        reader = csv.DictReader(f)
        vals = relational_to_matrix(reader, 'timing')

    result = dict()
    for p in vals:
        for i in xrange(len(vals[p])):
            # Hack: didn't record for first slot
            del vals[p][i][0]
            for j,v in enumerate(vals[p][i]):
                vals[p][i][j] = float(v)
        result[p] = (np.mean(vals[p], 0), stats.sem(vals[p], 0))
    return result

def analyze_significance(hist_fname, frac=0.95):
    """ Analyzes significance of % saved over round-robin
        NOTE: Assumes rounds
    """

    with open(hist_fname, 'r') as f:
        reader = csv.DictReader(f)
        vals = relational_to_matrix(reader, 'accuracy')

    result = dict()
    for p in vals:
        for i in xrange(len(vals[p])):
            for j,v in enumerate(vals[p][i]):
                vals[p][i][j] = float(v)
        result[p] = (np.mean(vals[p], 0), stats.sem(vals[p], 0))

    #---------- different part -------

    against = 'Round robin'
    compare = ['Greedy reverse','Accgain reverse','Uncertainty reverse']
    def find_first(lst, f):
        first_index, first_value = [(i,v) for i,v in enumerate(lst) if f(v)][0]
        return first_index, first_value

    savings = defaultdict(list)
    num_iters = len(vals.itervalues().next())
    for i in xrange(num_iters):
        best_v = max(vals[against][i])
        goal_v = frac * best_v

        first = dict()
        for p in vals:
            first[p], _ = find_first(vals[p][i], lambda v: v >= goal_v)

        for p in set(vals).difference(set([against])):
            saved = (first[against] - first[p]) / first[against]
            savings[p].append(saved)
        
    return result, savings



def count_duplicates(hist_fname):
    with open(hist_fname, 'r') as f:
        reader = csv.DictReader(f)
        vals = relational_to_matrix(reader, 'duplicates')

    result = dict()
    for p in vals:
        for i in xrange(len(vals[p])):
            # Hack: didn't record for first slot
            del vals[p][i][0]
            for j,v in enumerate(vals[p][i]):
                vals[p][i][j] = int(v)
        result[p] = (np.mean(vals[p], 0), stats.sem(vals[p], 0))
    return result




if __name__ == '__main__':
    exps = ['p-real-0-subsample','p-real-0-sim']#,'p-sim-gold-params']
    csv_files = ['res/{}/hist.csv'.format(s) for s in exps]
    fracs = [0.95, 0.97]

    acc = {}
    saved = {}

    for fname in csv_files:
#        t = analyze_timings(fname)
        for f in fracs:
            acc[f], saved[f] = analyze_significance(fname, f)
#        dups = count_duplicates(fname)
        
#        for p in dups:
            #if p.lower().startswith('greedy'):
#            if p.lower() == 'greedy reverse':
#                u, stderr = dups[p]
#                u, stderr = t[p]
#        print p
#            print u
#
#            print np.mean(u)

        print
        print '-----' + fname + '-----'
       
        compare = [('Greedy reverse', 'Accgain reverse'),
                   ('Greedy reverse', 'Uncertainty reverse')] 
        for p1, p2 in compare:

            for f in fracs:
                tval, pval = stats.ttest_rel(saved[f][p1], saved[f][p2])
                print
                print f
                print '{} saved u={}, {} saved u={}'.format(
                            p1, np.mean(saved[f][p1]),
                            p2, np.mean(saved[f][p2]))
                print 'pval={}, tval={}'.format(pval, tval)


