#!/bin/env python

from __future__ import division
import numpy as np
import itertools
from scipy.misc import logsumexp
import scipy.optimize
import scipy.stats
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import random
import pickle
import datetime
import sys
import csv
from ut import dbeta
import json
import os

NUM_WORKERS = 10
NUM_QUESTIONS = 20
NUM_EXPS = 2
MAX_T = 20000
KNOWN_D = True
KNOWN_S = True
SAMPLE = True
#LAZY = True
REAL = True
SKILL_PARAMS = (2,4,2)
####policies = ['random','greedy','greedy_ent','rr','rr_mul']
policies = ['random','greedy','greedy_reverse','rr']
#policies = ['random','rr']



class ExpState(object):
    def __init__(self, known_difficulty, known_skill,
                 num_questions=None, num_workers=None,
                 gold=None, votes=None):
        self.prior = (2,2) # assume Beta(2,2) prior
        self.known_difficulty = known_difficulty
        self.known_skill = known_skill
        self.sample = SAMPLE


        if gold is not None:
            self.gt_labels = gold
            self.gt_observations = votes
            self.num_questions = len(gold)
            self.num_workers = votes.shape[0]

            self.gt_difficulties, self.gt_skills = self.est_final_params()
            self.gt_probs = self.allprobs(self.gt_skills, self.gt_difficulties)
        else:
            self.num_questions = num_questions
            self.num_workers = num_workers

            # generate random ground truth (random for now)
            self.gt_labels = np.round(self.gen_labels())
            self.gt_difficulties = self.gen_question_difficulties()
            self.gt_skills = self.gen_worker_skills()
            self.gt_probs = self.allprobs(self.gt_skills, self.gt_difficulties)
            self.gt_observations = self.gen_observations()

        # other initialization
        self.reset()
        self.posteriors = self.prior[0] / (self.prior[0] + self.prior[1]) * \
                          np.ones(num_questions)

        self.params = None # contains MAP label posteriors, difficulties



    ############ initialization methods ##############
    def reset(self):
        """reset for run of new policy"""
        self.remaining_votes = dict([((i,j),True) for i,j in itertools.product(
                                                xrange(self.num_workers),
                                                xrange(self.num_questions))])
        self.votes = []
        self.init_observations()
        self.heuristic_vals = dict()

    def gen_labels(self):
        """generate labels from same distribution as prior"""
        return np.random.beta(*self.prior,size=self.num_questions)

    def gen_worker_skills(self):
        """Should be in range [0,inf)

        """

        # try with Chris's parameters
#        lst = []
#        while len(lst) < self.num_workers:
#            r = np.random.normal(0.5465465876, 0.0583481484308)
#            if r > 0:
#                lst.append(r)
#        return np.array(lst)

        #  --- small gammma corresponds to good workers
        return np.random.beta(SKILL_PARAMS[0],
                              SKILL_PARAMS[1],
                              self.num_workers) * SKILL_PARAMS[2]



    def gen_question_difficulties(self):
        """Should be in range [0,1]
        p_correct = 1/2(1+(1-d_q)^(1/skill)

        """

        # try with Chris's parameters
#        lst = []
#        while len(lst) < self.num_questions:
#            r = np.random.normal(0.61312076, 0.131120713602)
#            if r > 0 and r < 1:
#                lst.append(r)
#        return np.array(lst)


#        return np.ones(self.num_questions) / 2 # all problems equal difficulty
        return np.random.beta(*self.prior,size=self.num_questions)

    def gen_observations(self):
        """Generate worker votes. Unvectorized, but shouldn't matter."""
        o = np.zeros((self.num_workers, self.num_questions))
        for w,q in itertools.product(xrange(self.num_workers),
                                     xrange(self.num_questions)):
            pCorrect = np.random.random() <= self.gt_probs[w,q]
            if pCorrect:
                v = self.gt_labels[q]
            else:
                v = not self.gt_labels[q]

            o[w,q] = int(v)  # store truth val as int for consistency

        return o

    def prob_correct(self, s, d):
        """p_correct = 1/2(1+(1-d_q)^(1/skill)"""
        return 1/2*(1+np.power(1-d,s))

    def prob_correct_ddifficulty(self, s, d):
        return -1/2*s*np.power(1-d,s-1)

    def prob_correct_dskill(self, s, d):
        return 1/2*np.power(1-d,s)*np.log(1-d)


    def allprobs(self, skills, difficulties):
        """Return |workers| x |questions| matrix with prob of worker answering
        correctly.
        """
        return self.prob_correct(skills[:, np.newaxis],
                                 difficulties)

    def allprobs_ddifficulty(self, skills, difficulties):
        return self.prob_correct_ddifficulty(skills[:, np.newaxis],
                                             difficulties)

    def allprobs_dskill(self, skills, difficulties):
        return self.prob_correct_dskill(skills[:, np.newaxis],
                                        difficulties)

    def init_observations(self):
        """observations is |workers| x |questions| matrix
           -1 - unobserved
           1 - True
           0 - False

        """
        self.observations = np.zeros((self.num_workers, self.num_questions))-1
        return
    
    def rand_observations(self):
        return np.random.randint(0,3,(self.num_workers, self.num_questions))

    def est_final_params(self):
        """BUG: random for now"""

        params, posteriors = self.run_em(self.gt_observations)
        
        d = params['difficulties']
        s = params['skills']
        return d,s
        
    ######### meta methods #########

    def run(self, policy):
        print 'Policy: {}'.format(policy)
        self.reset()
        self.accuracies = []
        posteriors = []
        votes = []

        pr = PolicyRun(policy, self.gt_difficulties, self.gt_skills,
                       self.prior, self.prior, self.gt_labels.tolist())
        
        T = 0

        def update():
            self.update_posteriors()
            self.accuracies.append(self.score())
            posteriors.append(self.posteriors)

        votes.append([])
        update()
        pr.add_obs(dict(), self.posteriors.tolist(), self.score())



        while len(self.remaining_votes_list()) > 0 and T < MAX_T:
            print "Remaining votes: " + str(len(self.remaining_votes_list()))

            # select votes
            next_votes = self.select_votes(policy)

            # make observations and update
            self.observe(next_votes)
            votes.append(next_votes)
            update()
            
            T += 1

            pr.add_obs(dict([(i,int(self.gt_observations[i])) for
                             i in next_votes]),
                       self.posteriors.tolist(),
                       self.score())

        # hack: write json
        #import os
        #os.mkdir('js')  # bug: ensure directory exists
        with open('js/' + policy + \
                          str(NUM_WORKERS) + 'w' + \
                          str(NUM_QUESTIONS) + 'q' + \
                  '.json', 'w') as f:
            f.write(pr.to_json())

        beliefs_to_write = np.vstack(posteriors)

    

        with open('js/{}{}w{}q.beliefs.csv'.format(policy,
                                               NUM_WORKERS,
                                               NUM_QUESTIONS),'w') as f:
            f.write('iteration,' + ','.join(str(x) for x in xrange(beliefs_to_write.shape[1])) + '\n')
            np.savetxt(f,
                       np.hstack([np.reshape(np.arange(beliefs_to_write.shape[0]),(-1,1)),beliefs_to_write]),
                       fmt=['%d'] + ['%.02f' for x in xrange(beliefs_to_write.shape[1])],
                       delimiter=',')

        with open('js/{}{}w{}q.gt.csv'.format(policy,
                                          NUM_WORKERS,
                                          NUM_QUESTIONS),'w') as f:
            np.savetxt(f, self.gt_labels[None], fmt='%d', delimiter=',')

                            
        
#        print
#        print "**************"
#        print "RESULTS for policy " + policy
#        print "accuracies: " + str(self.accuracies)
#        print "**************"
#        print

        return {"votes": votes,
                "belief": posteriors,
                "accuracies": self.accuracies,
                "gt_difficulties": self.gt_difficulties,
                "gt_skills": self.gt_skills,
                "gt_probs": self.gt_probs,
                "gt_observations": self.gt_observations}


    def select_votes(self, policy):
        acc = []
        num_skipped = 0
        while len(acc) < min(self.num_workers, self.remaining_votes_list()):
            evals = {}
            workers_in_acc = [x[0] for x in acc]
            candidates = [i for i in self.remaining_votes if 
                          self.remaining_votes[i] and
                          i[0] not in workers_in_acc]

            # BUG: assumes uses best known skill (GT or MAP)
            if policy == 'greedy':
                max_w,_ = min(candidates,
                              key=lambda x: self.params['skills'][x[0]])
                candidates = [c for c in candidates if c[0] == max_w]
            elif policy == 'greedy_reverse':
                min_w,_ = max(candidates,
                              key=lambda x: self.params['skills'][x[0]])
                candidates = [c for c in candidates if c[0] == min_w]
            #print "candidates: " + str(candidates)

            # BUG: removed incorrect LAZY code
            if policy == 'greedy' or policy == 'greedy_reverse':
                # BUG: force greedy_skill policy to be non-lazy
#                if LAZY:
#                    # BUG: no priority queue for after first vote
#
#                    cur_max = float('-inf')
#                    cur_max_c = None
#                    if len(self.heuristic_vals) > 0:
#                        candidates.sort(key=lambda k: self.heuristic_vals[k],
#                                        reverse=True)
#                    for c in candidates:
#                        # only update if previous eval was at least as large as best
#                        if ((c in self.heuristic_vals and 
#                             self.heuristic_vals[c] > cur_max) or
#                            c not in self.heuristic_vals) :
#
#                            v = self.hXA(acc, c) - self.hXU(acc, c)
#                            if v > cur_max:
#                                cur_max,cur_max_c = v,c
#                            if len(acc) == 0: # only update for first in batch
#                                self.heuristic_vals[c] = v
#                        else:
#                            num_skipped += 1
#
#                    acc.append(cur_max_c)
#                else:
                for c in candidates:
                    evals[c] = self.hXA(acc, c) - self.hXU(acc, c)
                        
                acc.append(max(evals, key=lambda k:evals[k]))

            elif policy == 'greedy_ent':
                # BUG: doesn't use lazy eval 
                for c in candidates:
                    evals[c] = self.hXA(acc, c)

                acc.append(max(evals, key=lambda k:evals[k]))

            elif policy == 'local_s':
                # Use local search to select argmin H(U|A)
                pass

            elif policy == 'random':
                acc.append(random.choice(candidates))
            elif policy == 'same_question':
                q_in_acc = set(q for w,q in acc)
                opts = [(w,q) for w,q in candidates if q in q_in_acc]
                if opts:
                    acc.append(random.choice(opts))
                else:
                    acc.append(random.choice(candidates))
            elif policy == 'rr':
                q_remaining = np.sum(self.observations == -1, 0)
                for w,q in acc:
                    q_remaining[q] -= 1
                q_opt = set(q for w,q in candidates)
                q_sel = max(q_opt, key=lambda x: q_remaining[x])
                l = [(w,q) for w,q in candidates if q == q_sel]
                acc.append(random.choice(l))
            elif policy == 'rr_mul':
                q_remaining = np.sum(self.observations == -1, 0)
                for w,q in acc:
                    q_remaining[q] -= 1
                q_opt = set(q for w,q in candidates)
                q_sel = max(q_opt, key=lambda x: q_remaining[x])
                l = [(w,q) for w,q in candidates if q == q_sel]
                acc.append(min(l, key=lambda k: self.posteriors[k[1]]))
            else:
                raise Exception('Undefined policy')

#        print "Lazy evaluations saved: {}".format(num_skipped)
        return acc

    # compute H(X | U)
    def hXU(self, acc, x):
        # remaining is a list of (#worker, #question) pairs
        # acc is a list of (#worker, #question) pairs that have been selected

        # compute P(X,U) = P(U | obs) * P(X | U)
        # NOTE: this code works vectorized, but we don't do that for now
        w,q = x
        pTT = self.posteriors[q] * self.probs[x] # P(True, True)
        pFT = self.posteriors[q] * (1-self.probs[x]) # P(False, True)
        pFF = (1-self.posteriors[q]) * self.probs[x] # P(False, False)
        pTF = (1-self.posteriors[q]) * (1-self.probs[x]) # P(True, False)

        # H(X | U)
        hXU = -1 * (pTT * np.log(self.probs[x]) + \
                    pFT * np.log(1-self.probs[x]) + \
                    pFF * np.log(self.probs[x]) + \
                    pTF * np.log(1-self.probs[x]))

        return hXU

    def hXA(self, acc, x):
        # remaining is a list of (#worker, #question) pairs
        # acc is a list of (#worker, #question) pairs that have been selected


        # compute P(X | A)
        # sample
        if self.sample:

            def calc_samples(epsilon, delta, domain_size=2):
                """Calculate number of samples, for error epsilon,
                confidence delta. From "Near-Optimal Nonmyopic Value of
                Information in Graphical Models".

                """
                return int(np.ceil(0.5 * np.log(1/delta) * \
                                   (np.power(np.log(domain_size)/epsilon, 2))))
            

            n_samples = calc_samples(0.5, 0.05)
            hXA = 0
            for i in xrange(n_samples):
                newobs = np.copy(self.observations)

                # sample
                for w,q in acc:
                    # sample true labels using poster dist
                    # BUG : is this valid? use rejection sampling instead?

                    if np.random.random() <= self.probs[w,q]:
                        v = np.random.random() <= self.posteriors[q]
                    else:
                        v = np.random.random() > self.posteriors[q]

                    if v:
                        newobs[w,q] = 1
                    else:
                        newobs[w,q] = 0

                newposts,_ = self.infer(newobs, self.params)
                pL = newposts[x[1]]
                pCorrect = self.probs[x]
                p = np.log(np.array([
                                [pL * pCorrect, pL * (1-pCorrect)],
                                [(1-pL) * (1-pCorrect), (1-pL) * pCorrect]]))
                p = np.exp(logsumexp(p,0))
                def entropy(a):
                    return -1 * np.sum(p * np.log(p))
                hXA += entropy(p) / n_samples
            #print str(x) + ":" + str((hXA, epsilon))
        else:
            # exact conditional entropy
            # chain rule H(X,A) - H(A)
            # can compute locally for a question since worker skill known
            hXA = 0
            hA = 0

            rel_q = x[1]
            rel_acc = [(w,q) for w,q in acc if q == rel_q]
#            newobs = np.copy(self.observations)
            for tup in itertools.product(*([0,1] for
                                           x in xrange(len(rel_acc)))):
                pA1 = self.posteriors[rel_q]
                pA0 = 1-self.posteriors[rel_q]

                for ind,v in zip(rel_acc,tup):
                    if v:
                        pA1 *= self.probs[ind]
                        pA0 *= (1-self.probs[ind])
                    else:
                        pA1 *= (1-self.probs[ind])
                        pA0 *= self.probs[ind]

                pA = pA1 + pA0
                hA -= pA * np.log(pA)

                pAX11 = pA1 * self.probs[x]
                pAX01 = pA0 * (1-self.probs[x])
                pAX1 = pAX11 + pAX01
                hXA -= pAX1 * np.log(pAX1)

                pAX10 = pA1 * (1-self.probs[x])
                pAX00 = pA0 * self.probs[x]
                pAX0 = pAX10 + pAX00
                hXA -= pAX0 * np.log(pAX0)

            hXA = hXA - hA

        return hXA

    ###### inference methods #######
    
    def update_posteriors(self):
        """ keep for legacy reasons """
        params, posteriors = self.run_em()
        self.params = params
        self.posteriors = posteriors


        self.probs = self.allprobs(self.params['skills'],
                                   self.params['difficulties'])
                   
       
    def run_em(self, observations=None):
        """ learn params and posteriors """
        if observations is None:
            known_d = self.known_difficulty
            known_s = self.known_skill
            observations = self.observations
        else: # estimating using final observations (for experiment)
            known_d = False
            known_s = False
            #! NOTE: observations should be self.gt_observations
            
           

        def E(params):
            if known_s and known_d:
                post, ll = self.infer(observations,
                                      {'label': params['label'],
                                       'skills': self.gt_skills,
                                       'difficulties': self.gt_difficulties})
            else:
                post, ll = self.infer(observations, params)

                # add prior for difficulty (none for skill)
                ll += np.sum(np.log(scipy.stats.beta.pdf(params['difficulties'],
                                                 self.prior[0],
                                                 self.prior[1])))


            # add beta prior for label parameter
            ll += np.sum(np.log(scipy.stats.beta.pdf(params['label'],
                                                 self.prior[0],
                                                 self.prior[1])))


            return post, ll / self.num_questions


        def M(posteriors):
            if known_s and known_d:
                params = dict()
                params['skills'] = self.gt_skills
                params['difficulties'] = self.gt_difficulties
                params['label'] = (self.prior[0] - 1 + sum(posteriors)) / \
                                  (self.prior[0] - 1 + self.prior[1] - 1 + \
                                   self.num_questions)

                return params

            else:
                #params_array = np.append(params['difficulties'],
                #                         params['label'])
                params = dict()
                params['label'] = (self.prior[0] - 1 + sum(posteriors)) / \
                                  (self.prior[0] - 1 + self.prior[1] - 1 + \
                                   self.num_questions)


                def f(params_array):
                    if not known_d and known_s:
                        difficulties = params_array
                        skills = self.gt_skills
                    elif not known_s and known_d:
                        skills = params_array
                        difficulties = self.gt_difficulties
                    else: 
                        difficulties = params_array[:self.num_questions]
                        skills = params_array[self.num_questions:]



                    probs = self.allprobs(skills,
                                          difficulties)
                    probs_dd = self.allprobs_ddifficulty(skills,
                                                         difficulties)
                    probs_ds = self.allprobs_dskill(skills,
                                                    difficulties)

#                    priors = prior * np.ones(self.num_questions)

                    true_votes = (observations == 1)   
                    false_votes = (observations == 0)   



#                    ptrue = np.log(priors) + \
                    ptrue = \
                            np.sum(np.log(probs) * true_votes, 0) + \
                            np.sum(np.log(1-probs) * false_votes, 0)
#                    pfalse = np.log(1-priors) + \
                    pfalse = \
                             np.sum(np.log(probs) * false_votes, 0) + \
                             np.sum(np.log(1-probs) * true_votes, 0)

                    ptrue_dd = \
                            np.sum(1/probs*probs_dd * true_votes, 0) + \
                            np.sum(1/(1-probs)*(-probs_dd) * false_votes, 0)

                    pfalse_dd = \
                            np.sum(1/probs*probs_dd * false_votes, 0) + \
                            np.sum(1/(1-probs)*(-probs_dd) * true_votes, 0)

                    ptrue_ds = \
                            1/probs*probs_ds * true_votes + \
                            1/(1-probs)*(-probs_ds) * false_votes

                    pfalse_ds = \
                            1/probs*probs_ds * false_votes + \
                            1/(1-probs)*(-probs_ds) * true_votes

                    # result
                    v = np.sum(posteriors * ptrue + (1-posteriors) * pfalse)
                    dd = np.array(posteriors * ptrue_dd + \
                                  (1-posteriors) * pfalse_dd)
                    ds = np.sum(posteriors * ptrue_ds + \
                                  (1-posteriors) * pfalse_ds, 1)


                    #dd = np.append(dd, np.sum(posteriors * 1/priors + \
                    #                          (1-posteriors) * -1/(1-priors)))

#                    print '---'
#                    print params_array
#                    print -v
#                    print
#                    print
#                    print
#                print dd
#                    print '---'

                    # TODO: FIX dV
                    pr = scipy.stats.beta.pdf(difficulties,*self.prior)

                    #                    print '************jjjjjj'
                    v += np.sum(np.log(pr))
                    dd += 1/pr * dbeta(difficulties,*self.prior)
                    #print difficulties, -v, -dd

                    if not known_d and known_s:
                        jac = dd 
                    elif not known_s and known_d:
                        jac = ds
                    else: 
                        jac = np.hstack((dd,ds))


                    # return negative to minimizer
                    return (-v,
                            -jac)
                    #                    return -v


                init_d = 0.1 * np.ones(self.num_questions)
                bounds_d = [(0.0000000001,0.9999999999) for 
                           i in xrange(self.num_questions)]
                init_s = 0.1 * np.ones(self.num_workers)
                bounds_s = [(0.0000000001,None) for 
                           i in xrange(self.num_workers)]

                if not known_d and known_s:
                    init = init_d
                    bounds = bounds_d
                elif not known_s and known_d:
                    init = init_s
                    bounds = bounds_s
                else: 
                    init = np.hstack((init_d,init_s))
                    bounds = bounds_d + bounds_s

                res = scipy.optimize.minimize(
                            f,
                            init,
                            method='L-BFGS-B',
                            jac=True,
                            bounds=bounds,
                            options={'disp':False})
#                print res.x
                if not known_d and known_s:
                    params['difficulties'] = res.x
                    params['skills'] = self.gt_skills
                elif not known_s and known_d:
                    params['skills'] = res.x
                    params['difficulties'] = self.gt_difficulties
                else: 
                    params['difficulties'] = res.x[:self.num_questions]
                    params['skills'] = res.x[self.num_questions:]
 
                return params
#                return {'label': res.x[self.num_questions],
#                        'difficulties': res.x[0:self.num_questions]}

        ll = float('-inf')
        ll_change = float('inf')
        params = {'difficulties':np.random.random(self.num_questions),
                  'skills':np.random.random(self.num_workers),
                  'label':np.random.random()}
        em_round = 0
        while ll_change > 0.001:  # run while ll increase is at least .1%
#            print 'EM round: ' + str(em_round)
            posteriors,ll_new = E(params)
            params = M(posteriors)

            if ll == float('-inf'):
                ll_change = float('inf')
            else:
                ll_change = (ll_new - ll) / np.abs(ll) # percent increase

            ll = ll_new
#            print 'em_round: ' + str(em_round)
#            print 'll_change: ' + str(ll_change)
#            print 'log likelihood: ' + str(ll)
#            print params['difficulties']
            assert ll_change > -0.001  # ensure ll is nondecreasing
            em_round += 1

#        print str(em_round) + " EM rounds"
#        print params['label']
#        print params['difficulties']
        return params, posteriors


    def infer(self, observations, params):
        """Probabilistic inference for question posteriors, having observed
        observations matrix. 
        """

        prior = params['label']
        probs = self.allprobs(params['skills'], params['difficulties'])
        priors = prior * np.ones(self.num_questions)
         
        true_votes = (observations == 1)   
        false_votes = (observations == 0)   

        ptrue = np.log(priors) + np.sum(np.log(probs) * true_votes, 0) + \
                                 np.sum(np.log(1-probs) * false_votes, 0)
        pfalse = np.log(1-priors) + np.sum(np.log(probs) * false_votes, 0) + \
                                    np.sum(np.log(1-probs) * true_votes, 0)


        norm = np.logaddexp(ptrue, pfalse)

        return np.exp(ptrue) / np.exp(norm), np.sum(norm)

    def observe(self, votes):
        """update self.observations ... mutation only"""

        for w,q in votes:

            # update observations matrix
            self.observations[w,q] = self.gt_observations[w,q]

            # update remaining / observed
            self.remaining_votes[w,q] = False
            self.votes.append((w,q))

#        print 'observing votes ' + str(votes)

        return 

    def remaining_votes_list(self):
        return [x for x in self.remaining_votes if self.remaining_votes[x]]

    def score(self):
        """return accuracy

        todo: return fscore?
        """

        correct = np.round(self.posteriors) == self.gt_labels
        accuracy = np.sum(correct) / len(correct)
        return accuracy
        


    def __str__(self):
        return "Observations:\n" + np.array_str(self.observations) + "\n" + \
               "Remaining:\n" + str(self.remaining_votes_list())

class Result(object):
    def __init__(self, res):
        """hack: res is an object defined in run() above"""
        self.res = res

    def get_keys(self):
        return self.res.keys()

    def get_iteration_num(self):
        return max([i for p,i in self.get_keys()]) + 1

    def get_policies(self):
        return set(p for p,i in self.get_keys())

    def iterator(self, p):
        """iterate through votes, beliefs for policy"""
        it = itertools.chain.from_iterable(
                (zip(self.res[p,i]['votes'],self.res[p,i]['belief']) for
                 i in xrange(self.get_iteration_num())))
        return it

class PolicyRun(object):
    def __init__(self, policy, difficulties, skills, priord, priorl, gt):
        self.policy = policy
        self.difficulties = difficulties
        self.skills = skills
        self.priord = priord
        self.priorl = priorl
        self.num_questions = len(difficulties)
        self.num_workers = len(skills)
        self.obs = []
        self.scores = []
        self.gt_labels = gt

    def add_obs(self, e, b, s):
        """e is a dictionary of type ((worker, question), value)
           s is the score at that round 
           b is belief at that round
        """
        self.obs.append((e,b,s))
             

    def to_json(self):
        l = []
        for i,(e,b,s) in enumerate(self.obs):
            d = {'score':s, 'belief':b, 'votes':[]}
#            print e
            for (w,q),v in e.iteritems():
                d['votes'].append({'round': i,
                                   'worker':w,
                                   'question':q,
                                   'vote':v})
            l.append(d)

        j = {'iters': l,
             'difficulties': self.difficulties.tolist(),
             'skills': self.skills.tolist(),
             'goldLabels': self.gt_labels}

        return json.dumps(j)
        



if __name__ == '__main__':
    accs = dict()
    res = dict()
    if not REAL:
        for i in xrange(NUM_EXPS):
            rint = random.randint(0,sys.maxint)
            print '------------'
            print 'iteration: ' + str(i)
            np.random.seed(rint)
            new_state = ExpState(KNOWN_D, KNOWN_S,
                                 num_questions=NUM_QUESTIONS,
                                 num_workers=NUM_WORKERS)
            print new_state.gt_difficulties
            print new_state.gt_skills
            print new_state.gt_probs
            for p in policies:
                np.random.seed(rint)
                r = new_state.run(p)
                if i == 0:
                    accs[p] = np.array(r['accuracies'])
                else:
                    accs[p] = np.vstack((accs[p], np.array(r['accuracies'])))

                res[p,i] = r
    else:
        # load experimental data
        with open('data/db_nel.csv','r') as f, open('data/gold.txt','r') as f_gold:
            d = dict()
            reader = csv.DictReader(f)
            for r in reader:
                w = r['nel.worker']
                v = int(r['nel.response'])
                item = int(r['nel.itemid'])
                if w not in d:
                    d[w] = dict()
                d[w][item] = v

            max_len = max(len(d[w]) for w in d)
            w_completed = sorted([w for w in d if len(d[w]) == max_len])
            q_list = sorted(d[w_completed[0]].keys())

            votes = np.array([[d[w][k] for k in q_list] for
                              w in w_completed])


            gold = [int(x.strip()) for x in f_gold]
            gold = np.array([gold[i] for i in q_list]) # filter

        
        for i in xrange(NUM_EXPS):
            rint = random.randint(0,sys.maxint)
            new_state = ExpState(KNOWN_D, KNOWN_S, gold=gold, votes=votes)
            for p in policies:
                # only run once for greedy policies
                if p in ['greedy','greedy_reverse'] and p in accs:
                    continue

                np.random.seed(rint)
                r = new_state.run(p)
                if i == 0:
                    accs[p] = np.array(r['accuracies'])
                else:
                    accs[p] = np.vstack((accs[p], np.array(r['accuracies'])))

                res[p,i] = r

        for i,w in enumerate(w_completed):
            print w, 1/new_state.gt_skills[i]
        print new_state.gt_difficulties





    mean = dict()
    stderr = dict()
    for p in policies:
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

    t = str(datetime.datetime.now())
    os.mkdir(t)



    # create figure
    for p in policies:
        plt.errorbar(xrange(len(mean[p])), mean[p], yerr=stderr[p], label=p)


    plt.ylim(ymax=1)
    plt.legend(loc="lower right")
    plt.xlabel('Number of iterations (batch in each iteration)')
    plt.ylabel('Prediction accuracy')
    with open(t+'/res2.png','wb') as f:
        plt.savefig(f, format="png", dpi=150)


    # save data to file
    d = dict()
    with open(t+'/res2.csv','wb') as f:
        rows = [['policy','type'] + range(len(mean[policies[0]]))]
        for p in policies:
            rows.append([p] + ['mean'] + list(mean[p]))
            if stderr[p] is not None:
                rows.append([p] + ['stderr'] + list(stderr[p]))
        writer = csv.writer(f) 
        writer.writerows(rows)


    pickle.dump(Result(res), open(t+'/dump.txt','w'))
