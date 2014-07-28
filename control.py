"""control.py
controllers"""

from __future__ import division
import numpy as np
import itertools
import collections
from collections import defaultdict
from scipy.misc import logsumexp
from scipy.special import betainc
import scipy.optimize
import scipy.stats
import random
import pickle
import heapq
import sys
import operator
import csv
from ut import dbeta
import json
import os
import time
import munkres
import pyphd
from extra import PlotController


class Controller():
    def __init__(self, policy,
                 platform,
                 num_questions, num_workers,
                 sample=True,
                 max_rounds=float('inf')):

        # parse policy and set defaults
        # method can be 'random', 'rr', 'greedy', 'greedy_reverse"
        self.policy = policy
        self.max_rounds = max_rounds
        self.method = policy['type']
        for s in (policy['known_d'],policy['known_s']):
            assert s == 'True' or s == 'False'
        self.known_difficulty = eval(policy['known_d'])
        self.known_skill = eval(policy['known_s'])
        self.post_inf = pyphd.dict_val_or(policy, 'eval', 'reg')
        self.maxdup = pyphd.dict_val_or(policy, 'max_dup', float('inf'))
        self.at_least_once = bool(pyphd.dict_val_or(policy,
                                                    'at_least_once',
                                                    False))
        self.policy['efficient'] = bool(pyphd.dict_val_or(policy,
                                                          'efficient',
                                                          False))
        self.policy['num_bins'] = pyphd.dict_val_or(policy,
                                                   'num_bins',
                                                   10)
        num_bins = self.policy['num_bins']


        self.platform = platform
        self.gt_labels = platform.gt_labels

#        self.bin_hash = dict()
        self.q_queue = []
        self.q_to_add = [] # list of questions to add to self.q_queue

        self.prior = (1,1) # prior for diff (was prior for diff and label)
        self.sample = sample

        self.bins = dict()
        self.bin_midpoints = dict()
        if self.known_skill:
            self.gt_skills = platform.gt_skills
            assert self.gt_skills is not None
            self.bins['gt_skills'] = np.linspace(min(self.gt_skills),
                                                 max(self.gt_skills),
                                                 num_bins+1)
            self.bin_midpoints['gt_skills'] = pyphd.bin_midpoints(self.bins['gt_skills'])
        else:
            self.gt_skills = None

        if self.known_difficulty:
            self.gt_difficulties = platform.gt_difficulties
            assert self.gt_difficulties is not None
            self.bins['gt_difficulties'] = np.linspace(0, 1, num_bins+1)
            self.bin_midpoints['gt_difficulties'] = pyphd.bin_midpoints(self.bins['gt_difficulties'])
            
        else:
            self.gt_difficulties = None

        self.bins['posteriors'] = np.linspace(0, 1, num_bins+1)
        self.bin_midpoints['posteriors'] = pyphd.bin_midpoints(self.bins['posteriors'])
        
        # calculate bin values for entropy gain and accuracy gain
        # HACK
        sub_controller = PlotController(skills=1/np.array(self.gt_skills),
                                        difficulties=self.bin_midpoints['gt_difficulties'],
                                        posteriors=self.bin_midpoints['posteriors'])
        self.bin_values = dict()
        self.bin_orders = dict()
        self.bin_values['ent'] = sub_controller.get_bin_values(method='entropy')
        self.bin_values['acc'] = sub_controller.get_bin_values(method='accuracy')
        for k in ['ent','acc']:
            self.bin_orders[k] = dict()
            for w in self.bin_values[k]:
                d = pyphd.array_to_dict(self.bin_values[k][w])
                self.bin_orders[k][w] = sorted(d,
                                               key=lambda x: d[x],
                                               reverse=True)

        self.num_questions = num_questions
        self.num_workers = num_workers


        self.params = defaultdict(dict)
        
        self.params['kg'] = dict()
        self.params['kg']['skills'] = [(4,2) for w in xrange(num_workers)]
        self.params['kg']['thetas'] = [(1.0001,1.0001) for q in xrange(num_questions)]

        self.params['kg_lbfgs'] = {'skills': np.array([.75 for w in xrange(num_workers)]),
                                   'thetas': np.array([.5 for q in xrange(num_questions)])}
        
        
        # other initialization
        self.reset()
        self.update_posteriors([])
        self.init_bins()
#        self.posteriors = self.prior[0] / (self.prior[0] + self.prior[1]) * \
#                          np.ones(num_questions)
#        self.posteriors = 0.5 * np.ones(self.num_questions)



    #----------- initialization methods --------------
    def reset(self):
        """reset for run of new policy"""

        # vote_status = None if unassigned
        # vote_status = 0 if assigned
        # vote_status = 1 if observed
        self.vote_status = dict([((i,j), None) for i,j in itertools.product(
                                                xrange(self.num_workers),
                                                xrange(self.num_questions))])
        self.init_observations()
        self.time_elapsed = 0

        self.hist = []
        self.hist_detailed = []
        self.votes_hist = []
        self.posteriors_hist = []
        self.accuracies = []
        self.worker_finished = dict()

    def reset_offline(self):
        """reset for offline policy"""

        # vote_status = None if unassigned
        # vote_status = 0 if assigned
        # vote_status = 1 if observed
        self.vote_status = dict([((i,j), None) for i,j in itertools.product(
                                                xrange(self.num_workers),
                                                xrange(self.num_questions))])
        self.init_observations()
        self.posteriors = 0.5 * np.ones(self.num_questions)




    def init_observations(self):
        """observations is |workers| x |questions| matrix
           -1 - unobserved
           1 - True
           0 - False

        """
        self.observations = np.zeros((self.num_workers, self.num_questions))-1
        return

    def init_bins(self):
        """Initialize question bin locations, based on posterior and difficulty
        """
        self.bin_to_q = dict()
        for w in xrange(self.num_workers):
            self.bin_to_q[w] = defaultdict(set)

        self.set_bins(xrange(self.num_questions))

    
    def set_bins(self, questions):
        """Set bins
        """
        for q in questions:
            for w in xrange(self.num_workers):
                _, question_bin = self.to_bin((None, q))
                if self.vote_status[w,q] is None:
                    self.bin_to_q[w][question_bin].add(q)

        
    #--------- utility -------------
    def get_votes(self, status):
        if status == 'unassigned':
            f = lambda x: x is None
        elif status == 'assigned':
            f = lambda x: x == 0
        elif status == 'observed':
            f = lambda x: x == 1
        elif status == 'unobserved':
            f = lambda x: x != 1
        else:
            raise Exception('Undefined vote status')

        return [v for v in self.vote_status if f(self.vote_status[v])]


    def to_bin(self, vote):
        """
        input: vote (w,q)
        output: worker bin, question bin
        """
        w,q = vote
        if w is None:
            worker_bin = None
        else:
            worker_bin = pyphd.digitize_right_closed([self.gt_skills[w]],  
                                                     self.bins['gt_skills'])[0]
        
        if q is None:
            question_bin = None
        else:
            diff_bin = pyphd.digitize_right_closed([self.gt_difficulties[q]],
                                                   self.bins['gt_difficulties'])[0]
            posterior_bin = pyphd.digitize_right_closed([self.posteriors[q]],
                                                        self.bins['posteriors'])[0]
            question_bin = (diff_bin, posterior_bin)
        
        return (worker_bin, question_bin)
        
    #--------- probability --------
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


    #---------- inference --------

    def run_em(self, gt_observations=None):
        """Learn params and posteriors"""
        if gt_observations is None:
            known_d = self.known_difficulty
            known_s = self.known_skill      
            observations = self.observations
        else: # estimating using final observations (for experiment)
            known_d = False
            known_s = False
            observations = gt_observations
            #! NOTE: observations should be self.gt_observations
            
            

        def E(params):
            if known_s and known_d:
                post, ll = self.infer(observations,
                                      {'label': params['label'],
                                       'skills': self.gt_skills,
                                       'difficulties': self.gt_difficulties})
            else:
                post, ll = self.infer(observations, params)
            
                if not known_d:
                # add prior for difficulty (none for skill)
                    ll += np.sum(np.log(scipy.stats.beta.pdf(
                                            params['difficulties'],
                                            self.prior[0],
                                            self.prior[1])))
                
                                  
            # add beta prior for label parameter
            #ll += np.sum(np.log(scipy.stats.beta.pdf(params['label'],
            #                                     self.prior[0],
            #                                     self.prior[1])))
                
            
            return post, ll / self.num_questions

       
        def M(posteriors, params_in):
            if known_s and known_d:
                params = dict()
                params['skills'] = self.gt_skills
                params['difficulties'] = self.gt_difficulties
                #params['label'] = (self.prior[0] - 1 + sum(posteriors)) / \
                #                  (self.prior[0] - 1 + self.prior[1] - 1 + \
                #                   self.num_questions)
                params['label'] = 0.5 # hard-code for this exp

                return params

            else:
                #params_array = np.append(params['difficulties'],
                #                         params['label'])
                params = dict()
                #params['label'] = (self.prior[0] - 1 + sum(posteriors)) / \
                #                  (self.prior[0] - 1 + self.prior[1] - 1 + \
                #                   self.num_questions)
                params['label'] = 0.5 # hard-code for this exp


                def f(params_array):
                    if not known_d and known_s:
                        difficulties = params_array
                        skills = self.gt_skills
                    elif not known_s and known_d:
                        skills = params_array
                        difficulties = self.gt_difficulties
                    else:  # both skill and difficulty unknown
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
#                init_s = 0.9 * np.ones(self.num_workers)
                init_s = params_in['skills']
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
#                print 'success: ',res.success
                if not known_d and known_s:
                    params['difficulties'] = res.x
                    params['skills'] = self.gt_skills
                elif not known_s and known_d:
                    params['skills'] = res.x
                    params['difficulties'] = self.gt_difficulties
                else: 
                    params['difficulties'] = res.x[:self.num_questions]
                    params['skills'] = res.x[self.num_questions:]
 

#                print params['skills']
#                print params['difficulties']
                return params
#                return {'label': res.x[self.num_questions],
#                        'difficulties': res.x[0:self.num_questions]}

        # hack for running only M-step for initial gold estimate
        if gt_observations is not None:
            return M(self.gt_labels)

        ll = float('-inf')
        ll_change = float('inf')
        params = {'difficulties':np.random.random(self.num_questions),
                  'skills':np.random.random(self.num_workers),
                  #'label':np.random.random()}
                  'label':0.5}
        em_round = 0
        while ll_change > 0.001:  # run while ll increase is at least .1%
#            print 'EM round: ' + str(em_round)
            posteriors,ll_new = E(params)
            params = M(posteriors, params)

            if ll == float('-inf'):
                ll_change = float('inf')
            else:
                ll_change = (ll_new - ll) / np.abs(ll) # percent increase

            ll = ll_new
            # print 'em_round: ' + str(em_round)
            # print 'll_change: ' + str(ll_change)
            # print 'log likelihood: ' + str(ll)
            # print params['skills'][:5]
            # print

            # NOTE: good to have this assert, but fails w/ gradient ascent
            #assert ll_change > -0.001  # ensure ll is nondecreasing

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

        # log P(U = true,  votes)
        ptrue = np.log(priors) + np.sum(np.log(probs) * true_votes, 0) + \
                                 np.sum(np.log(1-probs) * false_votes, 0)
        # log P(U = false,  votes)
        pfalse = np.log(1-priors) + np.sum(np.log(probs) * false_votes, 0) + \
                                    np.sum(np.log(1-probs) * true_votes, 0)

        # log P(U = true | votes)
        norm = np.logaddexp(ptrue, pfalse)

        return np.exp(ptrue) / np.exp(norm), np.sum(norm)

    #----- KG inference functions
    def new_params(self, w, q):
        a,b = self.params['kg']['thetas'][q]
        c,d = self.params['kg']['skills'][w]
        
        new = dict()

        # v=1 case
        e_1q = a*((a+1)*c + b*d) / ((a+b+1)*(a*c + b*d))
        e_2q = a*(a+1)*((a+2)*c + b*d) / ((a+b+1)*(a+b+2)*(a*c + b*d))
            
        e_1w = c*(a*(c+1) + b*d) / ((c+d+1)*(a*c + b*d))
        e_2w = c*(c+1)*(a*(c+2) + b*d) / ((c+d+1)*(c+d+2)*(a*c + b*d))
        
        a_new = e_1q * (e_1q - e_2q) / (e_2q - e_1q**2)
        b_new = (1-e_1q) * (e_1q - e_2q) / (e_2q - e_1q**2)

        c_new = e_1w * (e_1w - e_2w) / (e_2w - e_1w**2)
        d_new = (1-e_1w) * (e_1w - e_2w) / (e_2w - e_1w**2)
        
        new[1] = ((a_new, b_new), (c_new, d_new))
        
        # v=0 case
        e_1q = a*(b*c + (a+1)*d) / ((a+b+1)*(b*c + a*d))
        e_2q = a*(a+1)*(b*c + (a+2)*d) / ((a+b+1)*(a+b+2)*(b*c + a*d))
            
        e_1w = c*(b*(c+1) + a*d) / ((c+d+1)*(b*c + a*d))
        e_2w = c*(c+1)*(b*(c+2) + a*d) / ((c+d+1)*(c+d+2)*(b*c + a*d))
            
        a_new = e_1q * (e_1q - e_2q) / (e_2q - e_1q**2)
        b_new = (1-e_1q) * (e_1q - e_2q) / (e_2q - e_1q**2)

        c_new = e_1w * (e_1w - e_2w) / (e_2w - e_1w**2)
        d_new = (1-e_1w) * (e_1w - e_2w) / (e_2w - e_1w**2)
        
        new[0] = ((a_new, b_new), (c_new, d_new))
        
        return new


    def update_params_kg(self, w, q, v):
        new_params = self.new_params(w, q)
        
        ((a,b),(c,d)) = new_params[v]
        
        self.params['kg']['thetas'][q] = (a, b)
        self.params['kg']['skills'][w] = (c, d)

        
    def optimize_kg(self):
        assert not self.known_difficulty
        assert not self.known_skill

        v1 = self.observations == 1
        v0 = self.observations == 0
        
        q_prior = (1.1, 1.1)
        w_prior = (4,2)
        
        def f(params):
            """Returns likelihood of [thetas, skills]"""
            params_q = params[:self.num_questions]
            params_w = params[self.num_questions:]
            params_w_t = params[self.num_questions:][:, np.newaxis]
            
            
            # calculate loglikelihood

            # print '1votes,', (v1 * params_q * params_w_t + v1 * (1-params_q) * (1-params_w_t))
            # print '2votes,', (v0 * (1-params_q) * params_w_t + v0 * params_q * (1-params_w_t))
            ll = 0
            ll += np.sum(v1 * np.logaddexp(
                            np.log(params_q * params_w_t),
                            np.log((1-params_q) * (1-params_w_t))))
            # print '1: ',v1 * np.logaddexp(
            #                 np.log(params_q * params_w_t),
            #                 np.log((1-params_q) * (1-params_w_t)))
            
            ll += np.sum(v0 * np.logaddexp(
                            np.log((1-params_q) * params_w_t),
                            np.log(params_q * (1-params_w_t))))
            # print '0: ',v0 * np.logaddexp(
            #                 np.log((1-params_q) * params_w_t),
            #                 np.log(params_q * (1-params_w_t)))

            prior_q = np.log(scipy.stats.beta.pdf(params_q, *q_prior))
            prior_w = np.log(scipy.stats.beta.pdf(params_w, *w_prior))
            
            ll += np.sum(prior_q) + np.sum(prior_w)
            
            # print 'll: ',ll
        
            # calculate gradient            
            # print params_q*params_w_t + (1-params_q)*(1-params_w_t)
            # print 1/(params_q*params_w_t + (1-params_q)*(1-params_w_t))
            # print 1/(params_q*params_w_t + (1-params_q)*(1-params_w_t)) *\
            #              (2*params_w_t - 1)
            # print v1 * (1/(params_q*params_w_t + (1-params_q)*(1-params_w_t)) *\
            #              (2*params_w_t - 1))
            # print
            dldq = v1 * (1/(params_q*params_w_t + (1-params_q)*(1-params_w_t)) *\
                         (2*params_w_t - 1))
            dldw = v1 * (1/(params_q*params_w_t + (1-params_q)*(1-params_w_t)) *\
                         (2*params_q - 1))

            # print 'dldq',dldq

            dldq += v0 * (1/((1-params_q)*params_w_t + params_q*(1-params_w_t)) *\
                         (-2*params_w_t + 1))
            dldw += v0 * (1/((1-params_q)*params_w_t + params_q*(1-params_w_t)) *\
                         (-2*params_q + 1))

            # print 'dldq',dldq
                         
            dldq = np.zeros(self.num_questions) + np.sum(dldq, axis=0)
            dldw = np.zeros(self.num_workers) + np.sum(dldw, axis=1)
        

            # print ':',dbeta(params_q, *q_prior)
            dldq += 1/prior_q * dbeta(params_q, *q_prior)
            dldw += 1/prior_w * dbeta(params_w, *w_prior)
            
            
            # print 'q: ',params_q
            # print 'dldq: ',dldq
            # print 'w: ',params_w
            # print 'dldw: ',dldw
            # print 'll: ',ll
            # print
            
            return (-ll, -np.hstack((dldq, dldw)))
#            return -ll


        res = scipy.optimize.minimize(
                    f,
                    # 0.5 * np.ones(self.num_questions + self.num_workers),
                    np.hstack((self.params['kg_lbfgs']['thetas'],
                               self.params['kg_lbfgs']['skills'])),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=[(0.000000001,0.9999999999) for 
                            i in xrange(self.num_questions + self.num_workers)],
                    options={'disp': False})
#                             'maxiter': 50})
#                print res.x

        # print 'f: ', res.fun
        print 'success: ', res.success
        params = dict()
        params['thetas'] = res.x[:self.num_questions]
        params['skills'] = res.x[self.num_questions:]
        
        posteriors = np.copy(params['thetas'])
        
        return params, posteriors


    #----------- control -----------

    def select_votes_offline(self, depth):
        assert self.known_difficulty and self.known_skill
        assert depth > 0 and depth <= self.num_questions

        policy = self.method

        if policy == 'greedy':
            w_order = sorted(xrange(self.num_workers),
                             key=lambda i:self.params['reg']['skills'][i])
        elif policy == 'greedy_reverse':
            w_order = sorted(xrange(self.num_workers),
                             key=lambda i:-self.params['reg']['skills'][i])
        else:
            raise 'Undefined policy'

        print 'Depth: {0}'.format(depth)
        acc = []
        #print 'worker skills', [(w,self.params['skills'][w]) for w in w_order]
        for w in w_order:
            h = [(float('-inf'),(w,q)) for q in xrange(self.num_questions)]
            heapq.heapify(h)

            for i in xrange(depth):
                max_v = float('-inf')
                max_c = None

                # use -min instead of max because min heap
                while max_v < -min(h)[0]:
                    _, c = heapq.heappop(h)

                    v = self.hXA(acc, c) - self.hXU(acc, c)

                    # push negative val because min heap
                    heapq.heappush(h,(-v,c))

                    if max_v < v:
                        max_v = v
                        max_c = c

                acc.append(max_c)

                h = [(v,c) for v,c in h if c != max_c]
                heapq.heapify(h)

        return acc


    def select_votes(self):
        policy = self.method
        num_skipped = 0

        # get votes with different statuses (messy)
        assigned_votes = self.get_votes('assigned')
        unobserved_votes = self.get_votes('unobserved')
        unassigned_votes = self.get_votes('unassigned')
        acc = assigned_votes

        # store values considered when making decisions
        alternatives = []
        inner_times = []
        
        assigned_workers = {w for w,q in acc}
        unassigned_workers = {w for w in xrange(self.num_workers) if
                              w not in assigned_workers}
        if self.post_inf == 'reg':
            workers_good_to_bad = sorted(
                            unassigned_workers,
                            key=lambda x: self.params['reg']['skills'][x])
            workers_bad_to_good = list(workers_good_to_bad)
            workers_bad_to_good.reverse()

        candidates_all = defaultdict(set)
        for w,q in unassigned_votes:
            if w in unassigned_workers:
                candidates_all[w].add((w,q))
        
        q_asked = np.sum(self.observations != -1, 0)
        q_remaining = np.sum(self.observations == -1, 0)
        q_assigned = np.zeros(self.num_questions)

        for w,q in acc:
            q_asked[q] += 1
            q_assigned[q] += 1
            q_remaining[q] -= 1

        # define evaluation functions
        if policy in ['greedy', 'greedy_reverse', 'greedy_matching']:
            eval_f = lambda c, acc: self.ent_gain(acc, c)
            eval_metric= 'ent'
        elif policy in ['accgain', 'accgain_reverse', 'accgain_matching']:
            eval_f = lambda c, acc: self.acc_gain(acc, [c])
            eval_metric = 'acc'
        elif policy == 'greedy_ent':
            eval_f = lambda c, acc: self.hXA(acc, c)

        def eval_f_approx(c, metric):
            skill_bin, question_bin = self.to_bin(c)
            return self.bin_values[metric][skill_bin][question_bin]
                                                
        
        # # arbitrarily use worker 0 to evaluate question value
        # # POSSIBLE BUG: what if worker 0 is in acc?
        # eval_f_gen = lambda q, acc: eval_f((0, q), acc)
        #
        # # hack to load all questions into the queue at the first time step
        # if self.policy['efficient'] and np.sum(q_asked) == 0:
        #     self.q_to_add = range(self.num_questions)
        #
        # # add questions asked or skipped back into the queue
        # # NOTE: re-evaluating skipped questions when we may not need to do so
        # while len(self.q_to_add) > 0:
        #     # evaluate and use negative value to use min heap as max heap
        #     q_ = self.q_to_add.pop()
        #     v = -1 * eval_f_gen(q_, acc)
        #     print 'adding (v, q_)'
        #     heapq.heappush(self.q_queue, (v, q_))


        #--- policies
        if policy in ['greedy_matching', 'accgain_matching']:
            # NOTE: calculate for all questions, including in acc
            # TODO: allow evaluation by accuracy gain too

            MAX_WEIGHT_INT = 100 # can't be too large...

            # build matrix
            worker_indices = sorted(unassigned_workers)
#            question_indices = np.where(q_assigned==0)[0]
            question_indices = range(self.num_questions)
            m = []
            for i,w in enumerate(worker_indices):
                m.append([])
                for q in question_indices:
                    c = (w,q)
                    if c in candidates_all[w]:
                        profit = 20*eval_f(c, acc)
                    else:
#                        print "don't add {}".format(c)
                        profit = 0

                    # if enabled, boost scores for unasked questions
                    # NOTE: assumes profit never exceeds MAX_WEIGHT_INT / 2
                    if self.at_least_once and q_asked[q]==0:
                        profit = profit + MAX_WEIGHT_INT / 2
                        print 'Increased profit for {} ({} to {})'.format(c, profit - MAX_WEIGHT_INT / 2, profit)

                    cost = MAX_WEIGHT_INT - profit
#                    print cost
                    m[i].append(cost)
            
#            print m
#            munkres.print_matrix(m)
            sol_indices = munkres.Munkres().compute(m)
#            print sol_indices
            for i,j in sol_indices:
                w = worker_indices[i]
                q = question_indices[j]
                if (w,q) in candidates_all[w]:
                    acc.append((w,q))
                else:
                    # TODO: double-check
                    q_indices = [_q for w,_q in candidates_all[w]]
                    q_new = min(q_indices, key=lambda q_: m[w][q_])
                    acc.append((w,q_new))
             
        if policy in ['greedy',
                      'accgain',
                      'uncertainty',
                      'random']:
            workers_sorted = list(workers_bad_to_good) 
        elif policy in ['greedy_reverse',
                        'accgain_reverse',
                        'uncertainty_reverse']:
            workers_sorted = list(workers_good_to_bad) 
          

        while len(acc) < self.num_workers:
            # no more votes remain unassigned for some workers, so break

            try:
                itertools.chain.from_iterable(candidates_all.itervalues()).next()
            except StopIteration:
                # remaining_workers = workers_in_acc
                # print 'Remaining workers: {}'.format(sorted(remaining_workers))
                break

            # BUG: assumes uses best known skill (GT or MAP)
            if policy in ['greedy',
                          'accgain',
                          'uncertainty',
                          'random']:
                max_w = workers_bad_to_good.pop()
                next_w = max_w
                # BUG: we have deleted max_w from candidates_all in 'efficient' mode
                candidates = list(candidates_all[max_w])
                # candidates = [(w,q) for w,q in candidates if q_assigned[q] == 0]
            elif policy in ['greedy_reverse',
                            'accgain_reverse',
                            'uncertainty_reverse']:
                min_w = workers_good_to_bad.pop()
                next_w = min_w
                # BUG: we have deleted max_w from candidates_all in 'efficient' mode
                candidates = list(candidates_all[min_w])
                # candidates = [(w,q) for w,q in candidates if q_assigned[q] == 0]
            else:
                candidates = list(itertools.chain.from_iterable(candidates_all.itervalues()))
            #print "candidates: " + str(candidates)
            # TODO: make this a generator
            candidates = [(w,q) for w,q in candidates if
                          q_assigned[q]<self.maxdup] or list(candidates)

            t1 = time.clock()
            if policy in ['greedy', 'greedy_reverse',
                          'accgain', 'accgain_reverse',
                          'greedy_ent']:
                # if enabled, make sure ask once about each question
                restricted_p = self.at_least_once and any(q_asked == 0)
                if restricted_p:
                    candidates = [c for c in candidates if q_asked[c[1]] == 0] or list(candidates)

                # t1 = time.clock()
                # t2 = time.clock()
                # evals = dict((c, eval_f(c, acc)) for c in candidates)
                if restricted_p or not self.policy['efficient']:
                    evals = dict()
                    for c in candidates:
                        # if self.policy['efficient'] and q_assigned[c[1]] == 0:
                        #     evals[w,q] = eval_f_approx(c, eval_metric)
                        # else:
                        evals[w,q] = eval_f(c, acc)


                    top = max(evals, key=lambda k:evals[k])
                    acc.append(top)
                    alternatives.append({'selected': top,
                                         'set': acc[:-1],
                                         'heuristic': evals.copy()})
                # # efficient version
                else:
                    bin_order = self.bin_orders[eval_metric][next_w]
                    # print 'skill bin: {}'.format(skill_bin)

                    next_q = None
                    for b in bin_order:
                        bin_questions = self.bin_to_q[w][b]
                        if len(bin_questions) > 0:
                            next_q = bin_questions.pop()
                            next_bin = b
                            break

    
                    print 'adding {}'.format((next_w, next_q))
                    acc.append((next_w, next_q))

                    for w in xrange(self.num_workers):
                        self.bin_to_q[w][b].discard(next_q)
                
                
                #     #                    print 'queue: {}'.format(len(self.q_queue))
                #     next_w = None
                #     while next_w is None:
                #         try:
                #             _, next_q = heapq.heappop(self.q_queue)
                #             for w in reversed(workers_sorted):
                #                 if (w, next_q) in candidates_all[w]:
                #                     next_w = w
                #                     break
                #
                #             if q_remaining[next_q] > 1:
                #                 self.q_to_add.append(next_q)
                #
                #         # no questions in queue
                #         except IndexError:
                #             raise 'What do we do?'
                #     acc.append((next_w, next_q))
                #     workers_sorted.remove(next_w)


            elif policy == 'local_s':
                # Use local search to select argmin H(U|A)
                pass

            elif policy == 'random':
#                print 'candidates: {}'.format(candidates)
                acc.append(random.choice(candidates))
                
            elif policy == 'same_question':
                q_in_acc = set(q for w,q in acc)
                opts = [(w,q) for w,q in candidates if q in q_in_acc]
                if opts:
                    acc.append(random.choice(opts))
                else:
                    acc.append(random.choice(candidates))
                    
            elif policy == 'rr':
                q_opt = set(q for w,q in candidates)
                max_remaining_votes = max(q_remaining[q] for q in q_opt)

                q_opt = [q for q in q_opt if
                         q_remaining[q] == max_remaining_votes]


                # was selecting whatever max returns
                #q_sel = max(q_opt, key=lambda x: q_remaining[x])

                # random question
                q_sel = random.choice(q_opt)

                # random worker for selected question
                w_sel = random.choice([w for w,q in candidates if q == q_sel])
                acc.append((w_sel,q_sel))
                
            elif policy == 'rr_match':
                assert self.known_difficulty and self.known_skill
                q_opt = set(q for w,q in candidates)
                max_remaining_votes = max(q_remaining[q] for q in q_opt)
                q_opt = [q for q in q_opt if
                         q_remaining[q] == max_remaining_votes]

                # was selecting whatever max returns
                #q_sel = max(q_opt, key=lambda x: q_remaining[x])

                # easiest question
                min_diff = min(self.gt_difficulties[q] for q in q_opt)
                q_sel = random.choice([q for q in q_opt if
                                       self.gt_difficulties[q] == min_diff])

                # least skilled worker
                w_opt = [w for w,q in candidates if q == q_sel]
                min_skill = max(self.gt_skills[w] for w in w_opt)
                w_sel = random.choice([w for w in w_opt if 
                                       self.gt_skills[w] == min_skill])

                acc.append((w_sel,q_sel))
#            elif policy == 'rr_mul':
#                q_remaining = np.sum(self.observations == -1, 0)
#                for w,q in acc:
#                    q_remaining[q] -= 1
#                q_opt = set(q for w,q in candidates)
#                q_sel = max(q_opt, key=lambda x: q_remaining[x])
#                l = [(w,q) for w,q in candidates if q == q_sel]
#                acc.append(min(l, key=lambda k: self.posteriors[k[1]]))

            elif policy == 'uncertainty' or policy == 'uncertainty_reverse':
                assert self.known_difficulty and self.known_skill
                q_in_acc = defaultdict(int)
                for w,q in acc:
                    q_in_acc[q] = 1

                v = min((((w,q),q_in_acc[q],np.abs(self.posteriors[q]-0.5)) for
                         w,q in candidates),
                        key=operator.itemgetter(1,2))
                acc.append(v[0])

            # KG POLICIES
            
            elif policy == 'kg_greedy':
                evals = dict()
                for w,q in candidates:
                    a,b = self.params['kg']['thetas'][q]
                    p = max(betainc(a,b,0.5), 1-betainc(a,b,0.5))
                    
                    new_params = self.new_params(w,q)
                    ((a1,b1),(_,_)) = new_params[1]
                    ((a0,b0),(_,_)) = new_params[0]
                    p1 = max(betainc(a1,b1,0.5), 1-betainc(a1,b1,0.5))                    
                    p0 = max(betainc(a0,b0,0.5), 1-betainc(a0,b0,0.5))

                    evals[w,q] = max(p1-p, p0-p)
                
                        
                top = max(evals, key=lambda k:evals[k])
                acc.append(top)
                alternatives.append({'selected': top,
                                     'set': acc[:-1],
                                     'heuristic': evals.copy()})

            elif policy == 'kg_greedy_unasked':
                evals = dict()
                for w,q in candidates:
                    a,b = self.params['kg']['thetas'][q]
                    p = max(betainc(a,b,0.5), 1-betainc(a,b,0.5))
                    
                    new_params = self.new_params(w,q)
                    ((a1,b1),(_,_)) = new_params[1]
                    ((a0,b0),(_,_)) = new_params[0]
                    p1 = max(betainc(a1,b1,0.5), 1-betainc(a1,b1,0.5))                    
                    p0 = max(betainc(a0,b0,0.5), 1-betainc(a0,b0,0.5))

                    evals[w,q] = max(p1-p, p0-p)
                
                # figure out questions asked
                q_in_acc = defaultdict(int)
                for w,q in acc:
                    q_in_acc[q] = 1

                top,_,_ = min((((w,q),q_in_acc[q],-1*evals[w,q]) for
                               w,q in candidates),
                              key=operator.itemgetter(1,2))
                        
                acc.append(top)
                alternatives.append({'selected': top,
                                     'set': acc[:-1],
                                     'heuristic': evals.copy()})
            
            elif policy == 'kg_uncertainty':
                evals = dict()
                for w,q in candidates:
                    a,b = self.params['kg']['thetas'][q]
                    p = max(betainc(a,b,0.5), 1-betainc(a,b,0.5))
                    evals[w,q] = p
                
                q_in_acc = defaultdict(int)
                for w,q in acc:
                    q_in_acc[q] = 1

                top,_,_ = min((((w,q),q_in_acc[q],evals[w,q]) for
                               w,q in candidates),
                              key=operator.itemgetter(1,2))
                        
                acc.append(top)
                alternatives.append({'selected': top,
                                     'set': acc[:-1],
                                     'heuristic': evals.copy()})

            else:
                raise Exception('Undefined policy')

            t2 = time.clock()
            # bookkeeping
            assigned_w, assigned_q = acc[-1]
            q_asked[assigned_q] += 1
            q_assigned[assigned_q] += 1
            q_remaining[assigned_q] -= 1
            assigned_workers.add(assigned_w)
            unassigned_workers.remove(assigned_w)
            del candidates_all[assigned_w]

            inner_times.append(t2-t1)
            

#        print "Lazy evaluations saved: {}".format(num_skipped)
        print 'sum={:.1f}, u={:.3f}, len={} (inner)'.format(sum(inner_times),
                                                            np.mean(inner_times),
                                                            len(inner_times))
        return acc, alternatives

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
        self.sample = False
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

                # BUG: don't need to do inference for all questions if skill known
                newposts,_ = self.infer(newobs, self.params['reg'])
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

    def ent_gain(self, acc, c):
        """Entropy gain
        
        c:      vote to add
        acc:    list of votes already being asked
        """
        return self.hXA(acc, c) - self.hXU(acc, c)

    def acc_gain(self, acc, x):
        """Exact expected accuracy gain (change to sampling if slow)

        x:      list of votes to add
        acc:    list of votes already being asked
        """
        # NOTE: compute locally for a question since worker skill known

        accgain = 0

        rel_q = set(q for w,q in x)
        for cur_q in rel_q:
            rel_acc = [(w,q) for w,q in acc if q == cur_q]

            rel_x = [(w,q) for w,q in x if q == cur_q]

            for tup in itertools.product(*([0,1] for
                                           x in xrange(len(rel_acc) + \
                                                       len(rel_x)))):
                # NOTE: changed this to log (different from hXA above)
                pA1 = np.log(self.posteriors[cur_q])
                pA0 = np.log(1-self.posteriors[cur_q])

                # NOTE: only adds votes in tup corresponding to acc (not x)
                for ind,v in zip(rel_acc,tup):
                    if v:
                        pA1 += np.log(self.probs[ind])
                        pA0 += np.log(1-self.probs[ind])
                    else:
                        pA1 += np.log(1-self.probs[ind])
                        pA0 += np.log(self.probs[ind])

                # log P(a)
                pA = np.logaddexp(pA1,pA0)
                accuracy_old = max(np.exp(pA1-pA), np.exp(pA0-pA))

                pAX1 = pA1
                pAX0 = pA0

                for ind,v in zip(rel_x,tup[len(rel_acc):]):
                    if v:
                        pAX1 += np.log(self.probs[ind])
                        pAX0 += np.log(1-self.probs[ind])
                    else:
                        pAX1 += np.log(1-self.probs[ind])
                        pAX0 += np.log(self.probs[ind])


                # weight
                pAX = np.logaddexp(pAX0,pAX1)

                new_prob = np.exp(pAX1-pAX)
                accuracy_new = max(new_prob, 1-new_prob)

#            print 'v(before)={}'.format(accgain)
#            print '{} * {}'.format(np.exp(pAX), accuracy_new - accuracy_old)
                accgain += np.exp(pAX) * (accuracy_new - accuracy_old)
#            print 'v(after)={}'.format(accgain)
#        print 'final: {}'.format(accgain)

        assert accgain > -.000001
#        print 'accgain {}: {}'.format(x,accgain)


        return max(accgain,0)



    #------- meta control ------

    def observe(self, votes, time_elapsed=1):
        """Request unassigned votes, retrieve after time_elapsed seconds.
        Update self.observations ... mutation only"""

        votes_out = []
        for v in votes:
            assert self.vote_status[v] != 1
            if self.vote_status[v] is None:
                votes_out.append(v)
                self.vote_status[v] = 0
        
        self.platform.assign(votes_out)
        votes_back = self.platform.get_votes(time_elapsed)
        self.time_elapsed += time_elapsed

        for v in votes_back:
            self.vote_status[v] = 1
            self.observations[v] = votes_back[v]

#        print 'observing votes ' + str(votes)

        return votes_back


    def update_and_score(self, votes, vote_alts=None,
                         votes_assigned=None, timing=None):

        self.update_posteriors(votes)
        
        # Hack -- shouldn't really be here
        if votes:
            q_back = set(q for w,q in votes)
            self.set_bins(q_back)

        n_observed = len(self.get_votes('observed'))
        accuracy, exp_accuracy = self.score()
        
        def keys_to_str(d):
            return dict(('{},{}'.format(*k), d[k]) for k in d)
        
        def dict_to_json(d):
            return [{'worker': k[0],
                     'question': k[1],
                     'vote': d[k]} for k in d]
                     
        def params_to_json(d):
            params = dict()
            if self.params['kg']:
                params['kg'] = \
                    {'thetas': list(self.params['kg']['thetas']),
                     'skills': list(self.params['kg']['skills'])}
                                
            if self.params['reg']:
                params['reg'] = \
                    {'difficulties': self.params['reg']['difficulties'].copy(),
                     'skills': self.params['reg']['skills'].copy()}
            return params.copy()
        
        # count number of times question is assigned at least two workers
        # simultaneously
#        if votes_assigned:
#            counts = defaultdict(int)
#            for w,q in votes_assigned:
#                counts[q] += 1
#            duplicates = len([q for q in counts if counts[q] > 1])
#        else:
#            duplicates = None


        if votes_assigned:
            duplicates = collections.Counter(collections.Counter([q for w,q in votes_assigned]).itervalues())
        else:
            duplicates = None
        
        
        
        workers = set(xrange(self.num_workers))
        workers_remaining = set(w for w,q in self.get_votes('unobserved'))        
        workers_finishing = workers.difference(workers_remaining).difference(
                                                set(self.worker_finished))
        # for w in workers_finishing:
        #     self.worker_finished[w] = {'observed': n_observed,
        #                                'time': self.time_elapsed}
        #                                #'skill': 1/self.gt_skills[w]}

        
        if timing is not None:
            self.hist[-1]['timing'] = timing
        self.hist.append({  't_': self.time_elapsed,
                            'observation': dict_to_json(votes),
                            'state_': {'posterior': self.posteriors.tolist(),
                                       'params': params_to_json(self.params)},
                            'other': {'observed': n_observed,
                                      'accuracy': accuracy,
                                      'exp_accuracy': exp_accuracy,
                                      'duplicates': duplicates,
                                      'workers_finishing': list(workers_finishing)}
                                      })
        
                                  # 'votes': keys_to_str(votes)})

                          
#        if vote_alts:
#            # NOTE: modifying vote_alts (side effects)
#            for d in vote_alts:
#                d['heuristic'] = keys_to_str(d['heuristic'])
#                d['selected'] = '{},{}'.format(*d['selected'])
#                d['set'] = ['{},{}'.format(*t) for t in d['set']]
#            self.hist_detailed[-1]['alternatives'] = vote_alts
        



    def score(self, metric='accuracy'):
        """Return average score for posterior predictions"""

        accuracy = np.mean(np.round(self.posteriors) == self.gt_labels)
        exp_accuracy = np.mean([max(x, 1-x) for x in self.posteriors])

        if metric == 'accuracy':
            return accuracy, exp_accuracy
        else:
            raise Exception('Undefined score metric')

    def update_posteriors(self, votes):
        #--- KG update
        if votes:
            for (w,q),v in votes.iteritems():
                self.update_params_kg(w, q, v)
            
        if self.post_inf == 'kg':
            #--- KG posterior mean of betas
            self.posteriors = np.array([a/(a+b) for (a,b) in self.params['kg']['thetas']])
        elif self.post_inf == 'kg_lbfgs':
            params, posteriors = self.optimize_kg()
            self.params['kg_lbfgs'] = params
            self.posteriors = posteriors
        elif self.post_inf == 'mv':
            # predict with MV
            n1 = np.sum(self.observations == 1, axis=0)
            n0 = np.sum(self.observations == 0, axis=0)
            self.posteriors =  (n1 + .01) / (n1 + n0 + .02)
        elif self.post_inf == 'reg':
            params, posteriors = self.run_em()
            self.params['reg'] = params
            self.posteriors = posteriors


            def mode_or_undef(a,b):
                if a>1 and b>1:
                    return (a-1)/(a+b-2)
                else:
                    return 'Undefined'
            # print 'kg skills: ',[mode_or_undef(a,b) for (a,b) in self.params['kg']['skills']]
            # print 'kg thetas: ',[a/(a+b) for (a,b) in self.params['kg']['thetas']]
            # print 'reg skills: ',self.params['reg']['skills']
            # print 'reg diffs: ',self.params['reg']['difficulties']
            # print 'reg post: ',self.posteriors
            

            # MAP estimate
            self.probs = self.allprobs(self.params['reg']['skills'],
                                       self.params['reg']['difficulties'])
        else:
            raise Exception('Undefined posterior inference method')


                   
    def get_results(self):
        return self.hist
        # return {"hist": self.hist,
        #         "hist_detailed": self.hist_detailed,
        #         "when_finished": self.worker_finished}



    def run_offline(self):
        assert self.known_difficulty
        assert self.known_skill
        policy = self.method
        self.reset()

        print 'RUNNING OFFLINE'
        print 'Policy: {0}'.format(policy)

        self.update_and_score(votes=[])

        for depth in xrange(self.num_questions):
            self.reset_offline()
            self.platform.reset()
            next_votes = self.select_votes_offline(depth+1)

            # make observations and update
            votes_back = self.observe(next_votes, float('inf')) # BUG: untested
#            print [len([v for v in next_votes if v[0]==w]) for
#                   w in xrange(self.num_workers)]
#            print np.sum(self.observations != -1)
            self.update_and_score(votes=votes_back)

        return self.get_results()


    def run(self):
        policy = self.method
        print 'Policy: {0}'.format(policy)
        self.reset()

        self.update_and_score(votes=[])


        rounds = 0
        while len(self.get_votes('unobserved')) > 0 and rounds < self.max_rounds:
            print 'Remaining votes: {0:4d}'.format(
                    len(self.get_votes('unobserved')))

            #            if not self.known_difficulty:
            #    #                print self.params['difficulties']
            #    print np.mean(np.abs(self.params['difficulties'] - \
                    #                         self.platform.gt_difficulties))

            # select votes
            t1 = time.clock()
            next_votes, alternatives = self.select_votes()
            t2 = time.clock()
            print '{:.1f} secs'.format(t2-t1)

            # make observations and update
            votes_back = self.observe(next_votes)
            if votes_back:
                self.update_and_score(votes=votes_back,
                                      vote_alts=alternatives,
                                      votes_assigned = next_votes,
                                      timing=t2-t1)

            rounds += 1
            


#        print "**************"
#        print "RESULTS for policy " + policy
#        print "accuracies: " + str(self.accuracies)
#        print "**************"
#        print

        return self.get_results()



    


