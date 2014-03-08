"""control.py
controllers"""

from __future__ import division
import numpy as np
import itertools
from collections import defaultdict
from scipy.misc import logsumexp
import scipy.optimize
import scipy.stats
import random
import pickle
import heapq
import sys
import csv
from ut import dbeta
import json
import os




class Controller():
    def __init__(self, method,
                 platform,
                 num_questions, num_workers,
                 known_s=True, known_d=True,
                 sample=True):

        # method can be 'random', 'rr', 'greedy', 'greedy_reverse"
        self.method = method  
        self.platform = platform

        self.prior = (2,2) # prior for diff (was prior for diff and label)
        self.sample = sample

        self.known_difficulty = known_d
        self.known_skill = known_s
        self.gt_labels = platform.gt_labels

        if known_s:
            self.gt_skills = platform.gt_skills
            assert self.gt_skills is not None
        else:
            self.gt_skills = None

        if known_d:
            self.gt_difficulties = platform.gt_difficulties
            assert self.gt_difficulties is not None
        else:
            self.gt_difficulties = None


        self.num_questions = num_questions
        self.num_workers = num_workers


        
        # other initialization
        self.reset()
#        self.posteriors = self.prior[0] / (self.prior[0] + self.prior[1]) * \
#                          np.ones(num_questions)
#        self.posteriors = 0.5 * np.ones(self.num_questions)

        self.params = None # contains MAP label posteriors, difficulties



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



    def init_observations(self):
        """observations is |workers| x |questions| matrix
           -1 - unobserved
           1 - True
           0 - False

        """
        self.observations = np.zeros((self.num_workers, self.num_questions))-1
        return

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

       
        def M(posteriors):
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
                init_s = 0.9 * np.ones(self.num_workers)
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

        ptrue = np.log(priors) + np.sum(np.log(probs) * true_votes, 0) + \
                                 np.sum(np.log(1-probs) * false_votes, 0)
        pfalse = np.log(1-priors) + np.sum(np.log(probs) * false_votes, 0) + \
                                    np.sum(np.log(1-probs) * true_votes, 0)


        norm = np.logaddexp(ptrue, pfalse)

        return np.exp(ptrue) / np.exp(norm), np.sum(norm)


    #----------- control -----------

    def select_votes_offline(self, depth):
        assert self.known_difficulty and self.known_skill
        assert depth > 0 and depth <= self.num_questions

        policy = self.method

        if policy == 'greedy':
            w_order = sorted(xrange(self.num_workers),
                             key=lambda i:self.params['skills'][i])
        elif policy == 'greedy_reverse':
            w_order = sorted(xrange(self.num_workers),
                             key=lambda i:-self.params['skills'][i])
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

        while len(acc) < self.num_workers:
            evals = {}
            workers_in_acc = [w for w,q in acc]
            candidates = [(w,q) for w,q in unassigned_votes if
                          w not in workers_in_acc]

            # no more votes remain unassigned for any workers, so break
            if not candidates:
                remaining_workers = workers_in_acc
                print 'Remaining workers: {}'.format(sorted(remaining_workers))
                break


            # BUG: assumes uses best known skill (GT or MAP)
            if policy == 'greedy' or policy == 'accgain':
                max_w,_ = min(candidates,
                              key=lambda x: self.params['skills'][x[0]])
                candidates = [c for c in candidates if c[0] == max_w]
            elif policy == 'greedy_reverse' or policy == 'accgain_reverse':
                min_w,_ = max(candidates,
                              key=lambda x: self.params['skills'][x[0]])
                candidates = [c for c in candidates if c[0] == min_w]
            #print "candidates: " + str(candidates)

            if policy == 'greedy' or policy == 'greedy_reverse':
                for c in candidates:
                    evals[c] = self.hXA(acc, c) - self.hXU(acc, c)
                        
                acc.append(max(evals, key=lambda k:evals[k]))

            elif policy == 'accgain' or policy == 'accgain_reverse':
                for c in candidates:
                    evals[c] = self.acc_gain(acc, c)

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
                q_remaining = np.sum(self.observations == -1, 0)
                for w,q in acc:
                    q_remaining[q] -= 1
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
                q_remaining = np.sum(self.observations == -1, 0)
                for w,q in acc:
                    q_remaining[q] -= 1
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

    def acc_gain(self, acc, x):
        """Exact expected accuracy gain (change to sampling if slow)"""
        # NOTE: compute locally for a question since worker skill known
        accgain = 0

        rel_q = x[1]
        rel_acc = [(w,q) for w,q in acc if q == rel_q]


        for tup in itertools.product(*([0,1] for x in xrange(len(rel_acc)+1))):
            # NOTE: changed this to log (different from hXA above)
            pA1 = np.log(self.posteriors[rel_q])
            pA0 = np.log(1-self.posteriors[rel_q])

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

            if tup[-1]:
                pAX1 = pA1 + np.log(self.probs[x])
                pAX0 = pA0 + np.log(1-self.probs[x])
            else:
                pAX0 = pA0 + np.log(self.probs[x])
                pAX1 = pA1 + np.log(1-self.probs[x])

            # weight
            pAX = np.logaddexp(pAX0,pAX1)

            new_prob = np.exp(pAX1 - pAX)
            accuracy_new = max(new_prob, 1-new_prob)

#            print 'v(before)={}'.format(accgain)
#            print '{} * {}'.format(np.exp(pAX), accuracy_new - accuracy_old)
            accgain += np.exp(pAX) * (accuracy_new - accuracy_old)
#            print 'v(after)={}'.format(accgain)
#        print 'final: {}'.format(accgain)

        return accgain



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


    def update_and_score(self):
        self.update_posteriors()

        n_observed = len(self.get_votes('observed'))
        score = self.score()

        self.accuracies.append({'observed': n_observed,
                                'time': self.time_elapsed,
                                'score': score})


    def score(self, metric='accuracy'):
        """Return average score for posterior predictions"""

        correct = np.round(self.posteriors) == self.gt_labels

        if metric == 'accuracy':
            return np.sum(correct) / len(correct)
        else:
            raise Exception('Undefined score metric')

    def update_posteriors(self):
        params, posteriors = self.run_em()
        self.params = params
        self.posteriors = posteriors


        self.probs = self.allprobs(self.params['skills'],
                                   self.params['difficulties'])
                   

    def run_offline(self):
        assert self.known_difficulty
        assert self.known_skill
        policy = self.method
        self.reset()

        print 'RUNNING OFFLINE'
        print 'Policy: {0}'.format(policy)

        self.accuracies = []
        self.update_and_score()

        for depth in xrange(self.num_questions):
            self.reset()
            votes = self.select_votes_offline(depth+1)

            # make observations and update
            self.observe(votes, float('inf'))  # BUG: untested
#            print [len([v for v in votes if v[0]==w]) for
#                   w in xrange(self.num_workers)]
#            print np.sum(self.observations != -1)
            self.update_and_score()

        return {'accuracies': self.accuracies}


    def run(self):
        policy = self.method
        print 'Policy: {0}'.format(policy)
        self.reset()
        self.accuracies = []
        posteriors = []
        votes = []

        votes.append([])
        self.update_and_score()
        posteriors.append(self.posteriors)



        while len(self.get_votes('unobserved')) > 0:
            print 'Remaining votes: {0:4d}'.format(
                    len(self.get_votes('unobserved')))

            # select votes
            next_votes = self.select_votes()

            # make observations and update
            votes_back = self.observe(next_votes)
            votes.append(votes_back)
            if votes_back:
                self.update_and_score()
                posteriors.append(self.posteriors)
            


#        print "**************"
#        print "RESULTS for policy " + policy
#        print "accuracies: " + str(self.accuracies)
#        print "**************"
#        print

        return {"votes": votes,
                "belief": posteriors,
                "accuracies": self.accuracies,
                "gt_difficulties": self.gt_difficulties,
                "gt_skills": self.gt_skills}



