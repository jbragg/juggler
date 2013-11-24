#!/bin/env python

from __future__ import division
import numpy as np
import itertools
from scipy.misc import logsumexp
import scipy.optimize
import scipy.stats
import random

NUM_WORKERS = 2
NUM_QUESTIONS = 10



class ExpState(object):
    def __init__(self, num_questions, num_workers, known_difficulty):
        self.num_questions = num_questions
        self.num_workers = num_workers
        self.known_difficulty = known_difficulty
        self.prior = (2,2) # assume Beta(2,2) prior
        self.sample = True
        self.reset()

        self.posteriors = self.prior[0] / (self.prior[0] + self.prior[1]) * \
                          np.ones(num_questions)

        self.params = None # contains MAP label posteriors, difficulties

        # generate random ground truth (random for now)
        self.gt_labels = np.round(self.gen_labels())
        self.gt_difficulties = self.gen_question_difficulties()
        self.gt_skills = self.gen_worker_skills()
        self.gt_probs = self.allprobs(self.gt_skills, self.gt_difficulties)
        self.gt_observations = self.gen_observations()


    ############ initialization methods ##############
    def reset(self):
        """reset for run of new policy"""
        self.remaining_votes = dict([((i,j),True) for i,j in itertools.product(
                                                xrange(self.num_workers),
                                                xrange(self.num_questions))])
        self.votes = []
        self.init_observations()

    def gen_labels(self):
        """generate labels from same distribution as prior"""
        return np.random.beta(*self.prior,size=self.num_questions)

    def gen_worker_skills(self):
        """Should be in range [0,inf)

        """

        # BUG: workers distributed according to Beta(2,2)
        return np.random.beta(2,2,self.num_workers)

    def gen_question_difficulties(self):
        """Should be in range [0,1]
        p_correct = 1/2(1+(1-d_q)^(1/skill)

        """

        # BUG: all problems equal difficulty
        return np.ones(self.num_questions) / 2

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
        return 1/2*(1+np.power(1-d,1/s))

    def prob_correct_ddifficulty(self, s, d):
        return -1/2*1/s*np.power(1-d,1/s-1)

    def allprobs(self, skills, difficulties):
        """Return |workers| x |questions| matrix with prob of worker answering
        correctly.
        """
        return self.prob_correct(skills[:, np.newaxis], difficulties)

    def allprobs_ddifficulty(self, skills, difficulties):
        return self.prob_correct_ddifficulty(skills[:, np.newaxis],
                                             difficulties)

    def init_observations(self):
        """observations is |workers| x |questions| matrix
           0 - unobserved
           1 - True
           2 - False

        """
        self.observations = np.zeros((self.num_workers, self.num_questions))
        return
    
    def rand_observations(self):
        return np.random.randint(0,3,(self.num_workers, self.num_questions))

    ######### meta methods #########


    def run(self, policy):
        self.reset()
        self.accuracies = []
        self.update_posteriors()
        self.accuracies.append(self.score())
        while len(self.remaining_votes_list()) > 0:
            # select votes
            next_votes = self.select_votes(policy)

            # make observations and update
            self.observe(next_votes)
            self.update_posteriors()
            self.accuracies.append(self.score())
        
        print
        print "**************"
        print "RESULTS for policy " + policy
        print "accuracies: " + str(self.accuracies)
        print "**************"
        print

    def select_votes(self, policy):
        acc = []
        while len(acc) < min(self.num_workers, self.remaining_votes_list()):
            evals = {}
            workers_in_acc = [x[0] for x in acc]
            candidates = [i for i in self.remaining_votes if 
                          self.remaining_votes[i] and
                          i[0] not in workers_in_acc]
            #print "candidates: " + str(candidates)
            if policy == 'greedy':
                for c in candidates:
                    evals[c] = self.heur(acc, c)

                acc.append(max(evals, key=lambda k:evals[k]))
            elif policy == 'random':
                acc.append(random.choice(candidates))
            else:
                raise Exception('Undefined policy')

        return acc

    # compute H(X | U)
    def heur(self, acc, x):
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


        # compute P(X | A)
        # sample
        if self.sample:
            hXA = 0
            epsilon = 0.1
            delta = 0.05
            N = int(np.ceil(0.5*np.power(np.log(2)/epsilon,2)*np.log(1/delta)))
            for i in xrange(N):
                newobs = np.copy(self.observations)

                # sample
                for w,q in acc:
                    # sample true labels using poster dist
                    # BUG : is this valid? use rejection sampling instead?

                    # return same or different answer as true value
                    if np.random.random() <= self.probs[w,q]:
                        v = np.random.random() <= self.posteriors[q]
                    else:
                        v = np.random.random() > self.posteriors[q]

                    if v:
                        newobs[w,q] = 1
                    else:
                        newobs[w,q] = 2

                newposts,_ = self.infer(newobs, self.params)
                pL = newposts[x[1]]
                pCorrect = self.probs[x]
                p = np.log(np.array([
                                [pL * pCorrect, pL * (1-pCorrect)],
                                [(1-pL) * (1-pCorrect), (1-pL) * pCorrect]]))
                p = np.exp(logsumexp(p,0))
                def entropy(a):
                    return np.sum(p * np.log(p))
                hXA += entropy(p) / N
            #print str(x) + ":" + str((hXA, epsilon))
        else:
            # TODO: Implement exact conditional entropy
            pass
              
        return hXA - hXU

    ###### inference methods #######
    
    def update_posteriors(self):
        """ keep for legacy reasons """
        self.run_em()
        self.probs = self.allprobs(self.gt_skills,
                                   self.params['difficulties'])
                   
       
    def run_em(self):
        """ learn params and sets self.posteriors """
        def E(params):
            if self.known_difficulty:
                post, ll = self.infer(self.observations,
                                      {'label': params['label'],
                                       'difficulties': self.gt_difficulties})

            else:
                post, ll = self.infer(self.observations, params)

            # add beta prior
            ll += np.sum(np.log(scipy.stats.beta.pdf(params['label'],
                                                         self.prior[0],
                                                         self.prior[1])))

            return post, ll / self.num_questions


        if self.known_difficulty:
            def M(posteriors):
                params = dict()
                params['difficulties'] = self.gt_difficulties
                params['label'] = (self.prior[0] - 1 + sum(posteriors)) / \
                                  (self.prior[0] - 1 + self.prior[1] - 1 + \
                                   self.num_questions)

                return params

        else:
            def M(posteriors):
                params_array = np.append(params['difficulties'],
                                         params['label'])

                def f(params_array):
                    difficulties = params_array[0:self.num_questions]
                    prior = params_array[self.num_questions]
                    probs = self.allprobs(self.gt_skills, difficulties)
                    probs_dd = self.allprobs_ddifficulty(self.gt_skills,
                                                         difficulties)
                    priors = prior * np.ones(self.num_questions)

                    true_votes = (self.observations == 1)   
                    false_votes = (self.observations == 2)   

                    ptrue = np.log(priors) + \
                            np.sum(np.log(probs) * true_votes, 0) + \
                            np.sum(np.log(1-probs) * false_votes, 0)
                    pfalse = np.log(1-priors) + \
                             np.sum(np.log(probs) * false_votes, 0) + \
                             np.sum(np.log(1-probs) * true_votes, 0)

                    ptrue_dd = \
                            np.sum(1/probs*probs_dd * true_votes, 0) + \
                            np.sum(1/(1-probs)*(-probs_dd) * false_votes, 0)

                    pfalse_dd = \
                            np.sum(1/(probs)*probs_dd * false_votes, 0) + \
                            np.sum(1/(1-probs)*(-probs_dd) * true_votes, 0)

                    # result
                    v = np.sum(posteriors * ptrue + (1-posteriors) * pfalse)
                    dd = np.array(posteriors * ptrue_dd + \
                                  (1-posteriors) * pfalse_dd)

                    dd = np.append(dd, np.sum(posteriors * 1/priors + \
                                              (1-posteriors) * 1/(1-priors)))

                    print '---'
                    print params_array
                    print -v
#                print dd
                    print '---'

                    # return negative to minimizer
#                return (-v,
#                       -dd)
                    return -v


                res = scipy.optimize.minimize(
                                        f,
                                        0.5 * np.ones(self.num_questions + 1),
                                        method='L-BFGS-B',
                                        jac=False,
                                        bounds=
                                            [(0,1) for 
                                             i in xrange(self.num_questions + 1)],
                                        options={'disp':True})

                return {'label': res.x[self.num_questions],
                        'difficulties': res.x[0:self.num_questions]}

        ll = float('-inf')
        ll_change = float('inf')
        params = {'difficulties':np.random.random(self.num_questions),
                  'label':np.random.random()}
        em_round = 0
        while ll_change > 0.01:  # run while ll increase is at least 1%
            # print 'EM round: ' + str(em_round)
            posteriors,ll_new = E(params)
            params = M(posteriors)

            if ll == float('-inf'):
                ll_change = float('inf')
            else:
                ll_change = (ll_new - ll) / np.abs(ll) # percent increase

            ll = ll_new
            # print 'll_change: ' + str(ll_change)
            # print 'log likelihood: ' + str(ll)
            assert ll_change > -0.001  # ensure ll is nondecreasing
            em_round += 1

        print str(em_round) + " EM rounds"
        self.params = params
        self.posteriors = posteriors


    def infer(self, observations, params):
        """Probabilistic inference for question posteriors, having observed
        observations matrix. 
        """

        prior = params['label']
        probs = self.allprobs(self.gt_skills, params['difficulties'])
        priors = prior * np.ones(self.num_questions)
         
        true_votes = (observations == 1)   
        false_votes = (observations == 2)   

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
            if self.gt_observations[w,q]:
                self.observations[w,q] = 1
            else:
                self.observations[w,q] = 2

            # update remaining / observed
            self.remaining_votes[w,q] = False
            self.votes.append((w,q))

        print 'observing votes ' + str(votes)

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




new_state = ExpState(NUM_QUESTIONS, NUM_WORKERS, True)
new_state.run('greedy')
new_state.run('random')
#new_state.update_posteriors()
#print new_state.observations
#print new_state.params
#new_state.infer(new_state.observations, new_state.params)
