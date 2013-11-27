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

NUM_WORKERS = 10
NUM_QUESTIONS = 20
NUM_EXPS = 80



class ExpState(object):
    def __init__(self, num_questions, num_workers, known_difficulty):
        self.num_questions = num_questions
        self.num_workers = num_workers
        self.known_difficulty = known_difficulty
        self.prior = (2,2) # assume Beta(2,2) prior
        self.sample = False
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

        # BUG: workers distributed according to Beta(2,20)
        #  --- small gammma corresponds to good workers
        return np.random.beta(2,4,self.num_workers) + 1

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
        return 1/2*(1+np.power(1-d,s))

    def prob_correct_ddifficulty(self, s, d):
        return -1/2*s*np.power(1-d,s-1)

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
        posteriors = []
        votes = []

        def update():
            self.update_posteriors()
            self.accuracies.append(self.score())
            posteriors.append(self.posteriors)

        votes.append([])
        update()

        while len(self.remaining_votes_list()) > 0:
            # select votes
            next_votes = self.select_votes(policy)

            # make observations and update
            self.observe(next_votes)
            votes.append(next_votes)
            update()
        
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
        while len(acc) < min(self.num_workers, self.remaining_votes_list()):
            evals = {}
            workers_in_acc = [x[0] for x in acc]
            candidates = [i for i in self.remaining_votes if 
                          self.remaining_votes[i] and
                          i[0] not in workers_in_acc]
            #print "candidates: " + str(candidates)
            if policy == 'greedy':
                for c in candidates:
                    evals[c] = self.hXA(acc, c) - self.hXU(acc, c)

                acc.append(max(evals, key=lambda k:evals[k]))
            elif policy == 'greedy_ent':
                for c in candidates:
                    evals[c] = self.hXA(acc, c)

                acc.append(max(evals, key=lambda k:evals[k]))
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
                q_remaining = np.sum(self.observations == 0, 0)
                for w,q in acc:
                    q_remaining[q] -= 1
                q_opt = set(q for w,q in candidates)
                q_sel = max(q_opt, key=lambda x: q_remaining[x])
                l = [(w,q) for w,q in candidates if q == q_sel]
                acc.append(random.choice(l))
            elif policy == 'rr_mul':
                q_remaining = np.sum(self.observations == 0, 0)
                for w,q in acc:
                    q_remaining[q] -= 1
                q_opt = set(q for w,q in candidates)
                q_sel = max(q_opt, key=lambda x: q_remaining[x])
                l = [(w,q) for w,q in candidates if q == q_sel]
                acc.append(min(l, key=lambda k: self.posteriors[k[1]]))
            else:
                raise Exception('Undefined policy')

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

            # add beta prior for label parameter
            ll += np.sum(np.log(scipy.stats.beta.pdf(params['label'],
                                                 self.prior[0],
                                                 self.prior[1])))


            return post, ll / self.num_questions


        def M(posteriors):
            if self.known_difficulty:
                params = dict()
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
                    difficulties = params_array[0:self.num_questions]
                    probs = self.allprobs(
                                        self.gt_skills,
                                        difficulties)
                    probs_dd = self.allprobs_ddifficulty(
                                        self.gt_skills,
                                        difficulties)
#                    priors = prior * np.ones(self.num_questions)

                    true_votes = (self.observations == 1)   
                    false_votes = (self.observations == 2)   

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
                            np.sum(1/(probs)*probs_dd * false_votes, 0) + \
                            np.sum(1/(1-probs)*(-probs_dd) * true_votes, 0)

                    # result
                    v = np.sum(posteriors * ptrue + (1-posteriors) * pfalse)
                    dd = np.array(posteriors * ptrue_dd + \
                                  (1-posteriors) * pfalse_dd)

                    #dd = np.append(dd, np.sum(posteriors * 1/priors + \
                    #                          (1-posteriors) * -1/(1-priors)))

#                    print '---'
#                    print params_array
#                    print -v
#                print dd
#                    print '---'

                    # return negative to minimizer
                    return (-v,
                            -dd)
#                    return -v


                res = scipy.optimize.minimize(
                            f,
                            0.3 * np.ones(self.num_questions),
                            method='L-BFGS-B',
                            jac=True,
                            bounds= [(0,1) for 
                                     i in xrange(self.num_questions)],
                            options={'disp':True})

                params['difficulties'] = res.x
                return params
#                return {'label': res.x[self.num_questions],
#                        'difficulties': res.x[0:self.num_questions]}

        ll = float('-inf')
        ll_change = float('inf')
        params = {'difficulties':np.random.random(self.num_questions),
                  'label':np.random.random()}
        em_round = 0
        while ll_change > 0.001:  # run while ll increase is at least .1%
            # print 'EM round: ' + str(em_round)
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
            #assert ll_change > -0.001  # ensure ll is nondecreasing
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



policies = ['random','greedy','greedy_ent','rr','rr_mul']


if __name__ == '__main__':
    accs = dict()
    res = dict()
    for i in xrange(NUM_EXPS):
        rint = random.randint(0,sys.maxint)
        print '------------'
        print 'iteration: ' + str(i)
        np.random.seed(rint)
        new_state = ExpState(NUM_QUESTIONS, NUM_WORKERS, True)
        for p in policies:
            np.random.seed(rint)
            r = new_state.run(p)
            if i == 0:
                accs[p] = r['accuracies']
            else:
                accs[p] = np.vstack([accs[p], r['accuracies']])

            res[p,i] = r




    mean = dict()
    stderr = dict()
    for p in policies:
        assert accs[p].shape[1] == NUM_QUESTIONS + 1
        mean[p] = np.mean(accs[p],0)
        stderr[p] = 1.96 * np.std(accs[p],0) / np.sqrt(accs[p].shape[0])
        print
        print p + ':'
        print np.mean(accs[p],0)


#new_state.update_posteriors()
#print new_state.observations
#print new_state.params
#new_state.infer(new_state.observations, new_state.params)

    for p in policies:
        plt.errorbar(xrange(len(mean[p])), mean[p], yerr=stderr[p], label=p)

    plt.ylim(ymax=1)
    plt.legend(loc="lower right")
    plt.savefig('res.png')


    pickle.dump(Result(res), open(str(datetime.datetime.now()) + '.txt','w'))
