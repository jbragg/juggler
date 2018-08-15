# extra code to generate answers for camera-ready
#
#
from __future__ import division
import os
import itertools
from prob import ProbabilityDai
import numpy as np
from collections import defaultdict
import matplotlib as mpl
from matplotlib import pyplot as plt

class PlotController():
    def __init__(self, skills, difficulties, posteriors):
        self.sample = False
        
        # make questions for plot
        self.gt_difficulties = []
        self.posteriors = []        
        
        self.difficulty_steps = difficulties
        self.posterior_steps = posteriors
        for i in self.difficulty_steps:
            for j in self.posterior_steps:
                self.gt_difficulties.append(i)
                self.posteriors.append(j)
        
        
        # make workers for plot
        self.gt_skills = skills
        
        self.num_questions = len(self.gt_difficulties)
        self.num_workers = len(self.gt_skills)
        
        # make probabilities
        self.prob_module = ProbabilityDai()
        self.gt_difficulties = np.array(self.gt_difficulties)
        self.gt_skills = np.array(self.gt_skills)
        self.probs = self.prob_module.allprobs(self.gt_skills,
                                               self.gt_difficulties)

    
    def get_bin_values(self, method):
        values = defaultdict(list)
        for w in xrange(self.num_workers):
            print 'worker {}'.format(w)
            for q in xrange(self.num_questions):
                if method=='entropy':
                    v = self.ent_gain([], (w,q))
                elif method=='accuracy':
                    v = self.acc_gain([], [(w,q)])
                values[w].append(v)

            values[w] = np.reshape(values[w], (len(self.difficulty_steps),
                                               len(self.posterior_steps)))
        return values
        
    def make_plots(self, method='entropy'):
        values = self.get_bin_values(method)

        y_labels = ['{:.2f}'.format(s) for s in self.difficulty_steps]
        x_labels = ['{:.2f}'.format(s) for s in self.posterior_steps]
        print y_labels
        for w in xrange(self.num_workers):
            intensities =  values[w]

            print intensities
            fig = plt.figure()
            ax = fig.add_subplot(111)
            cax = ax.matshow(intensities,
                             interpolation='nearest',
                             # cmap=plt.cm.get_cmap('PuBu'))
                             cmap=plt.cm.Blues,
                             extent=(0,1,1,0)) # hack
            fig.colorbar(cax)
            ax.axis('image')

            # ax.set_xticklabels([''] + x_labels)
            # ax.set_yticklabels([''] + y_labels)
            plt.xlabel('Posterior axis')
            plt.ylabel('Difficulty')
            
            
            plt.title('Value of asking questions ({}-gain) for worker (skill={:.2f})'.format(method, self.gt_skills[w]))

            # plt.pcolor(intensities, cmap=plt.cm.Blues)
            filepath = os.path.join('img','heatmap-{}-{:02d}.png'.format(method,w))
            plt.savefig(filepath, dpi=250)
            plt.close('all')
    
    #---------------- duplicate code from control.py
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
        
    
if __name__ == '__main__':
    num_steps=21
    worker_skills = np.array([0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6])
    
    # this step is necessary because older code uses 1/skill, whereas
    # the prob module uses skill
    worker_skills = 1/worker_skills
    
    difficulties = np.linspace(0.001, .999, num_steps)
    posteriors = np.linspace(0.001, .999, num_steps)
    c = PlotController(skills=worker_skills,
                       difficulties=difficulties,
                       posteriors=posteriors)
    c.make_plots('entropy')
    c.make_plots('accuracy')
