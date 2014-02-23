"""platform.py
simulator for worker responses"""

from __future__ import division
import collections
import itertools
import numpy as np
from ut import sample_dist

ODESK_SKILL_DIST = {'type': 'normal_pos', 'params': [0.7936, 0.2861]}


class Platform():
    def __init__(self, gt_labels, 
                 votes=None, num_workers=None,
                 skills=None, difficulties=None):

        self.gt_labels = gt_labels
        self.num_questions = len(gt_labels)

        if votes is not None:
            self.num_workers = votes.shape[0]
            self.votes = votes
            self.gt_skills = skills
            self.gt_difficulties = difficulties
        elif num_workers is not None:
            skills = skills or ODESK_SKILL_DIST
            self.num_workers = num_workers
            self.gt_skills, self.gt_difficulties = self.get_params(
                                                                skills,
                                                                difficulties)
            self.votes = self.make_votes(self.gt_skills, self.gt_difficulties)
        else:
            raise Exception('Must specify number of workers or provide votes')

        self.response_times = self.make_times()
        self.reset()

    #-------- initialization -----------

    def reset(self):
        self.remaining_time = dict()

    def get_params(self, skills, difficulties):
        """Either use supplied parameters, or generate from supplied
        distribution."""

        if skills is None or difficulties is None:
            raise Exception('Must specify skills and difficulties')

        # get skills
        if isinstance(skills, collections.Mapping):
            lst = []
            while len(lst) < self.num_workers:
                v = sample_dist(skills)
                lst.append(v)
            skills = np.array(lst)
        else:
            # assume skills are given
            assert len(skills) == self.num_workers

        # get difficulties
        if isinstance(difficulties, collections.Mapping):
            lst = []
            while len(lst) < self.num_questions:
                v = sample_dist(difficulties)
                lst.append(v)
            difficulties = np.array(lst)
        else:
            # assume difficulties are given
            assert len(difficulties) == self.num_questions

        return skills, difficulties


    def prob_correct(self, s, d):
        """p_correct = 1/2(1+(1-d_q)^(1/skill)"""
        return 1/2*(1+np.power(1-d,s))


    def allprobs(self, skills, difficulties):
        """Return |workers| x |questions| matrix with prob of worker answering
        correctly.
        """
        return self.prob_correct(skills[:, np.newaxis],
                                 difficulties)

    def make_votes(self, skills, difficulties):
        """Generate worker votes. Unvectorized, but shouldn't matter."""
        probs = self.allprobs(skills, difficulties)

        o = np.zeros((self.num_workers, self.num_questions))
        for w,q in itertools.product(xrange(self.num_workers),
                                     xrange(self.num_questions)):
            pCorrect = np.random.random() <= probs[w,q]
            if pCorrect:
                v = self.gt_labels[q]
            else:
                v = not self.gt_labels[q]

            o[w,q] = int(v)  # store truth val as int for consistency

        return o

    def make_times(self):
        """Time to complete a vote. Uniform for now."""
        times = dict()
        for w,q in itertools.product(xrange(self.num_workers),
                                     xrange(self.num_questions)):
            times[w,q] = 1
        return times

    #-------------- access methods ------------
    def assign(self, votes):
        """Votes is list of (worker, question) pairs"""
        # BUG: no error checking to ensure not already asked
        for v in votes:
            self.remaining_time[v] = self.response_times[v]


    def get_votes(self, time_elapsed=1):
        """Returns dictionary of {(worker, question): vote}"""
        # NOTE: doesn't check to ensure votes haven't been asked before

        votes = dict() 

        new_times = dict((v, self.remaining_time[v] - time_elapsed) for
                         v in self.remaining_time)

        self.remaining_time = dict((v,new_times[t]) for
                                   v in new_times if new_times[v]>0)
        votes = dict((v,self.votes[v]) for v in new_times if new_times[v]<=0)

        return votes

