"""platform.py
simulator for worker responses"""

from __future__ import division
import collections
import itertools
import numpy as np
from ut import sample_dist

ODESK_SKILL_DIST = {'type': 'normal_pos', 'params': [0.7936, 0.2861]}

# NOTE: commented to assume instant workers
# params = shape, loc, scale
#ODESK_TIME_LOGNORM = {'type': 'lognorm',
#                      'params': [1.12177053427, 0, 37.6586432961]}
BETA_DIFFS = {'type': 'beta',
              'params': [1,1]}

class Platform():
    def __init__(self, gt_labels, 
                 votes=None, num_workers=None,
                 skills=None, difficulties=None,
                 times=None):

        self.gt_labels = gt_labels
        self.num_questions = len(gt_labels)

        if votes is not None:
            self.num_workers = votes.shape[0]
            self.votes = votes
            self.gt_skills = skills
            self.gt_difficulties = difficulties
        elif num_workers is not None:
            if not isinstance(skills, np.ndarray):
                skills = skills or ODESK_SKILL_DIST
            if not isinstance(difficulties, np.ndarray):
                difficulties = difficulties or BETA_DIFFS
            self.num_workers = num_workers
            self.gt_skills, self.gt_difficulties = self.get_params(
                                                                skills,
                                                                difficulties)
            self.votes = self.make_votes(self.gt_skills, self.gt_difficulties)
        else:
            raise Exception('Must specify number of workers or provide votes')

        # times = times or ODESK_TIME_LOGNORM
        self.response_times = self.make_times(times)
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

        o = dict()
        for w,q in itertools.product(xrange(self.num_workers),
                                     xrange(self.num_questions)):
            pCorrect = np.random.random() <= probs[w,q]
            if pCorrect:
                v = self.gt_labels[q]
            else:
                v = not self.gt_labels[q]

            o[w,q] = int(v)  # store truth val as int for consistency

        return o

    def make_times(self, times):
        """Time to complete a vote."""

        gen_times = dict()

        if isinstance(times, list) and isinstance(times[0], collections.Mapping):
            for i,d in enumerate(times):
                for q in xrange(self.num_questions):
                    gen_times[i,q] = sample_dist(d)
        elif isinstance(times, list) or isinstance(times, np.ndarray):
            # assume times given
            return times
        else:
            for w,q in itertools.product(xrange(self.num_workers),
                                         xrange(self.num_questions)):
                if isinstance(times, collections.Mapping):
                    gen_times[w,q] = sample_dist(times)
                elif times is None or times == '0':
                    gen_times[w,q] = 0
                else:
                    raise Exception('Unknown times given')
        print gen_times
        return gen_times

    #-------------- access methods ------------
    def assign(self, votes):
        """Votes is list of (worker, question) pairs"""
        for v in votes:
            if v in self.remaining_time:
                raise Exception('Vote already requested')

            self.remaining_time[v] = self.response_times[v]


    def get_votes(self, time_elapsed=1):
        """Returns dictionary of {(worker, question): vote}"""
        votes = dict() 

        for v in self.remaining_time:
            if self.remaining_time[v] is not None:
                new_t = self.remaining_time[v] - time_elapsed
                if new_t > 0:
                    self.remaining_time[v] = new_t
                else:
                    self.remaining_time[v] = None
                    votes[v] = self.votes[v]

        return votes

