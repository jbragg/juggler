"""platform.py
simulator for worker responses"""

from __future__ import division
import collections
import itertools
import random
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
                 times=None, subsample=None,
                 thetas=None):

        self.gt_labels = gt_labels
        self.num_questions = len(gt_labels)
        self.is_deterministic = True
        self.gen_method = 'reg'

        if votes is not None:
            self.num_workers = votes.shape[0]
            self.votes = votes
            self.gt_skills = skills
            self.gt_difficulties = difficulties
            
            if subsample:
                assert subsample < self.num_workers
                workers = sorted(random.sample(range(self.num_workers),subsample))
                
                votes_subsample = dict()
                skills_subsample = np.ones(subsample)

                for i,w in enumerate(workers):
                    for q in xrange(self.num_questions):
                        votes_subsample[i,q] = votes[w,q]
                    
                    skills_subsample[i] = skills[w]
                    
                self.votes = votes_subsample
                self.gt_skills = skills_subsample
                
                self.is_deterministic = False
                self.num_workers = subsample

                assert len(self.gt_skills) == subsample
                assert len(self.gt_skills) == len(set(w for w,q in self.votes))
                                
                print 'Subsampling workers {}'.format(workers)
                
                
        elif num_workers is not None:
            assert not subsample
 
            # new kg method
            if thetas is not None:
                difficulties = thetas
                self.gen_method = 'kg'
            
            if not isinstance(skills, np.ndarray):
                skills = skills or ODESK_SKILL_DIST
            if not isinstance(difficulties, np.ndarray):
                difficulties = difficulties or BETA_DIFFS
            self.num_workers = num_workers
            self.gt_skills, self.gt_difficulties = self.get_params(
                                                                skills,
                                                                difficulties)

            # new kg method
            if thetas is not None:
                assert all(self.gt_skills >= 0) and all(self.gt_skills <= 1)
                self.votes = self.make_votes(self.gt_skills, self.gt_difficulties,
                                             method='kg')
            else:
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
                self.is_deterministic = False
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
                self.is_deterministic = False
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

    def allprobs_kg(self, skills, thetas):
        """Return |workers| x |questions| matrix with prob of worker answering
        correctly. Uses new knowledge_gradient method.
        """
        def prob_kg(s, t):
            """Probability perfect worker correct and worker correct or
            both wrong.
            """
            return np.maximum(t,1-t)*s + np.minimum(t,1-t)*(1-s)
        
            
        return prob_kg(skills[:, np.newaxis],
                       thetas)


    def make_votes(self, skills, difficulties, method=None):
        """Generate worker votes. Unvectorized, but shouldn't matter."""
        self.is_deterministic = False
        
        if method == 'kg':
            probs = self.allprobs_kg(skills, difficulties)
        else:
            probs = self.allprobs(skills, difficulties)
        print probs

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
            self.is_deterministic = False
        elif isinstance(times, list) or isinstance(times, np.ndarray):
            # assume times given
            return times
        else:
            for w,q in itertools.product(xrange(self.num_workers),
                                         xrange(self.num_questions)):
                if isinstance(times, collections.Mapping):
                    gen_times[w,q] = sample_dist(times)
                    self.is_deterministic = False
                elif times is None or times == '0':
                    gen_times[w,q] = 0
                else:
                    raise Exception('Unknown times given')
        
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

