from __future__ import division
import numpy as np

class ProbabilityModule():
    def prob_correct(self, s, d):
        raise NotImplementedError

    def prob_correct_ddifficulty(self, s, d):
        raise NotImplementedError

    def prob_correct_dskill(self, s, d):
        raise NotImplementedError

    def allprobs(self, skills, difficulties):
        """Return |workers| x |questions| matrix with prob of worker answering
        correctly.
        """
        skills = np.array(skills)
        difficulties = np.array(difficulties)
        return self.prob_correct(skills[:, np.newaxis],
                                 difficulties)

    def allprobs_ddifficulty(self, skills, difficulties):
        """Return 1 x |difficulties| matrix
        """
        skills = np.array(skills)
        difficulties = np.array(difficulties)
        return self.prob_correct_ddifficulty(skills[:, np.newaxis],
                                             difficulties)

    def allprobs_dskill(self, skills, difficulties):
        """Return 1 x |skills| matrix
        """
        skills = np.array(skills)
        difficulties = np.array(difficulties)
        return self.prob_correct_dskill(skills[:, np.newaxis],
                                        difficulties)
    
class ProbabilityDai(ProbabilityModule):
    def transform_skill(self, s):
        """Transform skill to inverse skill"""
        return 1/s
        #return s
        
    def prob_correct(self, s, d):
        """p_correct = 1/2(1+(1-d_q)^(1/skill)"""
        s = self.transform_skill(s)
        return 1/2*(1+np.power(1-d,s))

    def prob_correct_ddifficulty(self, s, d):
        s = self.transform_skill(s)
        return -1/2*s*np.power(1-d,s-1)

    def prob_correct_dskill(self, s, d):
        s = self.transform_skill(s)
        return 1/2*np.power(1-d,s)*np.log(1-d)

                                    
class ProbabilityMDP(ProbabilityModule):
    def transform_difficulty(self, d):
        """Transform difficulty to probability perfect worker answers correctly
        """
        #return d
        return 1.5 - d
    
    def prob_correct(self, s, d):
        """ Difficulty between 0 and 1, (skill between 0.5 and 1)
        """
        #perfect_s = self.transform_difficulty(d)
        #return perfect_s * s + (1-perfect_s) * (1-s)
        return (1-1/2*d)*s + 1/2*d*(1-s)
        
    def prob_correct_ddifficulty(self, s, d):
        # return (2 * s - 1) * np.ones(d.shape)
        #return 1 - 2 * s
        return 1/2 - s
        
    def prob_correct_dskill(self, s, d):
        # perfect_s = self.transform_difficulty(d)
        # return (2 * perfect_s - 1) * np.ones(s.shape)
        #return 2 - 2 * d
        return 1 - d
        
