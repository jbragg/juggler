from scipy.special import gamma
from scipy.stats import beta

beta = beta.pdf

# beta.pdf(x, a, b)
#def beta(x, a, b):
#    return gamma(a+b)/(gamma(a)*gamma(b)) * x**(a-1) * (1-x)**(b-1)

def dbeta(x, a, b):
    return gamma(a+b)/(gamma(a)*gamma(b)) * \
           ((a-1) * x**(a-2) * (1-x)**(b-1) - x**(a-1) * (b-1) * (1-x)**(b-2))

assert dbeta(0.5, 2, 2) == 0
assert dbeta(0.6, 2, 2) != 0
