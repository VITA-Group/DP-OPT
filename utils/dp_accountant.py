"""DP accountant tools"""
from autodp.autodp_core import Mechanism
from autodp import transformer_zoo

import numpy as np


class DPDeltaOverflowError(Exception):
    pass


# Adapted from https://github.com/yuxiangw/autodp/blob/master/autodp/converter.py
def rdp_to_approxdp(rdp, alpha_max=np.inf, BBGHS_conversion=True):
    # from RDP to approx DP
    # alpha_max is an optional input which sometimes helps avoid numerical issues
    # By default, we are using the RDP to approx-DP conversion due to BBGHS'19's Theorem 21
    # paper: https://arxiv.org/pdf/1905.09982.pdf
    # if you need to use the simpler RDP to approxDP conversion for some reason, turn the flag off

    def approxdp(delta):

        """
        approxdp outputs eps as a function of delta based on rdp calculations
        :param delta:
        :return: the \epsilon with a given delta
        """

        if delta < 0 or delta > 1:
            print("Error! delta is a probability and must be between 0 and 1")
        if delta == 0:
            return rdp(np.inf)
        else:
            def fun(x):  # the input the RDP's alpha
                if x <= 1:
                    return np.inf
                else:
                    if BBGHS_conversion:
                        return np.maximum(rdp(x) + np.log((x - 1) / x) - (np.log(delta) + np.log(x)) / (x - 1), 0)
                    else:
                        return np.log(1 / delta) / (x - 1) + rdp(x)

            results = np.min([fun(alpha) for alpha in range(1, alpha_max)])
            return results

    return approxdp


def approxRDP_to_approxDP(total_delta, delta0, rdp_func, alpha_max=np.inf, BBGHS_conversion=True):
    if total_delta < delta0:
        raise DPDeltaOverflowError(f"Found total_delta < delta0: {total_delta} < {delta0}")
        # return np.inf

    delta1 = total_delta - delta0

    approxdp = rdp_to_approxdp(rdp_func, alpha_max, BBGHS_conversion)

    return approxdp(delta1)


# The RDP analysis here is based on Remark 3.4 in https://arxiv.org/abs/1605.02065
class EM(Mechanism):
    def __init__(self, eps, name='EM', monotone=False):
        Mechanism.__init__(self)
        self.name = name
        self.params = {'eps': eps}

        if monotone:

            def privloss(t, alpha):
                return (np.exp(alpha * (eps - t)) - np.exp(-alpha * t) - (
                            np.exp(alpha * eps - (alpha + 1) * t) - np.exp(eps - (alpha + 1) * t))) / (
                            np.exp(eps - t) - np.exp(-t))

            def RDP_EM(alpha):
                if alpha == np.infty:
                    return eps
                enegt = ((alpha - 1) * (np.exp(alpha * eps) - 1)) / ((alpha) * (np.exp(alpha * eps) - np.exp(eps)))
                return np.log(privloss(np.log(1 / enegt), alpha)) / (alpha - 1)

        else:

            def RDP_EM(alpha):
                if alpha == np.infty:
                    return eps * 2
                temp = (np.sinh(alpha * eps) - np.sinh((alpha - 1) * eps)) / np.sinh(eps)
                return min(1 / 2 * alpha * eps ** 2, np.log(temp) / (alpha - 1))

        self.propagate_updates(RDP_EM, 'RDP')


class EM_peeling(Mechanism):

    def __init__(self, eps, k, name='EM_peeling', monotone=False):
        Mechanism.__init__(self)
        self.name = name
        self.params = {'eps': eps, 'k': k}

        compose = transformer_zoo.Composition()

        mech = EM(eps, monotone=monotone)
        mech.neighboring = 'add_remove'

        mech = compose([mech], [k])
        rdp_total = mech.RenyiDP

        self.propagate_updates(rdp_total, type_of_update='RDP')


# delta0: the delta in the algorithm
class compose_subsampled_limiteddomain(Mechanism):

    def __init__(self, eps, delta0, k, prob, niter, name='compose_subsampled_limiteddomain', monotone=False):
        Mechanism.__init__(self)
        self.name = name

        subsample = transformer_zoo.AmplificationBySampling()  # by default this is using poisson sampling
        compose = transformer_zoo.Composition()

        mech = EM_peeling(eps, k, monotone=monotone)
        mech.neighboring = 'add_remove'

        if prob < 1:
            mech = subsample(mech, prob * (1 - delta0) / (1 - prob * delta0), improved_bound_flag=False)

        mech = compose([mech], [niter])
        rdp_total = mech.RenyiDP

        self.propagate_updates(rdp_total, type_of_update='RDP')


# params = {eps, delta0, k, prob, niter}
def compose_subsampled_LimitedDomain_to_approxDP(
        eps,
        delta,
        total_delta,
        # dleta1,
        subsampling_rate,
        niter,
        k=1,
        monotone=False,
        delta1=None,
):
    """
    Parameters:
    'eps': 1.,
    'k': 1, how many output (top-k)
    'delta': delta0 in the inner noising mechanism. delta-delta0 is the aditional cost for LimitedDomain compared to PATE.
    total_delta: total delta
    'delta1': residual probability added by limited-domain method.
    'prob': prob,
    'niter': max_token*n_prompt,
    'monotone': False  # => means sample-wise DP
    """

    mech = compose_subsampled_limiteddomain(eps, delta, k, subsampling_rate, niter,
                                            monotone=monotone)
    rdp_func = mech.RenyiDP
    accumulated_delta = delta * subsampling_rate * niter
    if delta1 is not None:
        assert total_delta is None, "Should not set toal delta when delta1 is provided."
        total_delta = delta1 + accumulated_delta

    return approxRDP_to_approxDP(total_delta, accumulated_delta, rdp_func, alpha_max=200,
                                 BBGHS_conversion=True), total_delta


# First run limited domain, if fail then run vanilla EM.
# Privacy is equivalent to run EM_peeling for two times.
class EM_peeling_double(Mechanism):
    def __init__(self, eps, k, name='EM_peeling', monotone=False):
        Mechanism.__init__(self)
        self.name = name
        self.params = {'eps': eps, 'k': k}

        compose = transformer_zoo.Composition()

        mech = EM_peeling(eps, k, monotone=monotone)
        mech.neighboring = 'add_remove'

        mech = compose([mech], [2])
        rdp_total = mech.RenyiDP

        self.propagate_updates(rdp_total, type_of_update='RDP')


# delta0: the delta in the algorithm
class compose_subsampled_limiteddomain_and_EM(Mechanism):
    def __init__(self, eps, delta0, k, prob, niter, name='compose_subsampled_limiteddomain', monotone=False):
        Mechanism.__init__(self)
        self.name = name

        subsample = transformer_zoo.AmplificationBySampling()  # by default this is using poisson sampling
        compose = transformer_zoo.Composition()

        mech = EM_peeling_double(eps, k, monotone=monotone)
        mech.neighboring = 'add_remove'

        if prob < 1:
            mech = subsample(mech, prob * (1 - delta0) / (1 - prob * delta0), improved_bound_flag=False)

        mech = compose([mech], [niter])
        rdp_total = mech.RenyiDP

        self.propagate_updates(rdp_total, type_of_update='RDP')


# params = {eps, delta0, k, prob, niter}
def compose_subsampled_LimitedDomain_and_EM_to_approxDP(
        eps,
        delta,
        total_delta,
        # dleta1,
        subsampling_rate,
        niter,
        k=1,
        delta1=None,
):
    """
    Parameters:
    'eps': 1.,
    'k': 1,
    'delta': delta in the inner noising mechanism. delta-delta0 is the aditional cost for LimitedDomain compared to PATE.
    total_delta: total delta
    'delta1': residual probability added by limited-domain method.
    'prob': prob,
    'niter': max_token*n_prompt,
    """

    mech = compose_subsampled_limiteddomain_and_EM(eps, delta, k, subsampling_rate, niter)
    rdp_func = mech.RenyiDP
    accumulated_delta = delta * subsampling_rate * niter
    if delta1 is not None:
        assert total_delta is None, "Should not set toal delta when delta1 is provided."
        total_delta = delta1 + accumulated_delta

    return approxRDP_to_approxDP(total_delta, accumulated_delta, rdp_func, alpha_max=200,
                                 BBGHS_conversion=True), total_delta


# It's just one iteration of EM
# k=1 if only report max
# It's monotonic in this case
def EMpeeling_to_approxDP(eps, k, delta, monotone=True):
    mech = EM_peeling(eps, k, monotone=monotone)

    approxdp = rdp_to_approxdp(mech.RenyiDP, alpha_max=200, BBGHS_conversion=True)

    return approxdp(delta)


if __name__ == '__main__':
    eps = 1.
    delta1 = 1e-4  # 1e-5 => The total delta (delta+delta' in the paper) Should be larger than delta0
    # => failure prob=delta0 * niter which should be larger than prob
    prob = 0.01  # 1. # 1e-3

    delta0 = 1e-6  # The (eps, delta) in the LimitedDomain Alg 1.

    max_token = 50
    n_prompt = 50

    print('noise scale', '1/eps', 1 / eps)
    print('v_perp addition', '1+ln(1/delta)/eps', 1 + np.log(1 / delta0) / eps)
    eps = compose_subsampled_LimitedDomain_to_approxDP(
        eps,
        delta0,
        delta1,
        subsampling_rate=prob,
        niter=max_token * n_prompt,
        monotone=False  # => means sample-wise DP
    )
    print('eps', eps)
