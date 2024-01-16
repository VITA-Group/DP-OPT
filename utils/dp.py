"""DP noise mechanisms"""
import numpy as np
import torch
from torch.distributions import Gumbel
from .dp_accountant import compose_subsampled_LimitedDomain_to_approxDP, \
    compose_subsampled_LimitedDomain_and_EM_to_approxDP, EMpeeling_to_approxDP


class NotFoundLDTop1(Exception):
    pass


class DPExpenseOverflow(Exception):
    pass


class LDGumbelMechanism(object):
    """Limit Domain Gumbel mechanism.
    
    Paper: Durfee, D., & Rogers, R. M. (2019). Practical differentially private 
    top-k selection with pay-what-you-get composition. NeurIPS.

    Inputs:
        eps, delta: The per-step privacy parameters.
        k_bar: The number of DP candidates.
    """

    def __init__(self, eps: float, delta: float, k_bar=10,
                 target_eps=None, target_delta=None, delta1=None,
                 subsampling_rate=1., compose_method='rdp', fail_mode='ld_pate'):
        self.eps = eps
        self.delta = delta
        self.k_bar = k_bar
        self.target_eps = target_eps
        self.target_delta = target_delta
        self.delta1 = delta if delta1 is None else delta1
        self.subsampling_rate = subsampling_rate

        self.total_k = 0
        self.total_queries = 0

        self.compose_method = compose_method
        
        self.fail_mode = fail_mode  # how to handle fail. Only use as a flag. Implementation is inside ensemble.py

    def get_top1(self, cnts: torch.Tensor, dim, k_bar=None, sens=2.):
        """Input histogram (cnts), output top1 idx in `cnt`.
        
        Parameters:
            dim: the dimension of the historgram.
            sens: l0 sensitivity.
        """
        self.total_queries += 1
        if k_bar is None:
            k_bar = self.k_bar
        assert k_bar <= dim, f"Invalid k_bar: k_bar which is larger than dim={dim}"
        real_len_cnts = len(cnts)
        sorted_cnts, sorted_idxs = torch.sort(cnts, descending=True)
        gumbel = Gumbel(0., 1 / self.eps)
        output_rand_if_fail = False

        if k_bar < dim:
            if k_bar + 1 > real_len_cnts:
                h_perp = 0
            else:
                h_perp = sorted_cnts[k_bar]

            h_perp = h_perp + 1 + np.log(min(sens, k_bar, dim - k_bar) / self.delta) / self.eps
            v_perp = h_perp + gumbel.sample()
            v_perp = v_perp.item()

            if k_bar > real_len_cnts:
                v_perp_0 = torch.max(gumbel.sample((k_bar - real_len_cnts,))).item()
                if v_perp_0 >= v_perp:  # at least, the run will not fail at k_bar.
                    v_perp = v_perp_0
                    output_rand_if_fail = True

                # v_perp = max(v_perp, v_perp_0)
                k_bar = real_len_cnts
        else:  # k_bar=dim, reduce to PATE
            if k_bar > real_len_cnts:
                # simulate the random sample of the rest tokens.
                v_perp = torch.max(gumbel.sample((k_bar - real_len_cnts,))).item()
                k_bar = real_len_cnts
                output_rand_if_fail = True
            else:  # real_len_cnts = k_bar = dim
                v_perp = - np.inf  # never fail

        v_cnts = sorted_cnts[:k_bar] + gumbel.sample((k_bar,)).to(sorted_cnts.device)
        v_max_idx = torch.argmax(v_cnts)
        if v_cnts[v_max_idx] >= v_perp:  # success
            self.total_k += 1
            self.check_dp_budget()  # has to be after self.total_k
            return sorted_idxs[v_max_idx]
        else:  # fail
            if output_rand_if_fail:
                self.total_k += 1
                rand_idx = torch.randint(real_len_cnts, dim, (1,))[0].item()
                self.check_dp_budget()
                return rand_idx  # a random choice out of the `cnts` set.
            else:
                self.check_dp_budget()
                raise NotFoundLDTop1()

    def get_dp_expense(self, total_queries=None, return_all_cases=False):
        if total_queries is None:
            total_queries = self.total_queries
        if self.compose_method == 'simple':
            k = self.total_k
            l = total_queries
            eps_ = [None] * 3
            eps_[0] = k * self.eps
            eps_[1] = k * self.eps * (np.exp(self.eps) - 1) / (np.exp(self.eps) + 1) + self.eps * np.sqrt(
                2. * k * np.log(1 / self.delta1))
            eps_[2] = k * (self.eps ** 2) / 2. + self.eps * np.sqrt(0.5 * k * np.log(1 / self.delta1))
            eps = np.min(eps_)
            delta = 2 * l * self.delta + self.delta1
            if return_all_cases:
                eps_, delta
            else:
                return eps, delta
        elif self.compose_method == 'rdp':
            total_eps, total_delta = compose_subsampled_LimitedDomain_to_approxDP(
                self.eps,
                self.delta,
                None,  # self.target_delta,
                subsampling_rate=self.subsampling_rate,
                # niter=max_token*n_prompt,
                niter=total_queries,
                delta1=self.delta1,
            )
            return total_eps, total_delta
        elif self.compose_method == 'rdp_em':
            # EM accoutant, very high
            total_eps, total_delta = compose_subsampled_LimitedDomain_and_EM_to_approxDP(
                self.eps,
                self.delta,
                None,  # self.target_delta,
                subsampling_rate=self.subsampling_rate,
                # niter=max_token*n_prompt,
                niter=total_queries,
                delta1=self.delta1,
            )
            return total_eps, total_delta
        else:
            raise NotImplementedError(f"compose_method: {self.compose_method}")

    def check_dp_budget(self, raise_error=True, verbose=False):
        if self.target_eps is not None:
            eps, delta = self.get_dp_expense()
            if verbose:
                print(f"# dp eps={eps:.4f}, delta={delta:g}")
            if eps > self.target_eps or delta > self.target_delta:
                if raise_error:
                    raise DPExpenseOverflow()
                else:
                    return False
        return True


class ExpMechanism(object):
    def __init__(self, eps: float, target_eps=None, target_delta=None):
        self.eps = eps
        self.target_eps = target_eps
        self.target_delta = target_delta

        self.total_k = 0
        self.total_queries = 0

    def get_topk(self, cnts: torch.Tensor, k, sens=1.):
        assert self.total_queries < 1, "Only allow once query."
        gumbel = Gumbel(0., sens / self.eps)
        v = cnts + gumbel.sample(cnts.shape)
        self.total_k += k
        self.total_queries += 1
        return torch.argsort(v, descending=True)[:k]

    def get_dp_expense(self, n_query=None, total_k=None):
        if n_query is None:
            n_query = self.total_queries
        if total_k is None:
            total_k = self.total_k
        assert n_query == 1, f"Invalid query times: {n_query}."
        eps = EMpeeling_to_approxDP(eps=self.eps, k=total_k, delta=self.target_delta)
        return eps, self.target_delta


if __name__ == "__main__":
    pass
