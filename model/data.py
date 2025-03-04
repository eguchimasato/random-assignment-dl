import itertools
import numpy as np
import torch
import random

def generate_permutation_array(N, num_goods):
    P = np.zeros((N, num_goods))
    for i in range(N): P[i] = np.random.permutation(num_goods)
    return P

class Data(object):


    def __init__(self, cfg): 
        self.cfg = cfg
        self.num_goods = cfg.num_goods
        self.corr = cfg.corr
        self.device = cfg.device

    def sample_ranking(self, N):
        """ 
        Samples ranked lists
        Arguments
            N: Number of samples
            prob: Probability of truncation       
        Returns:
            Ranked List of shape [N, Num_agents]
        """
                              
        N_trunc = int(N)
        P = generate_permutation_array(N, self.num_goods) + 1
               
        if N_trunc > 0:
            
            # Choose indices to truncate
            idx = np.random.choice(N, N_trunc, replace = False)
            
            # Choose a position to truncate
            trunc = np.random.randint(self.num_goods, size = N_trunc)
            
            # Normalize so preference to remain single has 0 payoff
            swap_vals = P[idx, trunc]
            P[idx, trunc] = 0
            P[idx] = P[idx] - swap_vals[:, np.newaxis]
        
        return P/self.num_goods

    def generate_all_ranking(self, include_truncation=True):
        """ 
        Generates all possible rankings 
        Arguments
            include_truncation: Whether to include truncations or only generate complete rankings
        Returns:
            Ranked of list of shape: [m, num_agents]
                where m = N! if complete, (N+1)! if truncations are included
        """
                  
        if include_truncation is False:
            M = np.array(list(itertools.permutations(np.arange(self.num_goods)))) + 1.0
        else:
            M = np.array(list(itertools.permutations(np.arange(self.num_goods + 1))))
            M = (M - M[:, -1:])[:, :-1]
            
        return M/self.num_goods
    
    def generate_batch(self, batch_size, corr = None):
        """
        Samples a batch of data from training
        Arguments
            batch_size: number of samples
            prob: probability of truncation
        Returns
            P: Agent's preferences, 
                P_{ij}: How much Agent-i prefers to be Good-j
        """

        if corr is None: corr = self.corr
        
        N = batch_size * self.num_goods
        
        P = self.sample_ranking(N)
        
        P = P.reshape(-1, self.num_goods, self.num_goods)                           
                
        if corr > 0.00:
            P_common = self.sample_ranking(batch_size).reshape(batch_size, 1, self.num_goods)
            
            P_idx = np.random.binomial(1, corr, [batch_size, self.num_goods, 1])
            
            P = P * (1 - P_idx) + P_common * P_idx
                
        return P
    
    def compose_misreport(self, P, M, agent_idx):
        """ Composes mis-report
        Arguments:
            P: Agent's preference, [Batch_size, num_goods, num_goods]
            M: Ranked List of mis_reports
                    either [num_misreports, num_goods]
                    or [batch_size, num_misreports, num_goods]                    
            agent_idx: Agent-idx that is mis-reporting
                    
        Returns:
            P_mis: [batch-size, num_misreports, num_goods, num_goods]
            
        """
        
        num_misreports = M.shape[-2]
        P_mis = np.tile(P[:, None, :, :], [1, num_misreports, 1, 1])
        
        P_mis[:, :, agent_idx, :] = M
        
        return P_mis
    
    def generate_all_misreports(self, P, agent_idx, include_truncation = False):
        """ Generates all mis-reports
        Arguments:
            P: Agent's preference, [Batch_size, num_goods, num_goods]
            agent_idx: Agent-idx that is mis-reporting
            include_truncation: Whether to truncate preference or submit complete preferences
                    
        Returns:
            P_mis: [batch-size, M, num_goods, num_goods]
                where M = (num_goods + 1)! if truncations are includes
                      M = (num_goods)! if preferences are complete 
        """
        
        M = self.generate_all_ranking(include_truncation = include_truncation)
        P_mis = self.compose_misreport(P, M, agent_idx)
        
        return P_mis