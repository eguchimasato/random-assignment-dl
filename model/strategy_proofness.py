import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from data import Data

class compute_spv:
    def __init__(self, cfg,  e_model, r, p):
        """
        P: n*n の二重確率行列 (torch.Tensor)
        preferences: n*n の選好行列 (torch.Tensor)
                     各行 i はエージェント i の選好を表し、値が大きいほど好む
        """
        self.n = cfg.num_goods
        self.device = cfg.device
        self.e_model = e_model
        self.r = r.clone()
        self.p = p
        

    def calculate_spv(self, cfg, e_model, r, p):
        """
        Computes the strategy proofness violation of the model

        Arguments:
            cfg: Config
            model to evaluate
            r: matching matrix [Batch_size * num_goods, * num_goods]
            p: Agent's preference [Batch_size, num_goods, num_goods]
        returns:
            sp_v: torch.Tensor of shape [num_goods, num_goods]
        """
        num_goods = self.n
        device = self.device
        G = Data(cfg)

        spv = torch.zeros((num_goods, num_goods), device=device)
        p = torch.tensor(p, device=device, dtype=torch.float32)
        for agent_idx in range(num_goods):
            P_mis = G.compose_misreport(p.cpu().numpy(), G.generate_all_ranking(), agent_idx)
            p_mis = torch.tensor(P_mis, device=device, dtype=torch.float32)
            r_mis = e_model(p_mis.view(-1, num_goods, num_goods))
            r_mis = r_mis.view(p.shape[0], -1, num_goods, num_goods)

            r_mis_agent = r_mis[:, :, agent_idx, :]

            r_agent = r[:, agent_idx, :].to(device)
            r_agent = r_agent.unsqueeze(1).repeat(r_mis_agent.shape[0] // r_agent.shape[0], 1, 1)

            #print(f"r_mis_agent shape: {r_mis_agent.shape}")
            #print(f"r_agent shape: {r_agent.shape}")

            for f in range(num_goods):
                mask = torch.where(p[:, agent_idx, :].to(device) >= p[:, agent_idx, f].view(-1, 1).to(device), torch.tensor(1, device=device),torch.tensor(0,device=device))
                mask = mask.repeat(1, r_mis_agent.shape[1]).view(r_mis_agent.shape[0], r_mis_agent.shape[1], r_mis_agent.shape[2])
                spv_values = ((r_mis_agent - r_agent) * mask).sum(-1).relu()
                spv_value = spv_values.sum(-1).mean()
                spv[agent_idx, f] = spv_value
                """
                if spv_value > 0:
                    print(f"SPV detected for agent {agent_idx} and f {f}")
                    print(f"SPV value: {spv_value}")
                    print(f"Misreported preferences (P_mis): {P_mis}")
                    print(f"Individual values: {spv_values}")
                """
        results = spv.sum()
        return results