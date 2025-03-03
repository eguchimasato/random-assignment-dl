import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from data import Data

def compute_spv(cfg, e_model, r, p, q):
    """
    Computes the strategy proofness violation of the model

    Arguments:
        cfg: Config
        model to evaluate
        r: matching matrix [Batch_size * num_goods, * num_goods]
        p: men's preference [Batch_size, num_goods, num_goods]
        q: women's preference [Batch_size, num_goods, num_goods]
    returns:
        sp_v: torch.Tensor of shape [num_goods, num_goods]
    """
    num_goods = cfg.num_goods
    device = cfg.device
    G = Data(cfg)

    spv = torch.zeros((num_goods, num_goods), device=device)
    for agent_idx in range(num_goods):
        P_mis, Q_mis = G.compose_misreport(p, q, G.mis_array, agent_idx, is_P=True)
        p_mis, q_mis = torch.Tensor(P_mis).to(device), torch.Tensor(Q_mis).to(device)
        r_mis = e_model(p_mis.view(-1, num_goods, num_goods), q_mis.view(-1, num_goods, num_goods))
        r_mis = r_mis.view(p.shape[0], -1, num_goods, num_goods)

        r_mis_agent = r_mis[:, :, agent_idx, :]

        r_agent = r[:, agent_idx, :].to(device)
        r_agent = r_agent.repeat(1, r_mis_agent.shape[1]).view(r_mis_agent.shape[0], r_mis_agent.shape[1], r_mis_agent.shape[2])

        for f in range(num_goods):
            mask = torch.where(p[:, agent_idx, :] >= p[:, agent_idx, f].view(-1, 1), 1, 0).to(device)
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
    return spv
