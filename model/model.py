import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
from efficiency import compute_ev
from strategy_proofness import compute_spv
from equal_treatment_of_equals import compute_etev

class NeuralNet(nn.Module):
    """ Neural Network Module for Matching """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        device = cfg.device
        num_goods = self.cfg.num_goods
        num_hidden_nodes = self.cfg.num_hidden_nodes

        self.layers = nn.Sequential(
            nn.Linear(num_goods * num_goods, num_hidden_nodes),
            nn.LeakyReLU(),
            nn.Linear(num_hidden_nodes, num_hidden_nodes),
            nn.LeakyReLU(),
            nn.Linear(num_hidden_nodes, num_hidden_nodes),
            nn.LeakyReLU(),
            nn.Linear(num_hidden_nodes, num_hidden_nodes),
            nn.LeakyReLU(),
            nn.Linear(num_hidden_nodes, num_goods * num_goods)
            ).to(device)

    def forward(self, p):
        def sinkhorn_normalization(r, num_iters=10):
            for _ in range(num_iters):
                r = F.normalize(r, p=1, dim=1, eps=1e-8)  # 行方向に正規化
                r = F.normalize(r, p=1, dim=2, eps=1e-8)  # 列方向に正規化
            return r

        if isinstance(p, np.ndarray):
            p = torch.tensor(p, device=self.cfg.device, dtype=torch.float32)

        x = p.view(-1, self.cfg.num_goods ** 2)
        r = self.layers(x)
        r = r.view(-1, self.cfg.num_goods, self.cfg.num_goods)
        r = F.softplus(r)
        r = sinkhorn_normalization(r)
        return r

def train_model(cfg, model, data):
    """
    """
    device = cfg.device
    num_epochs = cfg.epochs
    lr = cfg.lr
    batch_size = cfg.batch_size

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    # 学習率スケジューラの設定：100エポックごとにlrを0.9倍に減衰
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    # 損失関数の重み付け
    lambda_spv = 1.0  # c_1 の重み初期値
    lambda_etev = 1.0  # c_2 の重み初期値
    #lambda_c = 1.0  # c の重み初期値

    rho = 1  # 重み付けのパラメータ
    
    print("Training started.")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        P = data.generate_batch(batch_size, corr = None)
        r = model(P)
        
        # 損失の計算
        spv_computer = compute_spv(cfg, model, r, P)
        spv = spv_computer.calculate_spv(cfg, model, r, P)  # 制約条件1の損失
        etev_computer = compute_etev(cfg, r, P)
        etev = etev_computer.compute_violation_degrees(cfg).sum()  # 制約条件2の損失
        ev_computer = compute_ev(cfg, r, P)
        objective_loss = ev_computer.execute_all_cycles_batch().sum()  # 目的関数

        # 総合損失
        total_loss = objective_loss + lambda_spv * spv + lambda_etev * etev + rho * (spv + etev) ** 2
        
        # 逆伝播とパラメータ更新
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 学習率の更新
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            # パラメータの更新
            lambda_spv += rho * spv.sum().item()
            lambda_etev += rho * etev.sum().item()

            if etev.sum().item() > 0.1: 
                rho *= 1.01

        #lambda_c += rho * (spv.sum().item() + etev.sum().item())
        
        if (epoch + 1) % 100 == 0: 
            print(f"Epoch: {epoch+1}")
            print(f"Total Loss: {total_loss.item()}")
            print(f"SPV: {spv.item()}")
            print(f"ETEV: {etev.item()}")
            print(f"Objective Loss(ev): {objective_loss.item()}")
            print(f"Parameters: lambda_spv = {lambda_spv}, lambda_sv = {lambda_etev}, rho = {rho}, lr = {scheduler.get_last_lr()[0]}")
            print("---------------------------")

    print("Training completed.")
