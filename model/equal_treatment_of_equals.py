import torch
import numpy as np

class compute_etev:
    def __init__(self, cfg, P, preferences):
        """
        P: batch_size*n*n の二重確率行列 (torch.Tensor)
        preferences: batch_size*n*n の選好行列 (torch.Tensor)
                     各行 i はエージェント i の選好を表し、値が大きいほど好む
        """
        self.P = P.clone()
        
        if isinstance(preferences, np.ndarray):
            preferences = torch.tensor(preferences, dtype=torch.float32)
        self.preferences = preferences.clone().detach().float()
        self.n = cfg.num_goods

    def violation_degree(self, cfg, P, preferences):
        """
        P: (n x n) の確率配分行列。各行はエージェントの割当確率ベクトルを表す。
        preferences: (n x n) の選好行列。各行はエージェントの各財に対する評価を表し、
                     数値が大きいほど好ましいと解釈される。
        """
        if isinstance(preferences, np.ndarray):
            preferences = torch.tensor(preferences, dtype=torch.float32)
        preferences = preferences.clone().detach().float()
        n = self.n
        violation = 0.0  # Pythonのfloatとして初期化

        # 全てのエージェントペア (i, j) について、選好が完全一致していれば検証
        for i in range(n):
            for j in range(i + 1, n):
                if torch.all(preferences[i] == preferences[j]):
                    # L1ノルムを計算し、.item()で数値に変換
                    diff = torch.sum(torch.abs(P[i] - P[j])).item()
                    violation += diff
        return violation
    
    def compute_violation_degrees(self, cfg):
        """
        P と preferences の各バッチに対して violation_degree を計算し、
        それらの結果を合わせて n*1 の行列にする。
        """
        batch_size = self.P.shape[0]
        n = self.n
        results = torch.zeros((batch_size, 1), dtype=torch.float32)

        for b in range(batch_size):
            P_batch = self.P[b].view(n, n)
            preferences_batch = self.preferences[b].view(n, n)
            results[b, 0] = self.violation_degree(cfg, P_batch, preferences_batch)

        return results