import torch
import numpy as np

class compute_ev:
    def __init__(self, cfg, P, preferences):
        """
        P: n*n の二重確率行列 (torch.Tensor)
        preferences: n*n の選好行列 (torch.Tensor)
                     各行 i はエージェント i の選好を表し、値が大きいほど好む
        """
        self.cfg = cfg
        self.P = P.clone().to(P.device)  # ★変更箇所: P のデバイス維持
        if isinstance(preferences, np.ndarray):
            preferences = torch.tensor(preferences, dtype=torch.float32)
        self.preferences = preferences.clone().detach().float().to(self.P.device)  # ★変更箇所: preferences を P と同じデバイスに
        self.n = cfg.num_goods

    def vectorized_build_graph(self, Q):
        """
        ★変更箇所: グラフ構築をテンソル演算で実施  
        各 (a, b) ペアについて、条件 (preferences[i,a] > preferences[i,b] かつ Q[i,b] > 0)
        をブロードキャストで一括評価し、条件を満たす最初の witness agent とその Q 値を求めます。
        """
        n = self.n
        # self.preferences, Q は同一デバイス上にある前提
        cond = (self.preferences.unsqueeze(2) > self.preferences.unsqueeze(1)) & (Q.unsqueeze(1) > 0)
        exists = cond.any(dim=0)  # shape: (n, n)
        witness = torch.argmax(cond.float(), dim=0)  # shape: (n, n)
        witness = torch.where(exists, witness, torch.full_like(witness, -1))
        goods_idx = torch.arange(n, device=Q.device).view(1, n).expand(n, n)
        V = torch.where(witness != -1, Q[witness.clamp(min=0), goods_idx], torch.zeros_like(witness, dtype=Q.dtype))
        A = (V > 0).float()
        A = A * (1 - torch.eye(n, device=Q.device))
        return witness, V, A

    def vectorized_execute_all_cycles(self, max_cycle_length=4, max_iter=10):
        """
        ★変更箇所:  
         - 動的な while ループと .item() 呼び出しの代わりに、固定回数の for ループで反復処理を行います。
         - これにより、vmap 内での動的制御フローや .item() によるスカラー化を回避します。
         - 各反復で、2サイクル、3サイクル、4サイクルをすべて検出し、対応する更新を accumulate します。
        """
        n = self.n
        device = self.P.device
        Q = self.P.clone()
        total_violation = torch.tensor(0.0, device=device)

        for _ in range(max_iter):  # ★変更箇所: while ループを固定回数の for ループに変更
            witness, V, A = self.vectorized_build_graph(Q)
            update = torch.zeros_like(Q)
            cycle_violation = torch.tensor(0.0, device=device)

            # ----- 2サイクル検出・更新 -----
            # ★変更箇所: 固定シェイプのインデックスを torch.triu_indices で取得
            idx2 = torch.triu_indices(n, n, offset=1, device=device)  # shape: (2, num_pairs)
            a_idx2 = idx2[0]  # shape: (num_pairs,)
            b_idx2 = idx2[1]
            valid_2 = ((A[a_idx2, b_idx2] > 0) & (A[b_idx2, a_idx2] > 0)).float()  # 0/1 のブールマスク
            eps_2 = torch.min(V[a_idx2, b_idx2], V[b_idx2, a_idx2]) * valid_2  # 条件を満たさない箇所は 0
            cycle_violation = cycle_violation + eps_2.sum()  # ★.item() を使用せず tensor のまま加算
            w_ab = witness[a_idx2, b_idx2]
            w_ba = witness[b_idx2, a_idx2]
            update.index_put_((w_ab, b_idx2), -eps_2, accumulate=True)
            update.index_put_((w_ab, a_idx2),  eps_2, accumulate=True)
            update.index_put_((w_ba, a_idx2), -eps_2, accumulate=True)
            update.index_put_((w_ba, b_idx2),  eps_2, accumulate=True)

            # ----- 3サイクル検出・更新 -----
            if max_cycle_length >= 3:
                # ★変更箇所: 全 triplet (a,b,c) を固定シェイプで生成（flatten して使用）
                grid3 = torch.meshgrid(torch.arange(n, device=device),
                                       torch.arange(n, device=device),
                                       torch.arange(n, device=device),
                                       indexing='ij')
                a_idx3 = grid3[0].flatten()  # shape: (n^3,)
                b_idx3 = grid3[1].flatten()
                c_idx3 = grid3[2].flatten()
                # 各 triplet がすべて異なる条件（0/1 のマスク）
                distinct3 = ((a_idx3 != b_idx3) & (b_idx3 != c_idx3) & (a_idx3 != c_idx3)).float()
                cond3 = ((A[a_idx3, b_idx3] > 0) & (A[b_idx3, c_idx3] > 0) & (A[c_idx3, a_idx3] > 0)).float()
                valid_3 = distinct3 * cond3
                eps_3 = torch.min(torch.min(V[a_idx3, b_idx3], V[b_idx3, c_idx3]), V[c_idx3, a_idx3])
                eps_3 = eps_3 * valid_3
                cycle_violation = cycle_violation + eps_3.sum()
                w_ab_3 = witness[a_idx3, b_idx3]
                w_bc_3 = witness[b_idx3, c_idx3]
                w_ca_3 = witness[c_idx3, a_idx3]
                update.index_put_((w_ab_3, b_idx3), -eps_3, accumulate=True)
                update.index_put_((w_ab_3, a_idx3),  eps_3, accumulate=True)
                update.index_put_((w_bc_3, c_idx3), -eps_3, accumulate=True)
                update.index_put_((w_bc_3, b_idx3),  eps_3, accumulate=True)
                update.index_put_((w_ca_3, a_idx3), -eps_3, accumulate=True)
                update.index_put_((w_ca_3, c_idx3),  eps_3, accumulate=True)

            # ----- 4サイクル検出・更新 -----
            if max_cycle_length >= 4:
                # ★変更箇所: 全 quadruplet (a,b,c,d) を固定シェイプで生成
                grid4 = torch.meshgrid(torch.arange(n, device=device),
                                       torch.arange(n, device=device),
                                       torch.arange(n, device=device),
                                       torch.arange(n, device=device),
                                       indexing='ij')
                a_idx4 = grid4[0].flatten()  # shape: (n^4,)
                b_idx4 = grid4[1].flatten()
                c_idx4 = grid4[2].flatten()
                d_idx4 = grid4[3].flatten()
                # すべてのノードが異なる条件（0/1 マスク）
                distinct4 = ((a_idx4 != b_idx4) & (a_idx4 != c_idx4) & (a_idx4 != d_idx4) &
                             (b_idx4 != c_idx4) & (b_idx4 != d_idx4) & (c_idx4 != d_idx4)).float()
                cond4 = ((A[a_idx4, b_idx4] > 0) & (A[b_idx4, c_idx4] > 0) & 
                         (A[c_idx4, d_idx4] > 0) & (A[d_idx4, a_idx4] > 0)).float()
                valid_4 = distinct4 * cond4
                eps_4 = torch.min(torch.min(torch.min(V[a_idx4, b_idx4], V[b_idx4, c_idx4]), V[c_idx4, d_idx4]), V[d_idx4, a_idx4])
                eps_4 = eps_4 * valid_4
                cycle_violation = cycle_violation + eps_4.sum()
                w_ab_4 = witness[a_idx4, b_idx4]
                w_bc_4 = witness[b_idx4, c_idx4]
                w_cd_4 = witness[c_idx4, d_idx4]
                w_da_4 = witness[d_idx4, a_idx4]
                update.index_put_((w_ab_4, b_idx4), -eps_4, accumulate=True)
                update.index_put_((w_ab_4, a_idx4),  eps_4, accumulate=True)
                update.index_put_((w_bc_4, c_idx4), -eps_4, accumulate=True)
                update.index_put_((w_bc_4, b_idx4),  eps_4, accumulate=True)
                update.index_put_((w_cd_4, d_idx4), -eps_4, accumulate=True)
                update.index_put_((w_cd_4, c_idx4),  eps_4, accumulate=True)
                update.index_put_((w_da_4, a_idx4), -eps_4, accumulate=True)
                update.index_put_((w_da_4, d_idx4),  eps_4, accumulate=True)

            # ★変更箇所: 固定反復回数のため、更新がなくてもループは max_iter 回実行
            Q = Q + update
            total_violation = total_violation + cycle_violation

        return total_violation

    def execute_all_cycles(self):
        """
        参考用の逐次処理版
        """
        Q = self.P.clone()
        cycles_exchanges = []
        violation = 0.0
        while True:
            graph = self.build_graph(Q)
            cycle = self.find_cycle(graph)
            if cycle is None:
                break
            epsilons = [edge[3] for edge in cycle]
            epsilon = min(epsilons)
            violation += epsilon
            for (a, b, agent, avail) in cycle:
                Q[agent, b] -= epsilon
                Q[agent, a] += epsilon
            cycles_exchanges.append((cycle, epsilon))
        return violation

    def execute_all_cycles_batch(self):
        """
        ★変更箇所:  
         - torch.vmap を使用して各バッチを一斉に処理  
         - vectorized_execute_all_cycles を vmap で呼び出す  
         ※ self.P と self.preferences のバッチサイズが異なる場合は（例: preferences が2倍なら先頭半分を採用）調整します。
        """
        def process_single(P_batch, preferences_batch):
            ev_instance = compute_ev(self.cfg, P_batch, preferences_batch)
            return ev_instance.vectorized_execute_all_cycles(max_cycle_length=4, max_iter=10)

        if self.P.shape[0] != self.preferences.shape[0]:
            if self.preferences.shape[0] == 2 * self.P.shape[0]:
                fixed_preferences = self.preferences[:self.P.shape[0]]
            else:
                raise ValueError(
                    f"Batch size mismatch: self.P shape {self.P.shape} vs self.preferences shape {self.preferences.shape}. "
                    "These must match."
                )
        else:
            fixed_preferences = self.preferences

        results = torch.vmap(process_single)(self.P, fixed_preferences)
        return results.unsqueeze(-1)
