import sys
import os
import unittest
import torch

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.efficiency import compute_ev

class Config:
    def __init__(self, num_goods):
        self.num_goods = num_goods

class TestEfficiency(unittest.TestCase):
    def test_execute_all_cycles_batch(self):
        # 設定の準備
        cfg = Config(num_goods=4)
        P = torch.randn(2, 4, 4)  # バッチサイズ2のダミーデータの準備
        preferences = torch.randn(2, 4, 4)  # バッチサイズ2のダミーデータの準備
        
        # compute_evクラスのインスタンスを作成
        ev_instance = compute_ev(cfg, P, preferences)
        
        # execute_all_cycles_batch関数を呼び出す
        result = ev_instance.execute_all_cycles_batch()
        
        self.assertIsNotNone(result)

if __name__ == '__main__':
    unittest.main()