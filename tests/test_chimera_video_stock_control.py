import unittest
from experiments.chimera_recovery import video_stock_control as S


class StockCapacityTests(unittest.TestCase):
    def test_original_24gib_gpu_stock_full_load_is_replaced_only_after_headroom_gate(self):
        gib=1024**3
        self.assertFalse(S.require_budget(128*gib,20*gib)['reference'])
        self.assertTrue(S.require_budget(96*gib,0,reference=True)['reference'])
        for available,free,reference in ((127*gib,24*gib,False),(128*gib,19*gib,False),(95*gib,24*gib,True)):
            with self.subTest(available=available,free=free,reference=reference):
                with self.assertRaisesRegex(ValueError,'headroom'):S.require_budget(available,free,reference=reference)


if __name__=='__main__':unittest.main()
