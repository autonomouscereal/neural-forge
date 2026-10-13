from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.chimera_recovery import geometry_conditioning as C


class ConditioningTests(unittest.TestCase):
    def test_all_source_channels_coordinates_and_both_frames_survive(self):
        pixels=np.arange(3*64*96,dtype=np.float32).reshape(3,64,96)
        flat,grid,yx,position=C.patchify(pixels)
        self.assertEqual(flat.shape,(24,1536));self.assertEqual(grid.tolist(),[[1,4,6]])
        restored=C.unpack(flat,grid)
        self.assertTrue(np.array_equal(restored[0],pixels));self.assertTrue(np.array_equal(restored[1],pixels))
        self.assertEqual(yx[:8].tolist(),[[0,0],[0,1],[1,0],[1,1],[0,2],[0,3],[1,2],[1,3]])
        self.assertEqual(set(map(tuple,yx.tolist())),{(y,x) for y in range(4) for x in range(6)})
        self.assertFalse(position[:,0].any());self.assertFalse(position[:,3].any())
        self.assertEqual(position[yx[:,0]==3,1].tolist(),[1.]*6)

    def test_wrong_geometry_dtype_or_missing_pixel_is_refused(self):
        for pixels in (np.zeros((3,33,64),dtype=np.float32),np.zeros((4,32,32),dtype=np.float32),
                       np.zeros((3,32,32),dtype=np.float64),np.full((3,32,32),np.nan,dtype=np.float32)):
            with self.subTest(shape=pixels.shape),self.assertRaises(ValueError):C.patchify(pixels)
        flat,grid,_,_=C.patchify(np.zeros((3,32,32),dtype=np.float32))
        with self.assertRaises(ValueError):C.unpack(flat[:-1],grid)

    def test_unreviewed_source_is_never_executed(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/"s7_pairs_parent.py"
            marker=Path(directory)/"executed"
            p.write_text("from pathlib import Path\nPath("+repr(str(marker))+").write_text('ran')\n")
            with self.assertRaisesRegex(ValueError,"source pin"):
                C.pinned_function(p,"qwen_vl_patchify",{})
            self.assertFalse(marker.exists())

    def test_manifest_pin_refuses_before_global_source_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/"manifest.json";p.write_text('{}')
            with self.assertRaisesRegex(ValueError,"preparation pin"):
                C.preparation(p,"a"*64)


if __name__=="__main__":unittest.main()
