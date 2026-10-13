import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from experiments.chimera_recovery import geometry_prepare as G
from experiments.chimera_recovery import audio_admission as A


V=[(-1,-.7,-.9),(1,-.7,-.9),(1,.7,-.9),(-1,.7,-.9),(-1,-.7,.9),(1,-.7,.9),(1,.7,.9),(-1,.7,.9)]
F=[(0,2,1),(0,3,2),(4,5,6),(4,6,7),(0,1,5),(0,5,4),(3,7,6),(3,6,2),(0,4,7),(0,7,3),(1,2,6),(1,6,5)]


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def obj(self,v=V,f=F):
        p=self.base/"physical.obj"
        p.write_text("".join("v "+" ".join(map(str,x))+"\n" for x in v)+
                     "".join("f "+" ".join(str(i+1) for i in x)+"\n" for x in f))
        return p

    def test_real_closed_mesh_volume_and_normalization(self):
        v,f,m=G.mesh(self.obj())
        self.assertAlmostEqual(m["volume"],8*.7*.9)
        normalized,transform=G.normalize(v)
        self.assertAlmostEqual(np.ptp(normalized,axis=0).max(),1.9998)
        self.assertEqual(transform["bbox_center"],[0.,0.,0.])
        self.assertTrue(m["consistent_orientation"])

    def test_open_planar_and_reversed_volume_refuse(self):
        for v,f in ((V,F[:-1]),([(x,y,0) for x,y,z in V],F),(V,[x[::-1] for x in F])):
            with self.subTest(f=f),self.assertRaises(ValueError):G.mesh(self.obj(v,f))

    def test_nonfinite_degenerate_and_extra_obj_commands_refuse(self):
        v=list(V);v[0]=(float("nan"),0,0)
        with self.assertRaises(ValueError):G.mesh(self.obj(v,F))
        with self.assertRaises(ValueError):G.mesh(self.obj(V,[F[0],*F]))
        p=self.obj();p.write_text(p.read_text()+"usemtl arbitrary\n")
        with self.assertRaises(ValueError):G.mesh(p)

    def test_rgba_is_camera_depth_render_with_exact_object_mask(self):
        v,f,_=G.mesh(self.obj());v,_=G.normalize(v)
        rgba,camera=G.render(v,f,side=96)
        self.assertEqual(rgba.shape,(96,96,4))
        self.assertEqual(set(np.unique(rgba[:,:,3])),{0,255})
        self.assertTrue(np.all(rgba[rgba[:,:,3]==0]==0))
        self.assertGreater(camera["opaque_object_pixels"],500)
        self.assertEqual(camera["depth_test"],"largest camera-near coordinate wins")
        self.assertFalse(camera["annotation_or_chart_present"])

    def test_nearer_triangle_wins_independent_of_face_order(self):
        v=np.array([[-1,-.3,-1],[1,-.3,-1],[0,-.3,1],[-1,.3,-1],[1,.3,-1],[0,.3,1]],dtype=float)
        faces=np.array([[0,1,2],[5,4,3]])
        a,_=G.render(v,faces,side=96,yaw=0,elevation=0)
        b,_=G.render(v,faces[::-1],side=96,yaw=0,elevation=0)
        self.assertTrue(np.array_equal(a,b))

    def test_seeded_actual_surface_has_normals_and_flag_not_fake_geometry(self):
        v,f,_=G.mesh(self.obj());v,_=G.normalize(v)
        a=G.sample_surface(v,f,2026);b=G.sample_surface(v,f,2026)
        self.assertTrue(np.array_equal(a,b));self.assertEqual(a.dtype,np.float32)
        self.assertEqual(a.shape,(1,81920,7));self.assertTrue(np.all(a[:,:,6]==0))
        self.assertTrue(np.allclose(np.linalg.norm(a[:,:,3:6],axis=-1),1,atol=.001))
        # Every actual surface sample lies on a real box face, within quantization.
        extent=np.max(np.abs(v),axis=0)
        self.assertTrue(np.all(np.min(np.abs(np.abs(a[0,:,:3])-extent),axis=1)<.001))
        self.assertFalse(np.array_equal(a,G.sample_surface(v,f,2027)))
        with self.assertRaises(ValueError):G.sample_surface(v,f,2026,count=4096)

    def test_surface_derived_asset_preserves_global_target_identity(self):
        original={"id":"mesh2026","split":"test","media":[{"type":"mesh","sha256":"b"*64}],
                  "provenance":{"group_id":"box2026","source":"operator2026","source_ids":["mesh1"]}}
        derived={"id":"new-render2026","split":"train","media":[{"type":"image","sha256":"c"*64}],
                 "provenance":original["provenance"]}
        groups=A.Groups();a=groups.add(A.identities(original));b=groups.add(A.identities(derived))
        self.assertEqual(groups.find(a),groups.find(b))


if __name__=="__main__":unittest.main()
