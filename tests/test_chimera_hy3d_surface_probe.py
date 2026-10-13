import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np

from experiments.chimera_recovery import hy3d_surface_probe as P
from experiments.chimera_recovery import geometry_prepare as G
from experiments.chimera_recovery import audio_admission as A
from tests.test_chimera_geometry_prepare import V,F


class SurfaceProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name)
        self.surface=self.base/"surface.npy";self.image=self.base/"image.png"
        arr=np.zeros((1,81920,7),dtype=np.float32);arr[:,:,3]=1
        np.save(self.surface,arr,allow_pickle=False);self.image.write_bytes(b"sealed captured input2026")
        self.row={"id":"mesh2026","geometry_target":{"surface":{"sha256":A.file_sha(self.surface),"bytes":self.surface.stat().st_size,
                     "shape":[1,81920,7],"policy":G.SURFACE_POLICY,"generation_input_allowed":False}},
                  "media":[{"type":"image","role":"conditioning","sha256":A.file_sha(self.image)}]}

    def tearDown(self):self.temp.cleanup()

    def seal(self,row=None):
        row=row or self.row;p=self.base/"source_ready.jsonl";p.write_bytes(A.canonical(row)+b"\n")
        manifest={"schema":G.SCHEMA,"sealed":True,"target_leakage_policy":{"clean_encoder_query_xyz_in_generation_inputs":False},
                  "outputs":{"source_ready":{"path":"original-host/source_ready.jsonl","sha256":A.file_sha(p),"rows":1}}}
        manifest["manifest_sha256"]=A.digest(manifest)
        out=self.base/"manifest.json";out.write_bytes(A.canonical(manifest))
        return out

    def test_explicit_path_remap_requires_exact_source_bytes(self):
        row,arr,_=P.source_item(self.seal(),"mesh2026",self.surface,self.image)
        self.assertEqual(row["id"],"mesh2026");self.assertEqual(arr.shape,(1,81920,7))
        self.image.write_bytes(b"different")
        with self.assertRaisesRegex(ValueError,"pairing changed"):
            P.source_item(self.base/"manifest.json","mesh2026",self.surface,self.image)

    def test_windows_manifest_path_remaps_on_linux_without_identity_waiver(self):
        p=self.seal();v=__import__("json").loads(p.read_text())
        v["outputs"]["source_ready"]["path"]="C:\\Users\\operator\\source_ready.jsonl"
        v.pop("manifest_sha256");v["manifest_sha256"]=A.digest(v);p.write_bytes(A.canonical(v))
        row,_,_=P.source_item(p,"mesh2026",self.surface,self.image)
        self.assertEqual(row["id"],"mesh2026")
        (self.base/"source_ready.jsonl").write_text("{}\n")
        with self.assertRaisesRegex(ValueError,"inventory changed"):
            P.source_item(p,"mesh2026",self.surface,self.image)

    def test_target_query_xyz_or_modified_surface_refuse(self):
        row=copy.deepcopy(self.row);row["geometry_target"]["surface"]["generation_input_allowed"]=True
        with self.assertRaisesRegex(ValueError,"pairing changed"):P.source_item(self.seal(row),"mesh2026",self.surface,self.image)
        self.seal();self.surface.write_bytes(b"changed native surface")
        with self.assertRaisesRegex(ValueError,"pairing changed"):P.source_item(self.base/"manifest.json","mesh2026",self.surface,self.image)

    def test_resealed_invalid_normal_and_flags_do_not_pass(self):
        for col,value in ((3,0),(6,1)):
            arr=np.load(self.surface);arr[:,:,col]=value;np.save(self.surface,arr,allow_pickle=False)
            self.row["geometry_target"]["surface"].update(sha256=A.file_sha(self.surface),bytes=self.surface.stat().st_size)
            with self.subTest(col=col),self.assertRaisesRegex(ValueError,"normalized XYZ"):
                P.source_item(self.seal(),"mesh2026",self.surface,self.image)

    def test_reviewed_vae_config_refuses_shape_and_scaling_change(self):
        params={"num_latents":4096,"embed_dim":64,"width":1024,"heads":16,"num_encoder_layers":8,"num_decoder_layers":16,
                "point_feats":4,"pc_size":81920,"pc_sharpedge_size":0,"include_pi":False,"qkv_bias":False,
                "qk_norm":True,"scale_factor":1.0039506158752403}
        self.assertEqual(P.validate_config({"vae":{"params":params}}),params)
        for key in ("scale_factor","point_feats","pc_size"):
            changed=dict(params);changed[key]=0
            with self.subTest(key=key),self.assertRaises(ValueError):P.validate_config({"vae":{"params":changed}})

    def test_real_mesh_surface_recomputation_and_convex_oracle(self):
        p=self.base/"mesh.obj"
        p.write_text("".join("v "+" ".join(map(str,x))+"\n" for x in V)+
                     "".join("f "+" ".join(str(i+1) for i in x)+"\n" for x in F))
        v,f,measure=G.mesh(p);v,transform=G.normalize(v);arr=G.sample_surface(v,f,2026)
        row={"geometry_target":{"mesh":{"sha256":A.file_sha(p)},"measure":measure,
                                "surface":{"normalization":transform,"seed":2026}}}
        out,f=P.actual_mesh_surface(row,arr,p)
        self.assertTrue(np.array_equal(out,v))
        queries,inside=P.convex_oracle(out,f)
        self.assertTrue(inside.any());self.assertTrue((~inside).any())
        self.assertTrue(inside[np.argmin(np.linalg.norm(queries,axis=1))])
        altered=arr.copy();altered[0,0,0]+=.01
        with self.assertRaisesRegex(ValueError,"not sampled"):
            P.actual_mesh_surface(row,altered,p)


if __name__=="__main__":unittest.main()
