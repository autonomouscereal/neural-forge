from __future__ import annotations

import json
from pathlib import Path
import struct
import tempfile
import unittest

from experiments.chimera_recovery import audio_admission as A
from experiments.chimera_recovery import h3_audio_fixture as H


class TensorFileTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)/"probe.safetensors"
        self.shapes={"input_waveform_stereo32k":[1,2,800],"decoded_waveform":[1,2,800],
                     "x0_raw_audio":[1,32,2,1],"x0_model_audio":[1,32,2,1],"packed_raw_audio":[2,32],"packed_model_audio":[2,32]}
        self.proof={"input_contract":{"expected_latent_frames":1,"target_samples":800},"tensors":{"keys":list(self.shapes)}}

    def tearDown(self):
        self.temp.cleanup()

    def write(self,mutate=None,tail=b""):
        header={};offset=0
        for name,shape in self.shapes.items():
            count=1
            for n in shape:count*=n
            header[name]={"dtype":"F32","shape":shape,"data_offsets":[offset,offset+count*4]}
            offset+=count*4
        if mutate:mutate(header)
        raw=A.canonical(header);raw+=b" "*((-len(raw))%8)
        self.path.write_bytes(struct.pack("<Q",len(raw))+raw+b"\0"*offset+tail)
        self.proof["tensors"]["sha256"]=A.file_sha(self.path)

    def test_exact_complete_file_header_matches_actual_geometry(self):
        self.write();header=H.tensor_inventory(self.path,self.proof)
        self.assertEqual(header["packed_model_audio"]["shape"],[2,32])

    def test_byte_tamper_refuses_even_when_header_unchanged(self):
        self.write();raw=bytearray(self.path.read_bytes());raw[-1]=1;self.path.write_bytes(raw)
        with self.assertRaisesRegex(ValueError,"differs from encoder receipt"):
            H.tensor_inventory(self.path,self.proof)

    def test_wrong_shape_resealed_file_refuses(self):
        self.write(lambda h:h["packed_model_audio"].update(shape=[1,64]))
        with self.assertRaisesRegex(ValueError,"dtype/shape/offset"):
            H.tensor_inventory(self.path,self.proof)

    def test_resealed_trailing_bytes_do_not_enter_tensor_evidence(self):
        self.write(tail=b"\0\0\0\0")
        with self.assertRaisesRegex(ValueError,"trailing bytes"):
            H.tensor_inventory(self.path,self.proof)


if __name__ == "__main__":
    unittest.main()
