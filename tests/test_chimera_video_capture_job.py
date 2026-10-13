import tempfile
import unittest
from pathlib import Path
from experiments.chimera_recovery import video_capture_job as J


class RequestSealTests(unittest.TestCase):
    def test_changed_source_fails_before_any_torch_gpu_or_teacher_import(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);source=p/'source.py';source.write_bytes(b'unit source')
            r={'schema':'recovery_video_teacher_request.v1','sealed':True,'training_started':False,'gpu_launch_performed':False,
                'sources':{'unit':J.B.V.pin(source)},'inputs':{},'jobs':{'capture':{'command':'capture'}}}
            r['manifest_sha256']=J.B.A.digest(r);path=p/'request.json';path.write_bytes(J.B.A.canonical(r)+b'\n')
            sha=J.B.A.file_sha(path);self.assertEqual(J.pinned_request(path,sha,'capture')['sources'],r['sources'])
            source.write_bytes(b'changed source')
            with self.assertRaises(ValueError):J.pinned_request(path,sha,'capture')


if __name__=='__main__':unittest.main()
