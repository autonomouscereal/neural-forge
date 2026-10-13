import ast
import hashlib
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from experiments.chimera_recovery.patch_d1_receipt import PIN_HELPER, patch_trainer, patch_launcher, replace_once


class D1ReceiptPatchTest(unittest.TestCase):
    def test_preregistration_verified_and_resolved(self):
        fake = ModuleType('stage2_training_receipt')
        fake.Stage2TrainingReceiptError = ValueError
        fake.assert_preregistration_unedited = lambda path: {'preregistration_sha256': 'a'*64, 'design_document_sha256': 'b'*64}
        ns = {'argparse': __import__('argparse'), 'Any': object, 'Path': Path, 'Refusal': RuntimeError}
        exec(PIN_HELPER, ns)
        with patch.dict('sys.modules', {'stage2_training_receipt': fake}):
            got = ns['preregistration_record'](Namespace(preregistration='contract.md'))
        self.assertEqual(got['preregistration_sha256'], 'a'*64)
        self.assertTrue(Path(got['path']).is_absolute())

    def test_pin_failure_stops_before_runtime(self):
        fake = ModuleType('stage2_training_receipt')
        fake.Stage2TrainingReceiptError = ValueError
        def refuse(path):
            raise ValueError('body changed')
        fake.assert_preregistration_unedited = refuse
        ns = {'argparse': __import__('argparse'), 'Any': object, 'Path': Path, 'Refusal': RuntimeError}
        exec(PIN_HELPER, ns)
        with patch.dict('sys.modules', {'stage2_training_receipt': fake}):
            with self.assertRaisesRegex(RuntimeError, 'body changed'):
                ns['preregistration_record'](Namespace(preregistration='contract.md'))

    def test_patch_refuses_ambiguous_anchor(self):
        for text in ['other source', 'x x']:
            with self.assertRaises(ValueError):
                replace_once(text, 'x', 'y')

    def test_production_launcher_carries_document(self):
        text='LAUNCHER_VERSION=d1_launch-1.1.0\nD1_ARGS=(--child "@BASE@" --parent "$PARENT" )'
        got=patch_launcher(text)
        self.assertIn('--preregistration "$PREREG"', got)


if __name__ == '__main__':
    unittest.main()
