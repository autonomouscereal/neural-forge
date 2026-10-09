"""Actual metadata guards and small, independent CPU algebra/readback checks."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import time
import unittest

import c18_ace_host_tables as adapter


class Metadata(unittest.TestCase):
    ace_path = None
    host_path = None

    def setUp(self):
        self.ace = adapter.strict_json(self.ace_path)
        self.host = adapter.strict_json(self.host_path)

    def test_current_actual_metadata(self):
        g = adapter.metadata_geometry(self.ace, self.host)
        self.assertEqual((g['source_hidden'], g['host_hidden']), (2048, 5120))
        self.assertEqual(g['reserved_rows'] - g['eligible_rows'], 1536)
        self.assertFalse(g['host_tied'])

    def test_changed_source_or_host_geometry(self):
        for source, key in ((True, 'hidden_size'), (False, 'hidden_size'), (False, 'intermediate_size')):
            a, h = copy.deepcopy(self.ace), copy.deepcopy(self.host)
            (a['source_config'] if source else h['config'])[key] += 1
            with self.assertRaises(ValueError): adapter.metadata_geometry(a, h)

    def test_incomplete_component_proof(self):
        for key in ('verified_layer_artifacts', 'tokenizer_assets', 'preserved_attention_and_global', 'tensors'):
            a = copy.deepcopy(self.ace)
            a[key].pop(next(iter(a[key])))
            with self.assertRaises(ValueError): adapter.metadata_geometry(a, self.host)

    def test_wrong_generator_and_ledger(self):
        a = copy.deepcopy(self.ace); a['codec_bridge']['source_generator_interval'][1] += 1
        with self.assertRaises(ValueError): adapter.metadata_geometry(a, self.host)
        for mutation in ('duplicate', 'gap', 'width', 'vocab'):
            h = copy.deepcopy(self.host)
            if mutation == 'duplicate': h['ledger']['ranges'].append(h['ledger']['ranges'][-1])
            if mutation == 'gap': h['ledger']['ranges'][1]['begin'] += 1
            if mutation == 'width': h['ledger']['ranges'][-1]['width'] -= 1
            if mutation == 'vocab': h['ledger']['vocab_size'] -= 1
            with self.assertRaises(ValueError): adapter.metadata_geometry(self.ace, h)

    def test_strict_json_and_file_bindings(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'x.json'
            for raw in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}'):
                p.write_text(raw)
                with self.assertRaises(ValueError): adapter.strict_json(p)
            p.write_text('{"x":1}')
            b = {'path': str(p), **adapter.file_pin(p)}
            self.assertEqual(adapter.checked_file(b), p.resolve())
            p.write_text('{"x":2}')
            with self.assertRaises(ValueError): adapter.checked_file(b)

    def test_chart_receipt_identity_and_directions(self):
        bindings = {'actual_identity': {'sha256': 'pinned'}}
        r = {'schema': adapter.CHART_SCHEMA, 'directions': adapter.DIRECTIONS,
             'bias_policy': 'zero_no_bias_tensors', 'numerical_policy': 'diagnostic_only',
             'bindings': bindings, 'pairing': 'exact_record_and_position', 'fit_split': 'FIT',
             'synthetic_fixture': False}
        adapter.validate_chart_receipt(r, bindings)
        for key, wrong in (('synthetic_fixture', True), ('fit_split', 'TEST'), ('bias_policy', 'fitted'),
                           ('pairing', 'different_position'), ('directions', {}), ('bindings', {})):
            bad = copy.deepcopy(r); bad[key] = wrong
            with self.assertRaises(ValueError): adapter.validate_chart_receipt(bad, bindings)

    def test_paired_receipt_record_position_identity(self):
        bindings = {k: {'path': k, 'bytes': 1, 'sha256': 'pin'} for k in
                    ('ace_component_manifest','host_bundle_receipt','host_config','host_index',
                     'host_runtime_contract','paired_capture_rows')}
        receipt = {'schema':'chimera.ace_host_paired_capture.development.v1', 'bindings':bindings,
                   'synthetic_fixture':False, 'row_count':1,
                   'row_semantics':{'source_embedding_rows':'source_codec_embedding',
                     'host_embedding_rows':'declared_host_codec_embedding_target',
                     'source_final_hidden_rows':'ACE_final_normalized_hidden',
                     'host_final_hidden_rows':'S3_final_normalized_hidden'},
                   'pairs':[{'record_id':'observed-row','position':3,'source_id':151669,
                             'native_id':252416,'split':'FIT'}]}
        geometry = adapter.metadata_geometry(self.ace,self.host)
        self.assertEqual(adapter.paired_capture_join(receipt,bindings,geometry),receipt['pairs'])
        for key, wrong in (('native_id',252417),('source_id',215669),('position',True),('split','HELD')):
            bad=copy.deepcopy(receipt);bad['pairs'][0][key]=wrong
            with self.assertRaises(ValueError):adapter.paired_capture_join(bad,bindings,geometry)
        bad=copy.deepcopy(receipt);bad['pairs']*=2;bad['row_count']=2
        with self.assertRaises(ValueError):adapter.paired_capture_join(bad,bindings,geometry)

    def test_source_output_separation(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); donor=root/'donor'; reader=root/'reader'; output=root/'output'
            (donor/'component').mkdir(parents=True);reader.mkdir();output.mkdir()
            source=donor/'component'/'native_component.json';source.write_text('{}')
            paths={'ace_component_manifest':source}
            self.assertEqual(adapter.guard_output(output/'tables.safetensors',paths,reader),
                             (output/'tables.safetensors').resolve())
            for location in (donor/'new.safetensors',donor/'component'/'new.safetensors',reader/'new.safetensors'):
                with self.assertRaises(ValueError):adapter.guard_output(location,paths,reader)
            (output/'taken.safetensors.json').write_text('{}')
            with self.assertRaises(FileExistsError):adapter.guard_output(output/'taken.safetensors',paths,reader)


class Algebra(unittest.TestCase):
    def setUp(self):
        import torch
        self.t = torch
        self.e = torch.tensor([[8., 9.], [9., 8.], [1., 2.], [3., 4.], [5., 6.], [7., 8.]])
        self.a = torch.tensor([[2., 0., 0.], [0., 3., 0.]])
        self.b = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
        self.c = {'host_to_source.weight': self.a, 'source_to_host.weight': self.b}
        self.g = {'source_hidden': 2, 'host_hidden': 3, 'source_start': 2,
                  'eligible_rows': 3, 'reserved_rows': 5, 'native_begin': 100, 'host_tied': False}

    def table(self, name, chunk=2):
        return self.t.cat(list(adapter.projected_batches(self.e, self.c, self.g, name=name, chunk_rows=chunk)))

    def test_directed_maps_have_different_observed_tables(self):
        self.assertTrue(self.t.equal(self.table('embed.weight'), self.t.tensor(
            [[1.,2.,3.], [3.,4.,7.], [5.,6.,11.], [0.,0.,0.], [0.,0.,0.]])))
        self.assertTrue(self.t.equal(self.table('lm_head.weight'), self.t.tensor(
            [[2.,6.,0.], [6.,12.,0.], [10.,18.,0.], [0.,0.,0.], [0.,0.,0.]])))

    def test_independent_source_logits_oracle(self):
        # Independent scalar oracle: source=(2,6), row logit=2*e0+6*e1.
        self.assertEqual((self.table('lm_head.weight')[:3] @ self.t.tensor([1.,2.,3.])).tolist(), [14.,30.,46.])

    def test_chunk_boundaries_and_bf16_source(self):
        for chunk in (1,2,3,5,1024):
            self.assertTrue(self.t.equal(self.table('embed.weight', chunk), self.table('embed.weight')))
        self.e = self.e.bfloat16()
        self.assertEqual(self.table('lm_head.weight').dtype, self.t.float32)
        self.assertEqual(self.table('lm_head.weight')[1].tolist(), [6.,12.,0.])

    def test_chart_direction_inventory_dtype_nonfinite(self):
        for name, value in (('host_to_source.weight', self.a.T), ('source_to_host.weight', self.b.double()),
                            ('source_to_host.weight', self.b * float('inf'))):
            bad = dict(self.c); bad[name] = value
            with self.assertRaises(ValueError): adapter.validate_tensors(self.e, bad, self.g)
        bad = dict(self.c); bad['host_to_source.bias'] = self.t.zeros(2)
        with self.assertRaises(ValueError): adapter.validate_tensors(self.e, bad, self.g)

    def test_nonfinite_rows_output_and_chunk_are_faults(self):
        self.e[2,0] = float('nan')
        with self.assertRaises(ValueError): self.table('embed.weight')
        self.e[2,0] = 3e38
        with self.assertRaises(ValueError): self.table('lm_head.weight')
        self.e[2,0] = 1
        for chunk in (0,1025,True):
            with self.assertRaises(ValueError): self.table('embed.weight', chunk)

    def test_large_finite_coefficients_are_not_fidelity_refusals(self):
        self.c['host_to_source.weight'] = self.a * 100000
        self.assertTrue(bool(self.t.isfinite(self.table('lm_head.weight')).all()))

    def test_file_readback_and_c16_tensor_hash_domain(self):
        from safetensors.torch import load_file
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'tables.safetensors'
            r = adapter.publish_tables(path, self.e, self.c, self.g, {'fixture': True}, chunk_rows=2)
            loaded = load_file(str(path))
            self.assertEqual(set(loaded), {'embed.weight','lm_head.weight'})
            for name, table in loaded.items():
                self.assertTrue(self.t.equal(table, self.table(name)))
                raw = struct.pack('<' + 'f' * 15, *table.flatten().tolist())
                expected = hashlib.sha256(b'F32[5,3]' + raw).hexdigest()
                self.assertEqual(r['tensor_sha256'][name], expected)
            self.assertEqual(r['table_payload_bytes'], 120)
            self.assertEqual(r['unused_rows_exact_zero'], 2)
            self.assertEqual(adapter.strict_json(str(path)+'.json'), r)
            self.assertEqual(r['training_eligibility'], 0)
            self.assertFalse(r['composite_child_installed'])
            before = path.read_bytes()
            with self.assertRaises(FileExistsError): adapter.publish_tables(path, self.e, self.c, self.g, {})
            self.assertEqual(path.read_bytes(), before)

    def test_tied_host_requires_exact_tables_and_failure_keeps_partial(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'tied.safetensors'
            self.g['host_tied'] = True
            with self.assertRaises(ValueError): adapter.publish_tables(path, self.e, self.c, self.g, {})
            self.assertTrue(path.exists())
            self.assertFalse(Path(str(path)+'.json').exists())
            self.c['host_to_source.weight'] = self.b.T.contiguous()
            good = Path(td) / 'tied-equal.safetensors'
            r = adapter.publish_tables(good, self.e, self.c, self.g, {})
            self.assertEqual(r['tensor_sha256']['embed.weight'], r['tensor_sha256']['lm_head.weight'])

    def test_capture_arrays_and_independent_fit_joins(self):
        rows={'source_embedding_rows':self.e[2:5].clone(), 'host_embedding_rows':self.table('embed.weight')[:3],
              'source_final_hidden_rows':self.t.tensor([[1.,2.],[2.,3.],[3.,4.]]),
              'host_final_hidden_rows':self.t.tensor([[2.,1.,0.],[4.,2.,1.],[6.,3.,2.]])}
        pairs=[{'source_id':i} for i in (2,3,4)]
        cr={'tensor_sha256':{k:adapter.tensor_digest(v) for k,v in rows.items()}}
        bindings={'paired_capture_rows':{'sha256':'observed-file-pin'}}
        receipt={'tensor_sha256':{k:adapter.tensor_digest(v) for k,v in self.c.items()},
                 'fit':{'method':'ridge_through_origin','maps_independently_fitted':True,'row_count':3,
                        'rows_file_sha256':'observed-file-pin',
                        'input_direction':'source_embedding_rows_to_host_embedding_rows',
                        'output_direction':'host_final_hidden_rows_to_source_final_hidden_rows'}}
        adapter.validate_capture_and_fit(rows,pairs,self.e,self.c,self.g,receipt,cr,bindings)
        bad=copy.deepcopy(rows);bad['source_embedding_rows'][0,0]+=1
        with self.assertRaises(ValueError):adapter.validate_capture_and_fit(bad,pairs,self.e,self.c,self.g,receipt,cr,bindings)
        for key,wrong in (('row_count',2),('rows_file_sha256','other'),('maps_independently_fitted',False),
                          ('input_direction','host_hidden_to_source_hidden')):
            bad=copy.deepcopy(receipt);bad['fit'][key]=wrong
            with self.assertRaises(ValueError):adapter.validate_capture_and_fit(rows,pairs,self.e,self.c,self.g,bad,cr,bindings)
        bad=copy.deepcopy(receipt);bad['tensor_sha256']['host_to_source.weight']='other'
        with self.assertRaises(ValueError):adapter.validate_capture_and_fit(rows,pairs,self.e,self.c,self.g,bad,cr,bindings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata-ace', required=True)
    parser.add_argument('--metadata-host', required=True)
    parser.add_argument('--metadata-only', action='store_true')
    args = parser.parse_args()
    Metadata.ace_path, Metadata.host_path = args.metadata_ace, args.metadata_host
    suite = unittest.TestLoader().loadTestsFromTestCase(Metadata)
    if not args.metadata_only:
        import torch
        torch.set_num_threads(2)
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(Algebra))
    start = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
                      'skips': len(result.skipped), 'success': result.wasSuccessful(),
                      'seconds': time.monotonic()-start, 'metadata_only': args.metadata_only,
                      'metadata_ace': adapter.file_pin(args.metadata_ace), 'metadata_host': adapter.file_pin(args.metadata_host)}))
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
