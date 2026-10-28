import copy
import json
from pathlib import Path
import tempfile
import unittest
from experiments.chimera_recovery import evaluation_generation as G


def task():
    return {'schema':'recovery_eval_task.v1','id':'authored-controller-test','group_id':'test-group','split':'test',
            'route':'tool_retry','input':{'messages':[{'role':'user','content':'Divide, observe the error, retry with2.'}],
            'tools':[G.S.TOOL_MAP['local_arithmetic']],'media':[]},'oracle':{'secret_expected_answer':12345},
            'environment':{'followup_user_messages':[],'followup_after_steps':[],
                'tool_catalog_sha256':G.S.sha(G.S.TOOLS),'tool_source_code_sha256':G.TOOL_SOURCE_SHA}}


def authored_decode(sequence):
    sequence=iter(sequence)
    def decoded(messages,tools):
        return {'text':next(sequence),'token_ids':[1],'stop_reason':'eos','model_forward_executed':True,
                'scope':'authored controller unit fixture; never a model recording'}
    return decoded


def call(name,args):return '<tool_call>'+json.dumps({'name':name,'arguments':args})+'</tool_call>'


class ControllerTests(unittest.TestCase):
    def test_real_error_retry_and_model_only_projection(self):
        projected=G.project_task(task());self.assertNotIn('oracle',projected)
        texts=[call('local_arithmetic',{'operation':'divide','a':8,'b':0}),
               call('local_arithmetic',{'operation':'divide','a':8,'b':2}),'Recorded value:4']
        recorded=G.record_one(projected,authored_decode(texts))
        self.assertEqual([x['status'] for x in recorded['tool_outcomes']],['error','success'])
        self.assertEqual(recorded['tool_outcomes'][1]['receipt']['result']['value'],'4')
        self.assertTrue(recorded['tool_outcomes'][0]['receipt']['actual_execution'])
        self.assertEqual(recorded['generation']['stop_reason'],'eos')
        self.assertEqual(recorded['messages'][1]['tool_call_id'],recorded['messages'][0]['tool_calls'][0]['id'])

    def test_real_parallel_dispatch_and_followup(self):
        original=task();original['environment']['followup_after_steps']=[1]
        original['environment']['followup_user_messages']=[{'role':'user','content':'Now multiply3and4.'}]
        first=call('local_arithmetic',{'operation':'add','a':2,'b':3})+call('local_arithmetic',{'operation':'subtract','a':9,'b':4})
        texts=[first,call('local_arithmetic',{'operation':'multiply','a':3,'b':4}),'Recorded value:12']
        recorded=G.record_one(G.project_task(original),authored_decode(texts))
        self.assertEqual([x['receipt']['result']['value'] for x in recorded['tool_outcomes']],[5,5,12])
        self.assertTrue(all(x['receipt']['concurrent_dispatch'] for x in recorded['tool_outcomes'][:2]))
        self.assertIn({'role':'user','content':'Now multiply3and4.'},recorded['messages'])

    def test_unknown_tools_and_duplicate_json_never_execute(self):
        for text in [call('shell',{'command':'do not execute'}),
                     '<tool_call>{"name":"local_arithmetic","name":"shell","arguments":{}}</tool_call>',
                     '<tool_call>{"name":"local_arithmetic"}</tool_call>','<tool_call>{']:
            with self.subTest(text=text):
                recorded=G.record_one(G.project_task(task()),authored_decode([text]))
                self.assertEqual(recorded['generation']['stop_reason'],'error')
                self.assertEqual(recorded['tool_outcomes'],[])

    def test_schema_errors_and_path_escape_are_actual_refusals(self):
        original=task();original['input']['tools']=[G.S.TOOL_MAP['local_file']]
        text=call('local_file',{'operation':'write','name':'../escape.txt','text':'abc'})
        recorded=G.record_one(G.project_task(original),authored_decode([text,'done']))
        self.assertEqual(recorded['tool_outcomes'][0]['status'],'error')
        self.assertEqual(recorded['tool_outcomes'][0]['receipt']['result']['error']['type'],'SchemaValidationError')

    def test_media_refusal_and_oracle_input_rejection(self):
        original=task();original['input']['media']=[{'type':'video','path':'example.mp4'}]
        with self.assertRaisesRegex(ValueError,'media'):G.project_task(original)
        projected=G.project_task(task());projected['oracle']={}
        with self.assertRaisesRegex(ValueError,'oracle-free'):G.record_one(projected,authored_decode(['x']))

    def test_no_authored_callback_can_publish_native_recording(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'actual source-authenticated'):
                G.record_tasks([task()],authored_decode(['x']),phase='baseline',pins={},model={},
                               training_group_ids=[],output=Path(d)/'never')
            self.assertFalse((Path(d)/'never').exists())

    def test_complete_checkpoint_inventory_and_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'config.json';p.write_bytes(b'{}')
            inventory=[{'path':'config.json','sha256':G.A.file_sha(p),'bytes':2}]
            G.verify_checkpoint(root,inventory)
            (root/'extra.bin').write_bytes(b'x')
            with self.assertRaisesRegex(ValueError,'complete'):G.verify_checkpoint(root,inventory)
            (root/'extra.bin').unlink();p.write_bytes(b'[]')
            with self.assertRaisesRegex(ValueError,'changed'):G.verify_checkpoint(root,inventory)


if __name__=='__main__':unittest.main()
