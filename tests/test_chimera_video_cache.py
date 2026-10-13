import unittest
import numpy as np

from experiments.chimera_recovery import video_cache as V
from experiments.chimera_recovery import video_source as S


class TemporalInputTests(unittest.TestCase):
    def test_physical_frame_pts_cannot_shift_duplicate_or_sample(self):
        self.assertEqual(V.full_frame_clock(list(range(8)), '1/8', '8', 8), [i/8 for i in range(8)])
        for pts in ([None]*8,list(range(1,9)),[0,1,2,3,4,5,6,6],list(range(0,16,2))):
            with self.assertRaises(ValueError):V.full_frame_clock(pts,'1/8','8',8)

    def test_distinct_eight_frames_all_pixels_survive_temporal_pack_inverse(self):
        pixels = np.arange(8*3*32*64, dtype=np.float32).reshape(8,3,32,64)
        flat, grid = V.patchify(pixels)
        self.assertEqual(grid.tolist(), [[4,2,4]])
        self.assertEqual(flat.shape, (32,1536))
        np.testing.assert_array_equal(V.unpack(flat,grid), pixels)
        self.assertEqual(flat[0,0], pixels[0,0,0,0])
        self.assertEqual(flat[0,256], pixels[1,0,0,0])

    def test_no_silent_duplicate_drop_pad_or_out_of_order_frames(self):
        with self.assertRaisesRegex(ValueError,'no silent frame pad'):
            V.patchify(np.zeros((7,3,32,32),np.float32))
        for indices,total in (([0,2,4,6],8),([0,1,2,3,4,5,6,6],8),(list(range(8))[::-1],8)):
            with self.assertRaisesRegex(ValueError,'full ordered coverage'):
                V.layout(np.array([[4,2,2]],np.int64),indices,8.,total)

    def test_actual_mean_pair_clock_and_all_time_segment_boundaries(self):
        layout=V.layout(np.array([[4,2,4]],np.int64),list(range(8)),8.,8)
        self.assertEqual(layout['paired_timestamps_seconds'].tolist(),[.0625,.3125,.5625,.8125])
        self.assertEqual(layout['teacher_cu_seqlens'].tolist(),[0,8,16,24,32])
        self.assertEqual(layout['tyx'][:8].tolist(),[[0,0,0],[0,0,1],[0,1,0],[0,1,1],[0,0,2],[0,0,3],[0,1,2],[0,1,3]])
        self.assertEqual(float(layout['native_media_positions'][24,0]),.8125)

    def dialogue(self):
        return V.expand_dialogue(np.array([12,55,13,14,15,16],np.int64),
            np.array([0,2,0,0,0,0],np.int64), np.array([0,0,0,0,1,1],np.bool_),video_token_id=55,media_rows=4)

    def test_parent_timestamps_headers_and_terminal_ids_preserved(self):
        d=self.dialogue()
        self.assertEqual(d['input_ids'].tolist(),[[12,V.PLACEHOLDER,V.PLACEHOLDER,V.PLACEHOLDER,V.PLACEHOLDER,13,14,15,16]])
        self.assertEqual(d['native_media_indices'].tolist(),[[1,2,3,4]])
        self.assertEqual(np.flatnonzero(d['assistant_token_mask']).tolist(),[7,8])
        self.assertEqual(np.flatnonzero(d['assistant_prediction_mask']).tolist(),[6,7])
        self.assertEqual(d['parent_to_native_token_index'].tolist(),[0,1,5,6,7,8])

    def test_media_tokens_must_match_actual_count_and_precede_answer(self):
        for types,mask in (([0,2,0,0,0,2],[0,0,0,0,1,1]),([0,0,0,0,0,2],[0,0,0,0,1,0])):
            with self.assertRaises(ValueError):
                V.expand_dialogue(np.array([12,55,13,14,15,16],np.int64),np.array(types,np.int64),
                    np.array(mask,np.bool_),video_token_id=55,media_rows=4)

    def answer(self):
        # Unit-test numbers only; never written to an authenticated teacher cache.
        d=self.dialogue();w=np.array([0,0,0,1,1,0],np.float32)
        return d,{'input_ids':d['parent_input_ids'],'assistant_token_mask':d['parent_assistant_token_mask'],
            'next_ids':np.array([55,13,14,15,16,0],np.int64),'ce_w':np.zeros(6,np.float32),'eos_w':np.zeros(6,np.float32),
            'q_w':w,'q_ids':np.tile(np.arange(64,dtype=np.int64),(6,1)),
            'q_lp':np.tile(np.arange(64,dtype=np.float32)*-.01-8,(6,1))}

    def test_answer_k64_rebased_once_on_native_prediction_positions(self):
        d,a=self.answer();result=V.rebase_answer(a,d)
        self.assertEqual(np.flatnonzero(result['q_w']).tolist(),[6,7])
        np.testing.assert_array_equal(result['q_lp'][6:8],a['q_lp'][3:5])
        self.assertEqual(result['next_ids'][6:8].tolist(),[15,16])
        self.assertFalse(result['q_w'][1:5].any())

    def test_wrong_assistant_shift_or_authentic_target_scope_refused(self):
        d,a=self.answer();a['q_w']=np.array([0,0,0,0,1,1],np.float32)
        with self.assertRaisesRegex(ValueError,'ownership differ'):
            V.rebase_answer(a,d)
        with self.assertRaisesRegex(ValueError,'genuine sealed temporal'):
            V.require_training_targets(feature_manifest=None,answer_manifest=None,
                source_id='unit',source_row_sha256='a'*64,asset_sha256='b'*64)

    def test_invalid_topk_duplicates_and_probability_mass_are_refused(self):
        for kind in ('duplicates','mass'):
            d,a=self.answer()
            if kind=='duplicates':
                a['q_ids'][3]=0
            else:
                a['q_lp'][3]=0
            with self.assertRaisesRegex(ValueError,'ownership differ'):
                V.rebase_answer(a,d)

    def test_visual_answer_projection_preserves_physical_clock_without_audio_labels(self):
        event={'color':'red','frame':1,'shape':'circle','start_seconds':.125,'x':25,'y':42,'frequency_hz':1400,'kind':'tone'}
        truth={'count':1,'frame_count':8,'events':[event]}
        answer=S.visual_answer(truth)
        self.assertNotIn('frequency_hz',answer['events'][0])
        self.assertEqual(answer['events'][0]['frame'],1)
        self.assertEqual(answer['events'][0]['start_seconds'],.125)
        truth['events'][0]['start_seconds']=.75
        with self.assertRaisesRegex(ValueError,'frame/onset clock'):
            S.visual_answer(truth)


if __name__=='__main__':
    unittest.main()
