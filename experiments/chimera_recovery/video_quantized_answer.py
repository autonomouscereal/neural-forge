"""Versioned video-only BF16-storage compatibility, with unchanged target bits.

This is a source-specific rounding-compatible check, not an exact normalization
certificate or a bound on every torch.logsumexp implementation. Native admission
also requires genuine matched stock parity and all original raw source seals.
The inherited bucket_terms loss and its nonnegative tail clamp are unchanged.
"""
import numpy as np
try:
    from . import video_cache as V
except ImportError:
    import video_cache as V

POLICY='video_bf16_interval_plus_recorded_fp32_lse_half_ulp_2026-10-13.v1'


def rounding_compatible(lp,lse):
    lp,lse=np.asarray(lp),np.asarray(lse)
    if (lp.dtype!=np.float32 or lp.ndim!=2 or lp.shape[1]!=64 or lse.dtype!=np.float32 or
            lse.shape!=(lp.shape[0],) or not np.isfinite(lp).all() or not np.isfinite(lse).all() or np.any(lp>0)):
        raise ValueError('finite recorded FP32 LSE and finite nonpositive BF16-decoded K64 required')
    raw=lp.view(np.uint32)
    if np.any(raw&0xffff):raise ValueError('logprobs must retain exact original BF16 representability')
    bits=raw>>16
    # For a negative BF16 value, incrementing its representation gives the
    # adjacent more-negative value. Signed/positive zero have their own edge.
    more=((bits+1)<<16).view(np.float32)
    lower=(lp.astype(np.float64)+more.astype(np.float64))/2
    lower[lp==0]=-float(np.array([0x00010000],np.uint32).view(np.float32)[0])/2
    minimum=np.exp(lower).sum(1)
    half_ulp=np.spacing(np.abs(lse)).astype(np.float64)/2
    envelope=np.exp(half_ulp)
    if np.any(minimum>envelope):
        raise ValueError('impossible top-K mass under selected BF16/recorded-LSE compatibility envelope')
    mass=np.exp(lp.astype(np.float64)).sum(1)
    return {'policy':POLICY,'source_rounding_compatible':True,'exact_normalization_proven':False,
            'library_correct_rounding_proven':False,'stored_mass_max':float(mass.max(initial=0)),
            'stored_mass_over_one_rows':int((mass>1).sum()),'stored_mass_over_1_001_rows':int((mass>1.001).sum()),
            'minimum_interval_mass_max':float(minimum.max(initial=0)),
            'minimum_envelope_margin':float((envelope-minimum).min(initial=np.inf)),
            'target_probabilities_rescaled':False,'stored_bf16_bits_preserved':True,
            'inherited_loss':'original bucket_terms: p=exp(storedLP); p_tail=max(0,1-sum(p)); unchanged',
            'true_kl_or_nonnegative_exact_loss_claimed':False}


def dense_answer(ids,mask,positions,cached_ids,topk_ids,raw_bf16,lse):
    ids,mask,positions,cached_ids,topk_ids,raw_bf16=map(np.asarray,(ids,mask,positions,cached_ids,topk_ids,raw_bf16))
    if ids.ndim!=2 or ids.shape[0]!=1 or ids.dtype!=np.int64 or mask.dtype!=np.bool_ or mask.shape!=(ids.shape[1],):
        raise ValueError('authentic parent IDs and target-indexed assistant mask required')
    length=ids.shape[1];owned=np.zeros(length,np.bool_);owned[:-1]=mask[1:];expected=np.flatnonzero(owned)
    if (not len(expected) or not np.array_equal(positions,expected) or cached_ids.shape!=expected.shape or
            not np.array_equal(cached_ids,ids[0,expected]) or topk_ids.shape!=(len(expected),64) or
            not np.issubdtype(topk_ids.dtype,np.integer) or raw_bf16.dtype!=np.uint16 or raw_bf16.shape!=topk_ids.shape or
            np.any(topk_ids<0) or np.any(topk_ids>=V.TEXT_VOCAB) or any(len(set(row.tolist()))!=64 for row in topk_ids)):
        raise ValueError('actual unique native IDs and once-shifted assistant coverage required')
    lp=(raw_bf16.astype(np.uint32)<<16).view(np.float32);proof=rounding_compatible(lp,lse)
    answer={'input_ids':ids.copy(),'assistant_token_mask':mask.copy(),'next_ids':np.zeros(length,np.int64),
            'ce_w':np.zeros(length,np.float32),'eos_w':np.zeros(length,np.float32),'q_w':owned.astype(np.float32),
            'q_ids':np.zeros((length,64),np.int64),'q_lp':np.zeros((length,64),np.float32)}
    answer['next_ids'][expected]=ids[0,expected+1];answer['q_ids'][expected]=topk_ids;answer['q_lp'][expected]=lp
    return answer,proof


def rebase_answer(answer,ingress,lse):
    keys={'input_ids','assistant_token_mask','next_ids','ce_w','eos_w','q_w','q_ids','q_lp'}
    if set(answer)!=keys:raise ValueError('complete authentic vision_answer_source.v1 tensors required')
    a={k:np.asarray(v) for k,v in answer.items()};ids,mask=ingress['parent_input_ids'],ingress['parent_assistant_token_mask']
    length=ids.shape[1];owned=np.zeros(length,np.bool_);owned[:-1]=mask[1:]
    if (not np.array_equal(a['input_ids'],ids) or not np.array_equal(a['assistant_token_mask'],mask) or
            any(a[k].shape!=(length,) for k in ('next_ids','ce_w','eos_w','q_w')) or
            a['q_ids'].shape!=(length,64) or a['q_lp'].shape!=(length,64) or any(not np.isfinite(v).all() for v in a.values()) or
            a['ce_w'].any() or a['eos_w'].any() or a['q_ids'].dtype!=np.int64 or a['next_ids'].dtype!=np.int64 or
            not np.array_equal(a['q_w'],owned.astype(a['q_w'].dtype)) or np.any(a['q_ids']<0) or np.any(a['q_ids']>=V.TEXT_VOCAB) or
            any(len(set(row.tolist()))!=64 for row in a['q_ids'][owned]) or
            not np.array_equal(a['next_ids'][:-1][owned[:-1]],ids[0,1:][owned[:-1]])):
        raise ValueError('actual source, native vocabulary or assistant ownership differs')
    rounding_compatible(a['q_lp'][owned],lse)
    new_length=ingress['input_ids'].shape[1]
    result={k:np.zeros((new_length,)+v.shape[1:],dtype=v.dtype) for k,v in a.items() if k not in ('input_ids','assistant_token_mask')}
    mapping=ingress['parent_to_native_token_index']
    for parent in np.flatnonzero(owned):
        native=int(mapping[parent+1])-1
        for key in result:result[key][native]=a[key][parent]
    result.update(input_ids=ingress['input_ids'].copy(),assistant_token_mask=ingress['assistant_token_mask'].copy())
    if not np.array_equal(result['q_w']>0,ingress['assistant_prediction_mask']):raise ValueError('native answer shift differs')
    return result
