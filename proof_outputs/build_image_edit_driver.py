from pathlib import Path
import hashlib
own=Path(__file__).resolve().parents[1]
root=own.parent/'chimera-training-recovery-oct13-eae1b72e2c3f'
source=root/'experiments/chimera_recovery/train_geometry.py'
text=source.read_text(encoding='utf-8')
# Independent driver: the proven source/rate/runtime/checkpoint policy is reused,
# while all target math, loader and ownership below are specific to image edits.
text=text.replace('geometry','image_edit').replace('Geometry','ImageEdit').replace('HY3D','QIMG').replace('hy3d','qimg')
text=text.replace('from . import train_media as T, consolidation_objectives as O','from . import train_media as T, image_edit as E')
text=text.replace('OBJECTIVE = O.GEOMETRY','OBJECTIVE = E.OBJECTIVE')
text=text.replace('Canonical sigma and native progress stay separate;\nImageEditFlowAdapter reconstructs genuine pixels/targets before every backward.',
 'The genuine before image conditions output rows;\nImageEditAdapter reconstructs genuine input/target/tokenizer proofs before backward.')
text=text.replace('STAGE = "D3b_image_edit_supervised_flow"','STAGE = E.STAGE')
text=text.replace('ImageEditFlowEngine','ImageEditEngine')
text=text.replace('ImageEditFlowAdapter','ImageEditAdapter')
text=text.replace('def image_edit_module():\n    from . import image_edit_flow\n    return image_edit_flow','def image_edit_module():\n    return E')
begin=text.index('def cache_manifest(');end=text.index('\n\nclass ImageEditEngine',begin)
text=text[:begin]+'''def cache_manifest(path, amendment_sha, *, aliases=None):
    """Bind all actual source/sigma pairs before model or optimizer allocation."""
    path=Path(path);data=T.read_json(path)
    if (data.get('schema')!=E.SCHEMA or data.get('stage')!=STAGE or data.get('objective')!=OBJECTIVE or
            data.get('sealed') is not True or data.get('complete_source_traversal') is not True or
            data.get('amendment_sha256')!=amendment_sha or data.get('adapter_sha256')!=I.sha(Path(E.__file__)) or
            data.get('dependencies')!=E.dependency_pins() or data.get('sigma_grid')!=E.SIGMAS or
            data.get('guided_teacher') is not False or data.get('native_parent_full_parity') is not False):
        raise Error('image editing needs the complete selected source-authenticated supervised cache')
    seen,pairs,selected=set(),set(),set()
    for item in data.get('batches',[]):
        if not item.get('id') or item['id'] in seen or not item.get('recovery_group_id'):
            raise Error('editing batch source/group identity missing or duplicated')
        seen.add(item['id']);I.pinned(item['tensors'],path.parent)
        r=T.read_json(I.pinned(item['teacher_receipt'],path.parent))
        if (r['source_id']!=item['source_id'] or r['recovery_group_id']!=item['recovery_group_id'] or
                r['tensors_sha256']!=item['tensors']['sha256'] or r['storage']!=item['storage'] or
                r['amendment_sha256']!=amendment_sha or r['source_admission']!=data['source_admission'] or
                r['vae_proof']!=data['vae_proof'] or r['tokenizer_files']!=data['tokenizer_files']):
            raise Error('editing manifest membership differs from its authentic source receipt')
        pair=r['source_id'],r['sigma']
        if pair in pairs:raise Error('editing cannot repeat an actual source/sigma pair')
        pairs.add(pair);selected.add(r['source_id'])
        # Fail before opening any model when source, target, tokenizer or chart changed.
        load_row(item,path.parent,aliases=aliases)
    admission_path=E.N.pinned(data['source_admission'],path.parent,aliases)
    admission=E.N.read(admission_path)
    records=E.N.rows(E.N.pinned(admission['sources']['source'],admission_path.parent,aliases))
    edit_ids={r['id'] for r in records if r['route']=='image_edit'}
    eligible={r['id'] for r in E.N.rows(E.N.pinned(admission['outputs']['admission'],admission_path.parent,aliases))
              if r['id'] in edit_ids and r['train_admitted'] is True}
    if not seen or selected!=eligible or pairs!={(source,sigma) for source in eligible for sigma in E.SIGMAS}:
        raise Error('editing must cover every eligible original source/sigma pair exactly once')
    return data
''' + text[end:]
text=text.replace('and 7 in p.type_ids','and 1 in p.type_ids').replace('hybrid.bands[7]["feature"] != (576, 640) or hybrid.bands[7]["emission"] != (320, 384)',
 'hybrid.bands[1]["feature"] != (512, 576) or hybrid.bands[1]["emission"] != (256, 320)')
text=text.replace('type7','type1').replace('[7]','[1]')
text=text.replace('mask = batch["native_media_loss_weights"]\n            units = O.scalar_flow_count(batch["native_media_type_ids"], mask, {7: self.hybrid.bands[1]})\n            loss = O.image_edit_backward(self.module, self.hybrid, batch, compute, units, 1.)',
 'mask = batch["native_media_loss_mask"]\n            units = int(mask.sum()) * 64\n            loss = E.full_depth_backward(self.module, self.hybrid, batch, compute)')
text=text.replace('or accepted.get("stage") != "D3b.qimg"','or accepted.get("stage") != "D3b_geometry_supervised_flow"')
text=text.replace('actual accepted completed qimg predecessor','actual accepted completed supervised geometry predecessor')
text=text.replace('config["image_edit_amendment"]["sha256"])','config["image_edit_amendment"]["amendment_sha256"], aliases=config.get("source_aliases"))')
text=text.replace('if checked["cache"]["amendment"] != config["image_edit_amendment"]:',
 'if checked["cache"]["amendment_sha256"] != config["image_edit_amendment"]["amendment_sha256"]:')
text=text.replace('optimizer_selection = D.read(config["image_edit_amendment"], root)[0].get("optimizer_scope", {})',
 'selected_amendment=D.read(config["image_edit_amendment"]["selection"],root)[0]\n        if (selected_amendment.get("selected") is not True or selected_amendment.get("amendment_sha256") != config["image_edit_amendment"]["amendment_sha256"]):\n            raise Error("editing needs the exact owner-selected amendment receipt")\n        optimizer_selection = selected_amendment.get("optimizer_scope", {})')
text=text.replace('parity.get("raw_sign_and_native_progress_verified")','parity.get("raw_sign_and_clean_reference_policy_verified")')
text=text.replace('Path(O.__file__), ','')
begin=text.index('def load_row(');end=text.index('\n\ndef train(',begin)
text=text[:begin]+'''def load_row(item,root,*,aliases=None):
    from safetensors.torch import load_file
    cpu=E.M.inflate_native_batch(load_file(str(I.pinned(item['tensors'],root)),device='cpu'),item['storage'])
    receipt=T.read_json(I.pinned(item['teacher_receipt'],root))
    compute=E.ImageEditAdapter(receipt,root,aliases=aliases).bind(cpu)
    return cpu,compute
''' +text[end:]
text=text.replace('context=verified["shared_r16"]["context"])',
 'context=verified["shared_r16"]["context"], aliases=config.get("source_aliases"))')
text=text.replace('export=True, clock=time.monotonic):','export=True, clock=time.monotonic, aliases=None):')
text=text.replace('batch, compute = load_row(item, root)','batch, compute = load_row(item, root, aliases=aliases)')
if ' O.' in text or 'image_edit_flow' in text or 'native_media_loss_weights' in text or 'bands[7]' in text:raise RuntimeError('unconverted geometry-specific assumption')
(own/'experiments/chimera_recovery/train_image_edit.py').write_text(text,encoding='utf-8')
print(hashlib.sha256(source.read_bytes()).hexdigest())
