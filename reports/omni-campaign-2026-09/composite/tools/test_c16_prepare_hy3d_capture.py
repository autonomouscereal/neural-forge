"""Synthetic receipt mechanics; no native inference, GPU, or real geometry claim."""
import copy
import json
from pathlib import Path
import sys

import pytest
torch = pytest.importorskip("torch")
from safetensors.torch import save_file

sys.path.insert(0,str(Path(__file__).parent))
import c16_prepare_hy3d_capture as tool


def write(path,value):
    path.write_text(json.dumps(value,indent=2,sort_keys=True),encoding="utf-8")


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(tool.capture,"CAPTURE_DIMS",{"tokens":4,"channels":2,"hidden":6,"context_tokens":3,"context_width":2})
    monkeypatch.setattr(tool,"IMAGE_SIZE",8)
    identity = {"fixture_only":"not_actual_source"}
    monkeypatch.setattr(tool.capture,"inspect_source",lambda *a:{"source_identity":identity})
    meshes,qimg = tmp_path/"meshes",tmp_path/"qimg"
    meshes.mkdir(); qimg.mkdir()
    generation = {**tool.generator.FIXED,"steps":3,"guidance_scale":5.,"bounds":1.01,
                  "octree_resolution":8,"query_chunk":64,"capture_sigma":0.5}
    images,qimages,geometries = [],[],{}
    for i,split in enumerate(("fit","fit","held","held")):
        name = f"fixture_{i}"
        image = qimg/(name+".png")
        image.write_bytes(f"fixture image {i}".encode())
        ih = tool.capture.digest_file(image)
        images.append({"id":name,"split":split,"seed":100+i,"conditioning_file":"/qimg/"+image.name,"conditioning_sha256":ih})
        qimages.append({"id":name,"split":"holdout" if split=="held" else split,"sha256":ih,"file":image.name})
    qfile = qimg/"parent_images_receipt.json"
    write(qfile,{"schema":"neural_forge.qimg_parent_images.v1","status":"complete","images":qimages})
    qhash = tool.capture.digest_file(qfile)
    plan = {"schema":tool.generator.PLAN_SCHEMA,"generation":generation,"images":images,
            "qimg_receipt_file":"/qimg/parent_images_receipt.json","qimg_receipt_sha256":qhash}
    pfile = tmp_path/"parent-plan.json"
    write(pfile,plan)
    phash = tool.capture.digest_file(pfile)
    common = {"source_identity":identity,"plan_sha256":phash,"qimg_receipt_sha256":qhash,
              "qimg_receipt_file":"/qimg/parent_images_receipt.json"}
    pre = {**common,"plan":plan,"extra_source_sha256":tool.generator.EXTRA_SOURCE,
           "comfy_source_sha256":tool.generator.COMFY_SOURCE}
    write(meshes/"preregistered-plan.json",pre)
    receipts,objects = [],[]
    for i,row in enumerate(images):
        name = row["id"]
        mesh = meshes/(name+".ply")
        mesh.write_bytes(f"fixture mesh {i}".encode())
        mh = tool.capture.digest_file(mesh)
        geometry = {"schema":tool.capture.GEOMETRY_SCHEMA,"sha256":str(i)*64,"vertices":4,"triangles":4}
        geometries[str(mesh.resolve())] = geometry
        tensors = {"context":torch.zeros(2,3,2).half(),"trajectory_latent_before":torch.zeros(3,1,4,2).half(),
            "trajectory_model_timestep":torch.zeros(3,2).half(),"trajectory_model_output":torch.zeros(3,2,4,2).half(),
            "trajectory_time_token":torch.zeros(3,2,1,6).half(),"trajectory_cfg_output":torch.zeros(3,1,4,2).half(),
            "trajectory_scheduler_timestep":torch.tensor([1000.,500.,0.]),
            "trajectory_sigma_pair":torch.tensor([[1.,.5],[.5,0.],[0.,0.]]),
            "trajectory_latent_after":torch.zeros(3,1,4,2).half(),"final_latent":torch.zeros(1,4,2).half(),
            "step_index_before":torch.tensor([-1,1,2]),"step_index_after":torch.tensor([1,2,3]),
            "field_logits":torch.zeros(1,9,9,9),"conditioning_image":torch.zeros(1,3,8,8),"conditioning_mask":torch.ones(1,1,8,8)}
        tr = meshes/(name+".safetensors")
        save_file(tensors,str(tr))
        rec = {**row,**common,"object_file":"/old-output/"+mesh.name,"object_sha256":mh,
               "geometry_identity":geometry,"generation":generation,"qimg_image":qimages[i],
               "extraction":{"geometry_identity":geometry,"native_winding_reversed":True,"trimesh_process":False},
               "source_parent_generated":True,"folded_child_proven":False,"field_kind":tool.FIELD,
               "trajectory":{"file":tr.name,"file_sha256":tool.capture.digest_file(tr),"bytes":tr.stat().st_size}}
        rfile = meshes/(name+".json")
        write(rfile,rec)
        receipts.append(rec)
        objects.append({k:rec[k] for k in ("object_file","object_sha256","conditioning_file","conditioning_sha256","geometry_identity","seed","split")})
        objects[-1].update(sigma=.5,generation_receipt_file=rfile.name,generation_receipt_sha256=tool.capture.digest_file(rfile))
    monkeypatch.setattr(tool.capture,"geometry_identity",lambda p,d:geometries[str(Path(p).resolve())])
    run = {**common,"schema":tool.generator.SCHEMA,"status":"complete","samples":receipts,"generation":generation,
           "extra_source_sha256":tool.generator.EXTRA_SOURCE,"comfy_source_sha256":tool.generator.COMFY_SOURCE,
           "source_parent_generated":True,"folded_child_proven":False,"mesh_sdf_pairs_complete":False}
    write(meshes/"mesh-run.json",run)
    write(meshes/"objects.json",{"schema":"neural-forge.hy3d-capture-objects.v1","objects":objects})
    return {"root":tmp_path,"plan":pfile,"meshes":meshes,"qimg":qimg,"run":run,"objects":objects,"geometry":geometries}


def prepare(f):
    return tool.prepare(f["plan"],f["meshes"],f["qimg"],f["root"]/"prepared")


def test_prepares_exact_four_objects_without_capture(fixture):
    got = prepare(fixture)
    assert got["block_index"] == 0 and got["status"] == "prepared_not_captured"
    assert got["parent_trajectory_used_as_capture_input"] is False
    assert got["dit_probe"] == tool.PROBE and got["folded_child_proven"] is False
    assert [x["seed"] for x in got["samples"]] == [100,101,102,103]
    assert all(x["sigma"] == .5 for x in got["samples"])
    assert got["capture_argv"][-6:] == ["--block","0","--query-chunk","256","--device","cuda:0"]
    assert not (fixture["root"]/"prepared/capture").exists()
    for item in got["samples"]:
        assert Path(item["object_file"]).parent == fixture["meshes"]
    with pytest.raises(ValueError,match="fresh"):
        prepare(fixture)


@pytest.mark.parametrize("fault",["partial","plan","order","seed","sigma","source","geometry","mesh","image","receipt","trajectory","qimg","postprocess"])
def test_bad_source_join_refused(fixture,fault):
    f=fixture; run=copy.deepcopy(f["run"]); objects=copy.deepcopy(f["objects"])
    if fault=="partial": run["status"]="running"
    if fault=="plan": run["plan_sha256"]="a"*64
    if fault=="order": run["samples"].reverse()
    if fault=="seed": objects[0]["seed"]+=1
    if fault=="sigma": objects[0]["sigma"]=.25
    if fault=="source": run["source_identity"]={"wrong":True}
    if fault=="geometry": objects[0]["geometry_identity"]={"wrong":True}
    if fault=="receipt": objects[0]["generation_receipt_sha256"]="e"*64
    if fault=="postprocess": run["generation"]={**run["generation"],"postprocess":"repair"}
    if fault in ("mesh","image","trajectory"):
        folder = f["qimg"] if fault=="image" else f["meshes"]
        suffix={"mesh":"ply","image":"png","trajectory":"safetensors"}[fault]
        with (folder/f"fixture_0.{suffix}").open("ab") as h: h.write(b"changed")
    if fault=="qimg":
        with (f["qimg"]/"parent_images_receipt.json").open("ab") as h:h.write(b" ")
    write(f["meshes"]/"mesh-run.json",run)
    write(f["meshes"]/"objects.json",{"schema":"neural-forge.hy3d-capture-objects.v1","objects":objects})
    with pytest.raises(ValueError): prepare(f)
    assert not (f["root"]/"prepared").exists()


@pytest.mark.parametrize("fault",["time_shape","dtype","index","sigma","final"])
def test_actual_trajectory_contract_refuses_mutation(fixture,fault):
    from safetensors.torch import load_file
    p=fixture["meshes"]/"fixture_0.safetensors"
    tensors=load_file(str(p))
    if fault=="time_shape": tensors["trajectory_time_token"]=tensors["trajectory_time_token"].squeeze(2)
    if fault=="dtype": tensors["context"]=tensors["context"].float()
    if fault=="index": tensors["step_index_before"][1]=0
    if fault=="sigma": tensors["trajectory_sigma_pair"][1,0]=.4
    if fault=="final": tensors["final_latent"].fill_(1)
    save_file(tensors,str(p))
    rec={"file_sha256":tool.capture.digest_file(p),"bytes":p.stat().st_size}
    with pytest.raises(ValueError): tool.trajectory(p,rec,fixture["run"]["generation"],tool.Pins())


def fake_capture(f,plan,monkeypatch):
    root=f["root"]/"capture"
    root.mkdir(); (root/"block-00").mkdir()
    samples,expected=[],[]
    for i,original in enumerate(plan["samples"]):
        item={k:original[k] for k in ("object_sha256","conditioning_sha256","geometry_identity","seed","split")}
        expected.append(item.copy())
        payload=root/f"sample-{i:04d}.safetensors"
        payload.write_bytes(b"synthetic capture payload")
        item.update(requested_sigma=original["sigma"],file=payload.name,file_sha256=tool.capture.digest_file(payload),bytes=payload.stat().st_size)
        write(root/f"sample-{i:04d}.json",item)
        samples.append(item)
    block={"schema":tool.CAPTURE_SCHEMA,"block_index":0,"source_identity":plan["source_identity"],"samples":expected}
    write(root/"block-00/capture.json",block)
    checked={"fixture_native_validator":True}
    monkeypatch.setattr(tool,"validate_capture",lambda *a:checked)
    run={"schema":tool.capture.SCHEMA,"source_identity":plan["source_identity"],"producer":tool.capture.PRODUCER,
         "dit_probe":tool.PROBE,"posterior":"mode","folded_child_proven":False,"samples":samples,"dit_capture":checked}
    write(root/"capture-run.json",run)
    return root,run


def test_post_capture_joins_actual_rows_order_and_sigma(fixture,monkeypatch):
    plan=prepare(fixture)
    root,run=fake_capture(fixture,plan,monkeypatch)
    p=fixture["root"]/"prepared/capture-plan.json"
    result=tool.verify_capture(p,tool.capture.digest_file(p),root)
    assert result["status"]=="source_capture_lineage_verified_not_quality"
    assert result["folded_child_proven"] is False
    run["samples"][0]["requested_sigma"] = .25
    write(root/"capture-run.json",run)
    with pytest.raises(ValueError,match="sample/sigma"):
        tool.verify_capture(p,tool.capture.digest_file(p),root)


def test_post_capture_rechecks_parent_artifacts(fixture,monkeypatch):
    plan=prepare(fixture)
    root,_=fake_capture(fixture,plan,monkeypatch)
    p=fixture["root"]/"prepared/capture-plan.json"
    with (fixture["qimg"]/"fixture_0.png").open("ab") as h:h.write(b"changed")
    with pytest.raises(ValueError,match="hash mismatch"):
        tool.verify_capture(p,tool.capture.digest_file(p),root)
