# Zagitta GPU unlock on the AI server's RTX 3090s: findings and plan (2026-10-21)

Status (updated 2026-10-21 ~13:20 server time): **ARMED for the next cold boot, on BOTH RTX 3090s (GPU0 + GPU4)**, on Cyril's go. Nothing has been rebooted, reloaded or stopped. Section 10 has what changed, the exact power-off steps, how to read the result and the rollback. Sections 0-9 are the original research (status at the time: read-only checks only).
Every number below comes from a cited source or from command output captured on 2026-10-21 ~12:10 server time.

## 0. Verdict

- **What it is:** `github.com/Zagitta/gpu-unlock` is a Linux kernel module that unlocks the **full-rate FP32-accumulate tensor path (BF16/FP16/TF32)** on GeForce GA102/GA104 cards. GeForce Ampere runs these at a reduced rate. It is **not** a P2P, BAR, VRAM, clock, power, vGPU or NVLink unlock.
- **Does it apply to our 3090s?** The cards match: both 3090s are `10de:2204`, which is in the module's ID table, with FLR supported, kernel 7.0 and Secure Boot off. The **driver is the open question.** Upstream has only been tested with drivers 615.71.09 and a modified 610.43.02. The P40s pin this box to the R580 branch, and nobody has reported running the unlock under R580.
- **Where it pays:** GPU4 media (bf16/fp8 diffusion in ComfyUI) and any torch bf16 work on a 3090. It likely does little for the Strata llama-server (IQ3_S quantized, and GPU0 holds only the smaller share of layers). It does nothing for the P40s or the RTX 8000.
- **Recommendation:** run a **canary on GPU4 only**, keeping GPU0 as an unmodified control. Put it inside a planned full power-off, because one is already due: kernel 7.0.0-38 is installed and will boot next. The go/no-go test is a GEMM TFLOPS comparison of GPU4 against GPU0. If R580 re-locks the card, roll back. The cost is one cold boot of the whole box.

## 1. What the unlock is

| Item | Finding | Source |
|---|---|---|
| Repo | `Zagitta/gpu-unlock`, GPL-2.0-only kernel module, "GPU unlock 1.0.0 packaging candidate". 2 commits: "Initial public release" on 2026-09-28 and "Add all GA102 and GA104 device ids based on user reports" on 2026-10-21. 3 open issues, all opened 2026-10-18. No releases. | https://github.com/Zagitta/gpu-unlock , https://github.com/zagitta/gpu-unlock/commits/main |
| Mechanism | The module runs **before** the stock NVIDIA driver binds. It runs FWSEC-FRTS from the card's own VBIOS PROM, then runs a Booter carrying a **V67-signed** payload on SEC2, which opens the feature **PLM**. It writes **speed selectors** SS0=`0x88888888` (README: "full BF16 rate") and SS1=`0x8` (effect untested), restores the stock signature, does a **PCIe FLR**, checks that SS0/SS1 survived the FLR, and leaves GSP halted with WPR2 cleared so the stock driver boots normally. Nothing is flashed. The code reads the PROM, and I saw no flash write in the parts I read. | README; `kernel/gpu_unlock/unlock_610.c` |
| Lineage | The technique is the V67/PLM method from the CMP mining-card unlocker (`cmpunlocker`, used for the CMP 170HX, which patches nvidia-open 610.43.03). Zagitta moved it into a standalone pre-driver module, so the NVIDIA driver is **not** patched. | README; https://github.com/abobasixseven/unlock-cmp-170hx ; background: https://knightli.com/en/2026/07/22/cmp-170hx-80gb-memory-unlock-ai-gpu-buying-risk/ (2026-07-22) |
| Build inputs | Pinned NVIDIA **610.43.03** inputs (`NVIDIA-kernel-module-source-610.43.03.tar.xz` and `nvidia-utils-610.43.03-1-x86_64_v3.pkg.tar.zst`), downloaded over HTTPS and SHA-256 checked. The build extracts `booter_load_ga102.bin`, `gsp_ga10x.bin`, `gsp_boot_ga102.bin` and `gsp_boot_desc_ga102.bin`, and generates `v67-ga102-610.43.03.bin`. These install to `/usr/lib/firmware/gpu-unlock/610.43.03/`, separate from the driver's firmware. | `Makefile` |
| Opt-in | Does nothing unless the kernel command line has `gpu_unlock.targets=all` or a comma list of full PCI addresses. It only accepts GA102/GA104 GeForce/CMP IDs, excluding professional parts. It skips any card that is already bound, powered down or bus-mastering. Unlocks run concurrently "in about a second". | README; `main.c` |
| Cold boot | **Requires a cold card.** Power off; do not reboot. A warm reboot is "expected to fail and leave the card unchanged". If WPR2 is already enabled, it refuses with "card already initialised since power on". Loading the module after `nvidia` has bound does nothing. | README; `unlock_610.c` |
| Failure mode | If the card's firmware never stops, the module logs "firmware did not stop; keeping its DMA buffers until power off". The card then stays held until the next power-off. | `unlock_610.c` |
| Tested | Author: one RTX 3090 (with a second 3090 as unmodified control), CachyOS 7.2.8, driver **615.71.09**. Issue #1: a 3090 and a 3090 Ti on **Ubuntu 7.0.0-38-generic** with a **modified NVIDIA 610.43.02**, a custom pre-udev initramfs hook, and no Xid. Issue #3: a Supermicro server on kernel 6.14 needed a `pci_set_power_state(D0)` fix; a fork comment says kernels 6.16 and later do not need it. | README; issues #1 and #3 |

### Reported gains (no independent benchmark exists yet)

| Report | Card | Numbers | Source / date |
|---|---|---|---|
| Dense 8192^3 GEMM, FP32 accumulate, 450 W, 5-sample medians | RTX 3090 **Ti** | FP16 87.46 -> 118.39 TFLOPS; BF16 87.50 -> 157.41; TF32 42.58 -> 82.61 | https://github.com/Zagitta/gpu-unlock/issues/1 (2026-10-18) |
| torch matmul | not stated (Supermicro server) | "90 up from 60" for fp16; after: float32 22.5, float16 90.8, bfloat16 97.8 TFLOPS | https://github.com/Zagitta/gpu-unlock/issues/3 (2026-10-18) |
| comment | not stated | "Faster BF16/F16 operations. About 40% gain." | https://github.com/Zagitta/gpu-unlock/issues/2 (2026-10-18) |
| ComfyUI | RTX 3090 | image gen ~33% faster with fp8 and bf16 models; int8 model ~8%; llama.cpp tok/s "roughly the same" (no before numbers were taken) | https://lemmy.dbzer0.com/post/77202260 (posted ~2026-10-19) |
| Background: why there is headroom | RTX 3090 | cuBLAS FP16 with FP32 accumulate measured 68.96 TFLOPS at 1024^2; the FP16-accumulate workaround gave 1.29-1.33x | https://toldyoufools.substack.com/p/make-gpus-30-faster-with-f16f16f16f16 (2025-03-26) |

## 2. Our server (read-only checks, 2026-10-21)

- **Host:** Supermicro X10DRG-OT+-CPU, BIOS 3.2 (2019-12-13), dual Xeon E5 v4 (QPI between sockets). Kernel **7.0.0-30-generic** is running. **7.0.0-38-generic is installed** with nvidia DKMS built, and GRUB_DEFAULT=0, so the next boot changes kernel. Last boot was 2026-09-02.
- **Driver:** **580.178.04 proprietary** (`modinfo nvidia`: `license: NVIDIA`; Ubuntu packages `nvidia-driver-580`, `nvidia-dkms-580` 580.178.04-0ubuntu0.24.04.1). GSP firmware 580.178.04 is loaded on both 3090s and the RTX 8000. On 2026-04-06 the `-open` driver was installed and purged. The P40s (Pascal) need the proprietary module, and **R580 is the last branch that supports Pascal** (https://www.phoronix.com/news/NVIDIA-580-Linux-Driver-Last-HW, 2025-07-01). This box therefore cannot move to the 610/615 drivers upstream tested.
- **Secure Boot:** disabled. **IOMMU:** on, in translated mode (groups present, domain type `DMA-FQ`). Kernel cmdline: `quiet splash` only.
- **No apt holds** on nvidia packages, and the unattended-upgrades Package-Blacklist is empty. The 2026-10-10 API mismatch (userspace 580.178.04 against module 580.173.02) is in the kernel log.

| GPU | Model | PCI | IDs | UUID | Link cap / now | BAR1 | NUMA |
|---|---|---|---|---|---|---|---|
| 0 | RTX 3090 (HP OEM `103c:88d5`, vBIOS 94.02.42.40.97) | 0000:04:00.0 | 10de:2204 | GPU-4ead7f3a-cc79-fa5b-de42-ef024e27f561 | Gen3 x16 / Gen1 x16 (idle) | **256 MiB** (ReBAR max 256 MB) | 0 |
| 1 | Tesla P40 | 0000:05:00.0 | 10de:1b38 | GPU-c95fb5b3-f97c-bfb2-9760-79fe107ffe6d | Gen3 x16 / Gen3 x16 | 32 GiB | 0 |
| 2 | Tesla P40 | 0000:08:00.0 | 10de:1b38 | GPU-a92b0dfe-83f8-11ff-db0d-c65921f365b4 | Gen3 x16 | 32 GiB | 0 |
| 3 | Tesla P40 | 0000:09:00.0 | 10de:1b38 | GPU-08b615f9-8dbe-3edd-d59f-f6ebe8212ca8 | Gen3 x16 | 32 GiB | 0 |
| 4 | RTX 3090 (HP OEM `103c:88d5`, vBIOS 94.02.42.40.97) | 0000:84:00.0 | 10de:2204 | GPU-2ce6a95c-7b1f-644a-3808-d781732a0582 | Gen3 x16 / Gen1 x16 (idle) | **256 MiB** | 1 |
| 5 | Tesla P40 | 0000:85:00.0 | 10de:1b38 | GPU-ce4a1da0-baa8-3d62-419c-a28f6565610f | Gen3 x16 | 32 GiB | 1 |
| 6 | Quadro RTX 8000 | 0000:88:00.0 | 10de:1e78 | GPU-f5a09535-fc85-a027-50f7-734b344747dd | Gen3 x16 | 32 GiB | 1 |

- **Both 3090s support FLR** (`FLReset+`, reset_method `flr bus`), which the module requires. Boot VGA is the ASPEED (`boot_vga=1` on 0000:0c:00.0) and `/proc/fb` shows only `astdrmfb`, so there is no firmware framebuffer on the 3090s to tear down. Xorg (slim) is configured on `PCI:4:0:0` (GPU0) and starts after `nvidia`, so it is unaffected.
- **Current initramfs** (7.0.0-38) contains the nvidia 580 firmware and modprobe confs but **not `nvidia.ko`**. A `gpu_unlock` module placed in the initramfs therefore loads before `nvidia` binds. **Corrected in 10.5: this is wrong.** The 7.0.0-38 initrd does contain `nvidia.ko` (only the running 7.0.0-30 one does not), and udev binds nvidia in init-top before `modules.d` is read.
- **Topology** (`nvidia-smi topo -m`): GPU0-GPU1 PIX, GPU0-GPU2/3 PHB, GPU2-GPU3 PIX, GPU4-GPU5 PIX, GPU4/5-GPU6 PHB, and everything across sockets is SYS. That includes **GPU0<->GPU4 (3090<->3090)** and **GPU0<->GPU6 (the Strata pair)**.
- **P2P** (`nvidia-smi topo -p2p r` and `w`): **OK only among the P40s GPU1/2/3**. Every pair involving a 3090 or the RTX 8000 is NS (not supported) or TNS (topology not supported). NVLink is NS everywhere.
- **Running GPU work:** `llama-server` (Strata IQ3_S, `-sm layer -ts 16,32`, GPU0+GPU6, `-fa on -ctk q8_0 -ctv q8_0`, pid 2743302, in a user session scope, no systemd unit yet); `teacher-score` on GPU1/2/3/5 (pid 1183917); Xorg on all GPUs. GPU4 is idle (15 MiB). All ComfyUI containers have exited.
- **Tools on the box:** no `p2pBandwidthLatencyTest` and no `nvcc`. Torch 2.11.0+cu128 is in `/home/cereal/model-surgery-lab/venv`. `llama-bench` is in `/home/cereal/omni-campaign/export/llamacpp/bin/`. `/dev/ipmi0` exists, but `ipmitool` is not installed.

### P2P: what this unlock does (nothing) and why a P2P unlock is not viable here

The Zagitta module does not touch BAR1, P2P or the P2P capability tables. It also never matches the P40s (`1b38`) or the RTX 8000 (`1e78`), because they are not in its ID table. With explicit `targets` they are never even considered. The P40 P2P silent-corruption finding is **unchanged** by it, so keep peer access disabled for the P40 lanes (for example `NCCL_P2P_DISABLE=1`).

A real P2P unlock for the 3090s, such as the tinygrad `open-gpu-kernel-modules` P2P branch (`565.57.01-p2p`, which lists 2204/2203 in its ID table; https://github.com/tinygrad/open-gpu-kernel-modules), is not viable on this box for three reasons:

1. It needs the **open** kernel modules, which do not support the P40s. One NVIDIA module serves every card, and the 2026-04-06 move to `-open` was purged.
2. Our HP 3090 vBIOS exposes only a **256 MiB BAR1** (ReBAR max 256 MB), and that kind of P2P path relies on a large BAR.
3. The 3090<->3090 and 3090<->RTX 8000 pairs are **cross-socket (SYS over QPI)**. The 3090<->RTX 8000 pair is also two different architectures (GA102 and TU102).

How much P2P would matter for our workloads:

- **llama.cpp `-sm layer`:** at each split boundary only a small activation crosses between cards, so P2P barely matters.
- **`-sm row`** and **torch tensor or data parallel:** P2P matters more. NCCL already falls back to host-staged transfers here, and the SYS path caps any gain anyway.

## 3. Compatibility with our exact stack

| Requirement | Ours | Status |
|---|---|---|
| GA102/GA104 GeForce ID | 10de:2204 x2 | OK (in table) |
| Board | HP OEM 3090 `103c:88d5` | **Unverified upstream** ("Other RTX 3090 boards ... have not been verified") |
| FLR | FLReset+ | OK |
| Kernel | 7.0.0-30 now, 7.0.0-38 next boot | OK: issue #1 ran on Ubuntu 7.0.0-38; the 6.14 power-state bug is not relevant |
| Initramfs | initramfs-tools 0.142ubuntu25.8; nvidia.ko not in initrd | Packaged drop-in should order correctly, but **verify**: issue #1 used a custom pre-udev hook on Ubuntu |
| NVIDIA driver after unlock | **580.178.04 proprietary** | **Untested upstream.** The module's evidence stops at the FLR ("Register values after NVIDIA binds cannot reconstruct this state"). If R580's GSP boot resets SS0/SS1, the unlock is a silent no-op. This is the go/no-go measurement. |
| Secure Boot | off | OK |
| IOMMU | on, DMA-FQ | Unknown whether upstream tested with translation on. The module sets 32-bit DMA masks. Watch for DMAR faults. |
| Cold boot | Remote box, BMC present | Needs a full power-off and a BMC power-on. Confirm the BMC is reachable **before** shutting down. |

## 4. Expected benefit for our workloads

| Workload | Card | Expected | What to measure (before -> after) |
|---|---|---|---|
| ComfyUI media (Qwen-Image 2.1, H3 SOL, LTX; bf16/fp8 DiT, with fp8 upcast on Ampere) | GPU4 | The largest gain. The one ComfyUI report is ~33% on fp8/bf16 image models (Lemmy). The GEMM ceiling comes from issue #1 (3090 Ti BF16 87.50 -> 157.41 TFLOPS). | Fixed Qwen-Image workflow s/it and wall time; fixed H3 SOL clip wall time |
| Torch bf16 training/distill on a 3090 (D2 ran on GPU0/4) | GPU0/4 | Scales with the GEMM share of step time. Upper bound is the GEMM numbers above. | step time / tokens per s on a fixed 50-step run |
| Strata llama-server (IQ3_S, MMQ int8 path, `-sm layer -ts 16,32`) | GPU0 + GPU6 | Small or none. Quantized matmuls do not use the unlocked path (Lemmy: llama.cpp "roughly the same"). GPU0 holds only the smaller share of layers, and the RTX 8000 is not unlocked. | prompt tok/s and decode tok/s from `/completion` `timings`, 5-run medians |
| P40 teacher-score / training | GPU1/2/3/5 | **None** (Pascal: not in the ID table, no tensor cores) | none, beyond a sanity check that throughput is unchanged |
| RTX 8000 | GPU6 | **None** (TU102, not in the ID table) | none |

## 5. Installation plan (each step needs Cyril's go; nothing here has been run)

**Phase A: baseline (no downtime)**
1. Run the GEMM rate and correctness scripts (section 6) on **GPU4**, which is idle now. Run them on GPU0 when Strata is next restarted.
2. Take a Strata `timings` baseline through the live server: same prompt and settings, 5 runs.
3. Run `llama-bench` on GPU4 (`CUDA_VISIBLE_DEVICES=4`) with one F16/BF16 GGUF and one K-quant GGUF already on disk, each of which fits in 24 GB (`-p 2048 -n 128 -r 5`).
4. Take a ComfyUI baseline on GPU4: a fixed Qwen-Image workflow with fixed seed and steps, recording s/it.

**Phase B: build (no downtime; this downloads code and NVIDIA archives, so it needs approval)**
5. Clone `Zagitta/gpu-unlock` at commit `bb1088b` (or a reviewed later commit) into `/home/cereal/src/gpu-unlock`. Read `unlock_610.c`, `main.c` and `debian/` before building.
6. Run `make deb`. This downloads the pinned NVIDIA 610.43.03 inputs and checks their SHA-256. `sudo apt-get build-dep` installs build dependencies, which is a system package change. The output goes to `build/deb/`.

**Phase C: install, canary GPU4 only (no downtime until step 10)**
7. Run `sudo apt-get install ./build/deb/gpu-unlock-dkms_*.deb ./build/deb/gpu-unlock-firmware_*.deb`. DKMS builds it for both 7.0.0-30 and 7.0.0-38. Confirm with `dkms status`.
8. Confirm the initramfs contains it: `lsinitramfs /boot/initrd.img-7.0.0-38-generic | grep -E 'gpu_unlock|gpu-unlock/610.43.03'`. If it is missing, run `sudo update-initramfs -u -k all` and check again.
9. Edit `/etc/default/grub` to set `GRUB_CMDLINE_LINUX_DEFAULT="quiet splash gpu_unlock.targets=0000:84:00.0"`. For the trial, also set `GRUB_TIMEOUT_STYLE=menu` and `GRUB_TIMEOUT=5` so the line can be edited from the BMC console. Run `sudo update-grub`. **GPU0 stays as the control.**
10. **Downtime window** (the whole box, all 7 GPUs):
    - stop Strata `llama-server` (pid 2743302, or `strata-llama.service` once that unit exists);
    - let `teacher-score` finish or checkpoint it, then stop it;
    - make sure no ComfyUI or training container is running;
    - post a Relay notice;
    - check BMC reachability;
    - `sudo systemctl poweroff`;
    - wait until the chassis reports off, then power on from the BMC (web UI or `ipmitool chassis power on` from another host).

    The upstream unlock itself takes "about a second". POST and service restart time on this box has not been measured, so time it.
11. After boot, run:
    - `journalctl -k -b | grep gpu_unlock`, which must show `unlocked` for 0000:84:00.0;
    - `lspci -nnk -s 0000:84:00.0`, which must show `Kernel driver in use: nvidia`;
    - `nvidia-smi -L`, which must list all 7 GPUs;
    - `journalctl -k -b | grep -Ei 'xid|DMAR'`, which must be empty.

    Restart Strata and the P40 lanes.
12. Run section 6. **Go:** GPU4 BF16/FP16/TF32 TFLOPS is materially above GPU0, and correctness passes. **No-go:** GPU4 equals GPU0, meaning R580 re-locked or the unlock failed. In that case roll back (section 7).

**Phase D: add GPU0 (only if Phase C passes and llama-bench or torch shows a gain worth having on GPU0)**
13. Set `gpu_unlock.targets=0000:04:00.0,0000:84:00.0`, then `update-grub`. At the next planned cold power-off, repeat steps 10-12 and re-measure Strata `timings`.

**Phase E: protect against unattended driver swaps (recommended regardless of the unlock)**
14. Add `/etc/apt/apt.conf.d/51-no-nvidia-unattended` with `Unattended-Upgrade::Package-Blacklist { "nvidia-.*"; "libnvidia-.*"; "xserver-xorg-video-nvidia-.*"; };`, or `apt-mark hold` the 580 packages, and upgrade the driver by hand in a window. Kernel upgrades are fine: DKMS rebuilds `gpu_unlock` and `nvidia`, and initramfs regeneration keeps the drop-in.

## 6. Verification tests

All runs use `/home/cereal/model-surgery-lab/venv/bin/python`. Only use idle cards, and run each test on the unlocked card and the control card.

**GEMM rate** (`gemm_rate.py <gpu>`): dense 8192^3, 3 warm-up runs, 5 samples of 10 matmuls, median TFLOPS for fp16, bf16, and fp32 with TF32 on.
```python
import sys, torch
d = int(sys.argv[1]); torch.cuda.set_device(d); n = 8192
torch.backends.cuda.matmul.allow_tf32 = True
for dt in (torch.float16, torch.bfloat16, torch.float32):
    a = torch.randn(n, n, device=d, dtype=dt); b = torch.randn(n, n, device=d, dtype=dt)
    for _ in range(3): a @ b
    torch.cuda.synchronize(); r = []
    for _ in range(5):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); [a @ b for _ in range(10)]; e.record(); torch.cuda.synchronize()
        r.append(2 * n**3 * 10 / (s.elapsed_time(e) / 1e3) / 1e12)
    print(d, dt, f"{sorted(r)[2]:.1f} TFLOPS")
```

**Correctness** (`gemm_check.py`):
- Seeded CPU inputs, 2048^3, in bf16, fp16 and tf32.
- Compute on GPU4 and GPU0 and compare: the two results should match bit for bit, or if not, sit within the same error against a CPU fp64 reference.
- Run 100 repeats per GPU and require an identical sha256 of the output bytes.
- Run a 30-minute GEMM soak on GPU4 while logging `nvidia-smi --query-gpu=power.draw,temperature.gpu,clocks.sm --format=csv -l 5`. Pass means no Xid, no DMAR faults, and a stable hash.

**Strata tok/s:** POST `/completion` to 127.0.0.1:18401 with the key read from the key file inside the script (never echoed), a fixed ~2k-token prompt, `cache_prompt:false`, `temperature:0`, `n_predict:256`, 5 runs. Report the median `timings.prompt_per_second` and `timings.predicted_per_second`.

**P2P matrix (regression only):** `nvidia-smi topo -p2p r` and `-p2p w` must be identical to section 2.

For idle-card P2P correctness (asked for; separate from this unlock), `p2p_check.py <src> <dst>` does the following:
- prints `torch.cuda.can_device_access_peer`;
- builds a seeded 256 MiB tensor on `src`;
- copies it with `x.to(dst)`, using torch's peer path when peer access is possible;
- compares sha256 of both copies on the CPU;
- times 20 copies with events to get GB/s.

Run it on the P40 pairs only in a window when they are idle. That reproduces the known corruption; it does not fix it.

## 7. Rollback

- **Stop unlocking:** remove `gpu_unlock.targets=...` from `/etc/default/grub`, run `sudo update-grub`, then **cold** power-off and on. The selector state survives FLR and lasts until power-off, so a warm reboot may leave the card unlocked. For a one-off, edit the line at the GRUB menu through the BMC console (this is why the menu timeout goes up for the trial).
- **Remove fully:** `sudo apt-get purge gpu-unlock-dkms gpu-unlock-firmware`, then `sudo update-initramfs -u -k all`, then confirm `lsinitramfs ... | grep gpu_unlock` is empty, then cold boot.
- **If a card is held** ("firmware did not stop; keeping its DMA buffers until power off"): cold power-off. Then remove that card's address from `targets` before the next boot.
- **Restore** GRUB_TIMEOUT_STYLE and GRUB_TIMEOUT to `hidden`/`0` once the trial ends.

## 8. Risks

1. **R580 is untested upstream.** The P40s pin us to R580, and upstream only proved the unlock under 615.71.09 and a modified 610.43.02. Most likely outcome on failure: no gain, which the GEMM test detects. Unknown: whether an R580 bind on an unlocked card misbehaves. Mitigation: canary on one card, plus the correctness and soak tests.
2. **The project is very young:** a first public release 23 days ago, 2 commits, one maintainer, issues opened 3 days ago. Only one 3090 board model is verified, and ours is an **HP OEM** board. SS1's effect is untested.
3. **Box-wide downtime per change.** Every enable or disable needs a full cold power cycle of all 7 GPUs: Strata, P40 teacher/training lanes, and Xorg. A failed unlock can leave a 3090 held until the next power-off.
4. **The unattended-upgrades driver swap**, which already happened on 2026-10-10 (580.178.04 userspace against the 580.173.02 module). The known fix (stop GPU processes, reload `nvidia`, `systemctl restart nvidia-cdi-refresh.service`) reloads the module on a **warm** card. `gpu_unlock` does not re-run then. The selectors survive FLR per the source, but nobody has verified that they survive an `nvidia` unload and reload, so re-run the GEMM test after any reload. A cold boot re-applies the unlock. Phase E removes this trigger.
5. **P40s and RTX 8000:** they are not touched, because they are not in the ID table and are not listed in `targets`. They are exposed only through the shared cold boot. The P40 P2P corruption is unchanged.
6. **IOMMU translated mode** may differ from the upstream test box. Watch for DMAR faults on first boot.
7. **Power and thermals:** higher tensor throughput under the same 350 W cap may raise sustained draw and temperature on GEMM-heavy jobs. This has not been measured on a 3090, so log it during the soak.
8. **Terms:** this is a firmware-exploit-based bypass of NVIDIA's product segmentation. The generated payload derives from the CMP reference, and NVIDIA's firmware stays under NVIDIA's terms (README licensing section). That is Cyril's call for his own hardware.

## 9. Sources

- https://github.com/Zagitta/gpu-unlock (README; `Makefile`; `kernel/gpu_unlock/main.c`, `unlock_610.c`); commits 2026-09-28 and 2026-10-21
- https://github.com/Zagitta/gpu-unlock/issues/1 , /issues/2 , /issues/3 (all 2026-10-18)
- https://lemmy.dbzer0.com/post/77202260 (~2026-10-19)
- https://toldyoufools.substack.com/p/make-gpus-30-faster-with-f16f16f16f16 (2025-03-26)
- https://github.com/abobasixseven/unlock-cmp-170hx (2026, no dates shown); https://knightli.com/en/2026/07/22/cmp-170hx-80gb-memory-unlock-ai-gpu-buying-risk/ (2026-07-22)
- https://github.com/tinygrad/open-gpu-kernel-modules (branch `565.57.01-p2p`; older project, used here for comparison only)
- https://www.phoronix.com/news/NVIDIA-580-Linux-Driver-Last-HW (2025-07-01)
- Server output: `nvidia-smi`, `nvidia-smi topo -m`/`-p2p r|w|n|a|p`, `nvidia-smi -q -d MEMORY`, `lspci -nn/-tv/-vv`, `modinfo nvidia`, `/proc/driver/nvidia/params`, `dpkg -l`, `dkms status`, `/etc/apt/apt.conf.d/*`, `lsinitramfs`, `/sys/bus/pci/devices/*/boot_vga`, `journalctl -k -b` (2026-10-21)

## 10. Armed for the next cold boot: both RTX 3090s (2026-10-21)

Cyril, 2026-10-21: *"[Cyril quote withheld]"*

This folds Phase D into Phase C. Both 3090s unlock at the same boot, so there is no unlocked/control pair on the box. The controls are now:
- a pre-unlock baseline of the same benchmark on each card, captured whenever the card is idle before the power-off (10.8);
- the absolute locked ceiling of an RTX 3090 (~71 TFLOPS BF16/FP16 with FP32 accumulate at spec boost; 60-75 measured);
- GPU6 (RTX 8000, stock) as a same-boot reference.

Nothing was rebooted, no nvidia module was loaded or unloaded, and no service or GPU job was stopped.

### 10.1 What Cyril has to do (the only manual step)

1. Optional, as in step 10 of section 5: let `teacher-score` checkpoint and post a Relay notice. The shutdown itself stops Strata, ComfyUI and the workers cleanly.
2. On the server, run `sudo shutdown -h now`.
3. Wait until the chassis is really off. In the BMC web UI at **https://192.168.50.130** (Supermicro IPMI), Remote Control → Power Control should say the host is off. Then wait about 30 more seconds.
4. Power on with BMC Power Control → **Power On**, or with the front button.
   - **A warm restart does not work.** That includes `reboot`, `systemctl reboot`, kexec and the BMC **Reset**: the cards keep power, so the module refuses ("card already initialised since power on") and nothing changes.
   - A BMC **Power Cycle** (off, then on) is cold and would work, but off + on is the unambiguous path.
5. The GRUB menu now shows for 5 s. Do nothing and it boots the default, `Ubuntu` (7.0.0-38) with the unlock.
   - **Boot without the unlock, one time:** pick *Advanced options → Ubuntu, with Linux 7.0.0-30-generic*. Its initrd is unchanged, so gpu_unlock never loads there and the cmdline parameter does nothing. A recovery-mode entry also works, because recovery entries carry no `gpu_unlock.targets`.
   - Alternative: press `e` on the default entry and delete `gpu_unlock.targets=...`.
6. About 2-5 minutes after the boot finishes, read the result (10.8).

### 10.2 Every file changed or added

| Path | What | Backup / note |
|---|---|---|
| `/home/cereal/src/gpu-unlock/` | Clone of `Zagitta/gpu-unlock`, detached at **`bb1088b6b5b8cbdafa1ea69601bd118f7f317158`** (2026-10-21 "Add all GA102 and GA104 device ids"). `git archive` sha256 `abd9e3ce...6d872`. | new dir; only addition is untracked `rollback.sh` |
| `/home/cereal/src/gpu-unlock/rollback.sh` | One-command rollback (10.11) | new |
| `/home/cereal/src/gpu-unlock-ops/` | My scripts: `make_assets.py`, `gpu_bench.py`, `gpu_unlock_verify.sh`, `gpu_unlock_verdict.py`, `gpu_unlock_report.py`, `gpu_unlock_xidwatch.py`, `gpu_unlock_before.py`, `omni_lab_log.py`, `ipmi_lan.py`, `install_system.sh`, the `etc/` sources of every config file below; `logs/` (install logs, `dpkg-selections.bak-20261021.txt`), `results/` (baseline capture) | new dir |
| `/home/cereal/.cache/gpu-unlock/` | `inputs/` (NVIDIA downloads), `run-extract/` (2 files from the .run), `assets.src/` (kernel source tree), `assets/` (the 5 firmware files + license) | new dir |
| `/usr/src/gpu-unlock-1.0.0/` | DKMS source (upstream `make install-dkms`) + `/usr/share/doc/gpu-unlock/README.md` | new |
| `/var/lib/dkms/gpu-unlock/1.0.0/` | DKMS registration (`dkms add`), built for 7.0.0-30 and 7.0.0-38 | new |
| `/lib/modules/7.0.0-30-generic/updates/dkms/gpu_unlock.ko.zst` | sha256 `0e43ad38f7dfa80382d263b8b168f2ac6c7054e2900d0f9fc68272ad8dcb5722`, vermagic 7.0.0-30, signed with the box's MOK key | new; depmod refreshed `modules.*` for that kernel |
| `/lib/modules/7.0.0-38-generic/updates/dkms/gpu_unlock.ko.zst` | sha256 `5b105a6b82f97ce90f78d50987780fe609dce166d5d1610c45d1ea7398ae55bd`, vermagic 7.0.0-38, MOK-signed | new; depmod refreshed `modules.*` |
| `/usr/lib/firmware/gpu-unlock/610.43.03/` | `booter_load_ga102.bin` 056cbc06..., `gsp_boot_desc_ga102.bin` cc5a3fc1..., `gsp_boot_ga102.bin` 3df651ac..., `gsp_ga10x.bin` 57237362..., `v67-ga102-610.43.03.bin` 2a0aa1f8... (all = repo pins) + `/usr/share/licenses/gpu-unlock-firmware/{NVIDIA-LICENSE,NOTICE.md}` | new; separate from the 580 driver's `/usr/lib/firmware/nvidia/580.178.04/` |
| `/usr/share/initramfs-tools/modules.d/gpu-unlock.conf` | Upstream initramfs-tools drop-in (README): puts gpu_unlock + its firmware in the image | new |
| `/etc/initramfs-tools/scripts/init-top/gpu_unlock` | **Added beyond the README**: `modprobe gpu_unlock` before udev (10.5) | new |
| `/boot/initrd.img-7.0.0-38-generic` | Regenerated: 251,584,774 → **324,940,209** bytes | backup `/boot/initrd.img-7.0.0-38-generic.bak-20261021` |
| `/boot/initrd.img-7.0.0-30-generic` | **Not touched** (79,234,196 bytes, 2026-10-10): the no-unlock fallback | - |
| `/etc/default/grub.d/90-gpu-unlock.cfg` | Appends `gpu_unlock.targets=0000:04:00.0,0000:84:00.0` to `GRUB_CMDLINE_LINUX_DEFAULT` (normal entries only) | new; `/etc/default/grub` itself unchanged |
| `/etc/default/grub.d/89-gpu-unlock-trial-menu.cfg` | `GRUB_TIMEOUT_STYLE=menu`, `GRUB_TIMEOUT=5` (step 9 of section 5) | new |
| `/boot/grub/grub.cfg` | `update-grub`: 3 entries carry the targets (default 7.0.0-38, Advanced 7.0.0-38, Advanced 7.0.0-30); recovery entries do not; menu 5 s | backup `/boot/grub/grub.cfg.bak-20261021` |
| `/etc/apt/apt.conf.d/51-no-nvidia-unattended` | Unattended-upgrades blacklist: `nvidia-.*-580`, `libnvidia-.*-580`, `xserver-xorg-video-nvidia-580`, `linux-(modules|objects|signatures)-nvidia-` | new; `50unattended-upgrades` unchanged |
| dpkg selections | `apt-mark hold` on the 16 installed 580 packages (list in 10.7) | selections before: `/home/cereal/src/gpu-unlock-ops/logs/dpkg-selections.bak-20261021.txt` |
| `/usr/local/lib/gpu-unlock-verify/` | Root-owned installed copies of the verify/report/xidwatch scripts + `gpu_bench.py`, `omni_lab_log.py` | new |
| `/etc/systemd/system/gpu-unlock-{verify,report,xidwatch}.service` | Enabled (`multi-user.target.wants` symlinks), **not started** | new |
| `/var/lib/gpu-unlock-verify/` | Status output dir (owner cereal). Holds 4 `test-*` dirs from today's tests; the boot run writes `boot-<boot_id>/` plus `latest.json`/`latest.txt` | new |

### 10.3 Source review of bb1088b (step 1): nothing suspicious

I read every file at the pinned commit: `main.c`, `unlock_610.c` (1,428 lines), `prep_610.c`, `vbios_610.c`, `layout_610.c`, `gfw_eval.c`, all headers, `Kbuild`, the kernel `Makefile` and `dkms.conf`, the top `Makefile`, `debian/*`, `packaging/*` (boot drop-ins, Arch, RPM), `firmware/*.py` and the manifest, and `tests/*` (Python and C, including the shims). The code does what the README says, and only that:

- **Device scope.** It walks `pci_get_device(NVIDIA, ANY)` and filters against the GA102/GA104 GeForce/CMP ID table (2204 is in it). It acts only on cards named in `targets` (or `all`). A card with a driver bound gets `-EBUSY` before anything else, and so does one with bus-master already on. A card that is not in D0, or has no FLR capability, gets `-EOPNOTSUPP`.
- **Hardware writes.** Every `writel` goes to the target card's own BAR0:
  - GSP Falcon at 0x110000 and SEC2 Falcon at 0x840000: engine reset, DMA setup, boot vector, mailboxes, start;
  - the chip-ID echo at 0x84;
  - the two speed selectors at 0x82381c (SS0) and 0x823820 (SS1).

  Config-space writes are only the card's own PCI COMMAND restore and `pci_set/clear_master`. The reset is one `pcie_flr` on that card, followed by `pci_restore_state`.
- **The VBIOS is only read.** It is copied from BAR0+0x300000 (PROM), up to 1 MiB, with `memcpy_fromio`. The FRTS "patch" edits the in-RAM DMA copy of the FWSEC ucode. There is no flash or PROM write anywhere.
- **Warm-boot refusal** (`card already initialised since power on`, WPR2 enabled) and an "already unlocked" check (open PLM → `-EALREADY`) are both in `unlock_transaction_gfw`.
- **Nothing outside the card.**
  - No network, socket or usermode-helper.
  - No file writes: the only file access is `request_firmware` of the 5 pinned names.
  - No sysfs/proc/debugfs, kthreads, timers, kprobes or symbol lookups.
  - No other device is written. `aperture_remove_conflicting_pci_devices` detaches only framebuffers overlapping this card's BARs; our boot VGA is the ASPEED.
  - The only persistence beyond the module is the documented failure case, which pins the module and its DMA buffers until power-off.
- **Build-time code.** `fetch_artifacts.py` is the only network code: HTTPS-only with a pinned SHA-256. The test harness only compiles user-space C against shims and reads files.

### 10.4 Build and provenance (step 2)

The step-2 rule was NVIDIA inputs only from nvidia.com or NVIDIA's GitHub. Upstream takes `gsp_ga10x.bin` and the license text from a **CachyOS repack** (`archive.cachyos.org/.../nvidia-utils-610.43.03-1-x86_64_v3.pkg.tar.zst`). I did not download that. Both files came from NVIDIA directly instead:

| Input | URL | sha256 | Check |
|---|---|---|---|
| Kernel module source | https://download.nvidia.com/XFree86/NVIDIA-kernel-module-source/NVIDIA-kernel-module-source-610.43.03.tar.xz | `7e118923c7a23edc36114d63273a46e3e04e9af98695a42203e7ac2dfe9fc1dc` | = NVIDIA's `.sha256sum` file, = repo pin |
| Driver installer | https://download.nvidia.com/XFree86/Linux-x86_64/610.43.03/NVIDIA-Linux-x86_64-610.43.03.run | `45e2d4c134a23c35e50f253a4aa63e7e5e8d17e3d185d4a07c8a58e9612ed392` | = NVIDIA's `.run.sha256sum` |
| `firmware/gsp_ga10x.bin`, from the .run | (makeself payload, `skip=1022`, zstd; extracted with `tail | zstd -d | tar`; **the installer was never executed**) | `572373620a37418f24dc16b5031c39338778c3257e48e8408de9a57291b24f3a` | = repo pin `GSP_SHA256` (byte-identical to the CachyOS copy) |
| `LICENSE`, from the .run | same | `c692fb14ad4f499c6aee1b9e144e6905148c420e6225b17cb2937366c1e507ad` | = repo pin for the notice |

`make_assets.py` makes the build from these inputs only:
- it runs the repo's own `generate_v67.build_payload()` and `extract_boot_assets.extract_assets()`, so every input and output is checked against the repo's pins;
- the 5 firmware files are therefore byte-identical to an upstream build.

Checks on the result:
- `make test ASSETS=...` passes 5/5, including `test_components`, which runs the whole unlock adapter against mocked hardware with these assets.
- Unprivileged test builds for both kernels are clean: no `undefined` symbols from modpost, only the usual "compiler/pahole differs" notes.
- `apt-get build-dep` / `make deb` were **not** used. The `.deb` route needs the CachyOS file. Installation used the repo's own `make install-dkms install-firmware install-initramfs-tools` and DKMS (AUTOINSTALL=yes: `/etc/kernel/postinst.d/dkms` rebuilds it for any future kernel).

### 10.5 Boot ordering: why there is an extra init-top script (step 4)

The README's initramfs-tools mechanism is the `modules.d/gpu-unlock.conf` drop-in, which I installed. On this box it is **not enough**, for three reasons:

1. **nvidia is inside the initrd.** `nvidia.ko` is in the 7.0.0-38 initrd. It is put there by the `framebuffer-nvidia` hook from `nvidia-kernel-common-580`, which runs because plymouth sets `FRAMEBUFFER=y`. The 7.0.0-30 initrd lacks it only because it was built on 2026-10-10 06:19, before DKMS built nvidia for that kernel at 06:25.
2. **udev binds the cards before `modules.d` is read.** initramfs-tools' `/init` runs `init-top` first, and `init-top/udev` triggers coldplug and settles, which binds nvidia to both 3090s. Only after that does `load_modules` read `modules.d`. gpu_unlock would then find both cards bound and fail with -16 (EBUSY).
3. **Upstream needed the same fix.** Issue #1 (Ubuntu 7.0.0-38) needed "a custom pre-udev initramfs hook" for this reason.

The added `/etc/initramfs-tools/scripts/init-top/gpu_unlock` therefore runs `modprobe gpu_unlock` in init-top, before udev:
- kmod passes `gpu_unlock.targets=` in from `/proc/cmdline`;
- module init blocks until both unlocks finish.

I verified it inside the regenerated initrd (unpacked):
- `scripts/init-top/ORDER` = `all_generic_ide`, `blacklist`, **`gpu_unlock`**, `udev`, `framebuffer`, `console_setup`;
- `conf/modules` = `gpu_unlock`;
- the module (`early3` cpio segment) has sha `5b105a6b...`, the same as the installed one;
- all 5 firmware files are present with the pinned shas;
- nvidia 580.178.04 and its GSP firmware are still present.

There is **no softdep**, as the README advises.

Load test on the running 7.0.0-30 kernel (it does not touch nvidia):
- `modprobe gpu_unlock` with no targets loaded and returned immediately (`targets` = `(null)`);
- `modprobe gpu_unlock targets=0000:05:00.0` (a P40) logged only `gpu_unlock: no GA102/GA104 card at "0000:05:00.0"`;
- both loads were unloaded at once, and GPU memory was unchanged.

### 10.6 Pending kernel and driver (step 3)

- **GRUB default.** `GRUB_DEFAULT=0`, and grubenv is empty (no saved or next entry). Entry 0 is `Ubuntu`: `linux /boot/vmlinuz-7.0.0-38-generic ... quiet splash gpu_unlock.targets=0000:04:00.0,0000:84:00.0`, with `initrd /boot/initrd.img-7.0.0-38-generic`.
- **Modules for 7.0.0-38.** `dkms status` shows `nvidia/580.178.04` and `gpu-unlock/1.0.0` installed for both 7.0.0-30 and 7.0.0-38. For 7.0.0-38, `modinfo` reports `nvidia.ko` version 580.178.04 with vermagic `7.0.0-38-generic`, so no rebuild was needed.
- **Never booted.** 7.0.0-38 has never booted on this box, and neither has any initrd with nvidia in it. See 10.14.

### 10.7 Automatic driver swaps stopped (step 5)

- **Unattended-upgrades.** `apt-config dump` shows the 4 patterns appended to the (empty) `Unattended-Upgrade::Package-Blacklist`. A re-implementation of u-u's `re.match` over the installed packages matches exactly these 16 and nothing else:
  - `libnvidia-cfg1-580`, `libnvidia-common-580`, `libnvidia-compute-580`, `libnvidia-decode-580`, `libnvidia-encode-580`, `libnvidia-extra-580`, `libnvidia-fbc1-580`, `libnvidia-gl-580`;
  - `nvidia-compute-utils-580`, `nvidia-dkms-580`, `nvidia-driver-580`, `nvidia-firmware-580-580.178.04`;
  - `nvidia-kernel-common-580`, `nvidia-kernel-source-580`, `nvidia-utils-580`, `xserver-xorg-video-nvidia-580`.
- **Left out on purpose.** The container toolkit, `nvidia-prime` and `nvidia-settings` are not the driver branch.
- **Holds.** The same 16 packages are `apt-mark hold`.
- **Upgrading the driver by hand, in a window:**
  1. `sudo apt-mark unhold $(apt-mark showhold | grep nvidia)`
  2. Upgrade the driver packages.
  3. `sudo apt-mark hold ...` again.
  4. Reload the module, then re-run the GEMM test: the unlock may not survive an nvidia reload (risk 4 in section 8). A cold boot re-applies it.
- **Kernel updates.** These are still allowed. DKMS rebuilds both modules, and initramfs-tools picks up the init-top script for every kernel.

### 10.8 Post-boot verification (step 6): what runs and where to look

| Unit | When | Does |
|---|---|---|
| `gpu-unlock-verify.service` (root, oneshot, `TimeoutStartSec=300`, `ExecStart=-`, always exits 0) | `After=` udev-trigger/modules-load/local-fs; **`Before=` `docker.service` (ComfyUI `comfyui_qwen21_gpu4` restarts with docker: `unless-stopped`), `strata-llama`, `strata-gate`, `media-gate`, `gen-worker-media`, `gen-worker-text`, `display-manager` (= `slim`, Xorg on GPU0)**. `systemctl show` confirms docker and strata-llama now list it in `After=`. | Saves `/proc/cmdline`, the full boot kernel log, the gpu_unlock lines (`pci 0000:04:00.0: ...` / `pci 0000:84:00.0: ...` / `gpu_unlock: ...`), lsmod, the module's `targets`, `lspci -nnk`, `nvidia-smi -L/-q`; waits ≤ 90 s for 7 GPUs. It then keeps one `nvidia-smi -l 5` client open until it exits. Nothing else holds the cards this early (persistence mode is off, Xorg is ordered after), so without it every query would re-initialise every idle card (GSP boot). No persistent driver setting changes. Runs `gpu_bench.py` as cereal on **GPU0, GPU4, GPU6**: 8192^3 GEMM rate for fp16, bf16, tf32 and strict fp32 (control), 5×10 medians; correctness at 2048^3 seeded CPU inputs for bf16/fp16/tf32 against CPU fp64 **and** fp32 references, 100 repeats each, which must hash identically; GPU0-vs-GPU4 output identity; comparison with the pre-unlock baseline (speedups and output hashes). Then greps Xid/DMAR-fault/fallen-off-bus/RmInitAdapter lines and writes the verdict. |
| `gpu-unlock-report.service` (cereal, oneshot) | after verify **and** docker (omni_lab = `agent-memory-postgres`) | Waits ≤ 10 min for omni_lab, then logs run `probe:gpu-unlock-verify:boot-<boot_id>` with kind=probe, project chimera-omni, owner claude, GPUs 0,4,6; config = the whole status.json; metrics `gpuN_{fp16,bf16,tf32,fp32}_tflops`, `gpuN_*_speedup`, `gpuN_*_relerr`, `gpuN_unlocked_by_module`, `gpuN_correct`, `gpuN_xid_dmar_lines`, `gpu6_*_tflops`, `gpu_count`. Status: pass = GO, fail = NO-GO, unknown = PARTIAL/INCONCLUSIVE. |
| `gpu-unlock-xidwatch.service` (cereal + adm, `Restart=on-failure`) | from boot to boot + 24 h | Follows `journalctl -k -b` (from the start of the boot) for lines naming 0000:04:00 / 0000:84:00: Xid (with code), DMAR fault, fallen off the bus, RmInitAdapter failed, other NVRM. The per-card driver banner is ignored. Each hit goes to `xidwatch.jsonl` at once and to omni_lab as a metric on `probe:gpu-unlock-xidwatch:boot-<boot_id>`, queued while the DB is down. It ends at 24 h with pass (no hits) or fail. |

**Reading the result**

| What | Command |
|---|---|
| One-line verdict | `cat /var/lib/gpu-unlock-verify/latest.txt` |
| Full status | `/var/lib/gpu-unlock-verify/latest.json`; everything else is in `/var/lib/gpu-unlock-verify/boot-<boot_id>/` (`gpu_unlock.log`, `bench.json`, `xid.log`, `verify.log`, `kernel.log`) |
| Module log | `journalctl -k -b \| grep -E 'gpu_unlock\|pci 0000:(04\|84):00.0'`. Success shows, for each card, `... SS0=88888888 SS1=00000008` on the `post-FLR SEC2 reset released` line and `pci 0000:0X:00.0: gpu_unlock: unlocked`. |
| Unit runs | `journalctl -b -u gpu-unlock-verify -u gpu-unlock-report -u gpu-unlock-xidwatch` |
| omni_lab rows | `cd /home/cereal/omni-campaign/lab && set -a && . ~/.config/omni-lab/env && set +a && /home/cereal/.venvs/chimera-intake/bin/python omni_lab.py sql "SELECT id, kind, status, name, notes FROM run WHERE run_key LIKE 'probe:gpu-unlock%' ORDER BY id DESC"` |
| Metrics of one run | `... omni_lab.py sql "SELECT name, step, value FROM metric WHERE run_id = <id> ORDER BY name"` |

**Before-numbers (step 6).** At 13:10 neither card was idle:
- GPU0: Strata `llama-server`, 22,428 MiB;
- GPU4: ComfyUI `comfyui_qwen21_gpu4`, 11-15 GB.

So the same benchmark is now waiting detached: `gpu_unlock_before.py`, pid 3321123, started 2026-10-21 19:10 UTC.
- It checks each card every 10 min for 16 h.
- A card counts as idle at under 2000 MiB with no compute PID; it then benchmarks that card alone (about 30 s).
- It keeps a run only if no other process appeared on the card during it.
- A kept run is written to `/home/cereal/src/gpu-unlock-ops/results/baseline.json` (which the post-boot verify uses), logged to omni_lab as `probe:gpu-unlock-baseline:gpuN:<stamp>`, and recorded in `results/before-capture.log` (every check, busy or idle).
- The cold boot kills it.
- GPU0 hosts Strata permanently (`Restart=always`), so **a GPU0 baseline is unlikely**. The verify then falls back to the absolute threshold for that card (10.9).
- Pre-unlock Xid baseline, from the current 49-day boot: **zero NVRM Xid lines on any card** and zero DMAR lines naming GPU0/GPU4. All DMAR faults name `05:00.0` (P40 GPU1, PTE write faults).

### 10.9 Go / no-go (section 5 step 12, adapted to both cards)

Per card, all four must hold:

| # | Check | Pass condition |
|---|---|---|
| (a) | module | the kernel log says `gpu_unlock: unlocked` for its PCI address |
| (b) | rate | bf16 ≥ **1.25×** its pre-unlock baseline when one exists, otherwise bf16 ≥ **90 TFLOPS**. ≤ 80 TFLOPS absolute, or < 1.1× the baseline, counts as stock. |
| (c) | correct | bf16/fp16/tf32 relative Frobenius error against CPU fp64 under 4e-3 / 1e-3 / 2e-3. Expected about 1.7e-3 / 2e-4 / 3e-4, measured today on a P40. All 100 repeats bit-identical. |
| (d) | clean | no Xid/DMAR/off-bus line for that card |

Overall:
- **GO** = both cards pass and 7/7 GPUs enumerate.
- **NO-GO → roll back** = any correctness, Xid or DMAR failure, or "unlocked by the module but stock rate" (R580 re-locked the card).
- PARTIAL (one card GO) and INCONCLUSIVE carry the reasons in `status.txt`.
- The strict-fp32 control should not move with the unlock; if it does, clocks changed rather than the tensor path.
- The 24 h xidwatch must also end `pass`.

### 10.10 What to expect after power-on

1. POST of the X10DRG with 7 GPUs. Not measured; several minutes is plausible.
2. The GRUB menu for 5 s.
3. The kernel, then init-top: about 1 s of unlock per card, run concurrently.
4. udev, then nvidia binds all 7 cards.
5. `gpu-unlock-verify`, about 1-2 min (torch import plus 3 cards), holds docker, Strata, media-gate, the workers and Xorg. Hard cap 5 min.
6. The services start. Within a few minutes `latest.txt` and the omni_lab probe row exist.

### 10.11 Rollback (step 7)

`sudo /home/cereal/src/gpu-unlock/rollback.sh`:

1. Moves `90-gpu-unlock.cfg`, the init-top script and the modules.d drop-in into `/var/backups/gpu-unlock-rollback-<stamp>/`.
2. Disables the 3 units.
3. Rebuilds the initrd of every kernel that contains gpu_unlock, after backing it up.
4. Runs `update-grub`.
5. Prints that a **full power-off** is needed: the selectors survive FLR and warm reboots.

Options:
- `--restore-menu` also drops the 5 s GRUB menu (back to hidden/0).
- `--purge` also runs `dkms remove --all` and moves `/usr/src/gpu-unlock-1.0.0`, the firmware, docs and licenses out.
- `--dry-run` prints everything and changes nothing; it was run today, both plain and with all flags.

Without root, only `--dry-run` runs. The one-off alternatives are in 10.1 step 5. If a card is "held" (`firmware did not stop`), do a cold power-off, then roll back or remove that card's address from `90-gpu-unlock.cfg`.

### 10.12 BMC (step 8)

- **How it was read.** The BMC LAN settings came over the local IPMI device: `ipmi_lan.py`, a read-only "Get LAN Configuration Parameters" ioctl on `/dev/ipmi0`, with no login and nothing installed.
- **Settings.** Channels 1 and 2: **192.168.50.130**, DHCP, mask 255.255.255.0, gateway 192.168.50.1, MAC ac:1f:6b:89:82:ff.
- **Reachability.**
  - From the workstation: ping 0% loss (2-7 ms), `https://192.168.50.130/` HEAD → 200.
  - From the server: ping 0% loss, HEAD → 200.
- **Advice.** The address is DHCP. A DHCP reservation for that MAC would stop it from moving.

### 10.13 Tests run today

1. `make test`: 5/5 OK.
2. Clean test builds for both kernels.
3. `gpu_bench.py` smoke on the idle P40 GPU5 (n=1024/256), plus `gpu_unlock_verify.sh` in test mode on GPU5:
   - rate path, check path, uuid→device mapping, the visibility pin, verdict and status files all work end to end;
   - this caught and fixed two bugs: cuDNN's import-time SM<7.5 check would have killed the boot benchmark with the P40s visible, and device-side `randn` has no SM 6.1 kernel.
4. Verify in test mode with the real boot GPU set (0,4,6; all busy): busy cards are recorded and skipped without CUDA init.
5. omni_lab write path: smoke rows **37998** (verify report, 7 metrics) and **37999** (xidwatch, read the full 49-day journal in < 30 s, `pass`).
6. `classify()` unit cases: Xid 79/13 on 84/04, DMAR on 84:00.0, init-fail and off-bus → counted. A P40 DMAR, a GPU6 Xid and the driver banner → ignored.
7. `systemd-analyze verify` clean.
8. GRUB config diff = exactly the targets and the menu timeout.
9. The initrd contents and ORDER, as in 10.5.
10. Module load/unload on 7.0.0-30.
11. Rollback dry runs.
12. BMC read and reachability.

### 10.14 Not verified, risks, deliberately left out

**Only the boot can answer these:**
- **Whether R580 (580.178.04) re-locks the cards.** Upstream proved the unlock only under 615.71.09 and a modified 610.43.02. If R580's GSP boot resets SS0/SS1, the module still logs `unlocked` but the rate stays stock. The verdict calls that NO-GO, and we roll back.
- **Whether the HP OEM vBIOS (94.02.42.40.97) parses.** That covers the FWSEC descriptor and the V3 mapper. If it does not, the module logs the failure and leaves the card locked, which is safe.
- **IOMMU translated mode (DMA-FQ)** during the Falcon DMA.
- **Whether both cards unlocking concurrently** behaves the same as upstream's one card.

**Risks of the boot itself, independent of the unlock:**
- **First boot of 7.0.0-38**, and the first boot with nvidia inside the initrd.
- **The large initrd.** The 7.0.0-38 initrd was already 251 MB and is now 325 MB (the 610 `gsp_ga10x.bin` is 84 MB raw). GRUB 2.12-1ubuntu7.3 on this UEFI board has never loaded an initrd this large. If GRUB reports out of memory, pick *Advanced → 7.0.0-30* from the menu (79 MB initrd, stock) and tell me. The fix would be a smaller initrd, for example dropping nvidia from it.

**Other unverified points:**
- **The GPU0 baseline** depends on Strata being down at some point in the next 16 h. GPU4 depends on ComfyUI unloading.
- **The benchmark's rate path on Ampere** has not run on a 3090 today; it did run on the P40. The cuBLAS kernels are the same code path, but the first real 3090 numbers come at boot (or from the baseline capture).

**Deliberately left out:**
- the plan's 30-min GEMM soak: not automated, because it would hold the serving cards after boot;
- the Strata tok/s and ComfyUI baselines (Phase A steps 2-4): they need the serving windows the rules ruled out;
- `make deb` / `apt-get build-dep`: they need the CachyOS input;
- installing `ipmitool`;
- a softdep, as the README says;
- regenerating the 7.0.0-30 initrd: it is kept as the stock fallback.

### 10.15 System-changing commands run (in order, all via `install_system.sh` unless noted)

| Phase | Commands | Result |
|---|---|---|
| prep (cereal) | `git clone` into `/home/cereal/src/gpu-unlock`, checkout bb1088b; 4 NVIDIA downloads into `~/.cache/gpu-unlock/inputs`; extract 2 files from the .run and untar the source; `make_assets.py`; `make test`; `cp rollback.sh` | - |
| `module` | `make install-dkms`; `make install-firmware ASSETS=...`; `make install-initramfs-tools`; `dkms add -m gpu-unlock -v 1.0.0`; `dkms build` + `dkms install -k 7.0.0-30-generic` and `-k 7.0.0-38-generic` | all rc=0; both modules signed and installed; initrds untouched |
| `boot` | `install` init-top script and the two `grub.d` cfgs; `cp -a grub.cfg grub.cfg.bak-20261021`; `update-grub` | rc=0 |
| `initramfs` | `cp -a initrd.img-7.0.0-38-generic ...bak-20261021`; `update-initramfs -u -k 7.0.0-38-generic` | rc=0; no missing-firmware warnings |
| `apt` | `install 51-no-nvidia-unattended`; `dpkg --get-selections > logs/...`; `apt-mark hold` (16 packages) | rc=0 |
| `units` | `install` scripts to `/usr/local/lib/gpu-unlock-verify`; `install -d /var/lib/gpu-unlock-verify`; `install` the 3 units; `systemctl daemon-reload`; `systemd-analyze verify`; `systemctl enable` (3) | rc=0; not started |
| tests (sudo) | `GPU_UNLOCK_VERIFY_TEST=1 gpu_unlock_verify.sh` twice; `modprobe gpu_unlock` / `modprobe -r`, twice; `ipmi_lan.py` | as in 10.13 |
| detached (cereal) | `setsid nohup python3 gpu_unlock_before.py --gpus 0,4 --hours 16 --every 10` | pid 3321123 |

Full outputs are in `/home/cereal/src/gpu-unlock-ops/logs/install-{module,boot,initramfs,apt,units}.log`.
