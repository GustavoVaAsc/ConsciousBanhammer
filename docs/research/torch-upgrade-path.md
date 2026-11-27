# Research: moving from torch 2.11 to a newer torch (as of 2026-11-25); does a newer torch drop setuptools<82?
Answer: Yes. torch 2.13.0 (2026-07-08) and later require only `setuptools>=77.0.3`, but 2.12.x still pins `setuptools<82`. Moving to 2.13 or later also means leaving cu128 (not published from 2.12), so the index URL and the CUDA base image change. Blackwell sm_120 then runs on cu130 or cu132 wheels with driver 580.65.06 or newer, which your driver 580 probably meets.

## Findings
- **F1 [high] 2.13.0 and 2.15.0 metadata say `setuptools>=77.0.3`. 2.12.1 still says `setuptools<82`.**
  Sources: [PyPI torch 2.15.0 JSON](https://pypi.org/pypi/torch/2.15.0/json), [PyPI torch 2.13.0 JSON](https://pypi.org/pypi/torch/2.13.0/json), [PyPI torch 2.12.1 JSON](https://pypi.org/pypi/torch/2.12.1/json)
  Support: direct (2.14.x not checked; it is almost certainly the same, but that is an inference).
- **F2 [medium] No 2.11.x patch relaxes the pin.** `https://pypi.org/pypi/torch/2.11.1/json` returned 404, so 2.11.0 appears to be the only 2.11 release on PyPI. I did not check the index listing.
  Sources: [PyPI torch 2.11.1 JSON (404)](https://pypi.org/pypi/torch/2.11.1/json)
  Support: interpretation
- **F3 [high] Releases after 2.11.0 (dates are as displayed on the GitHub releases page):**
  - 2.12.0: 13 May 2026.
  - 2.12.1: 18 Jun 2026 (PyPI upload 17 Jun UTC).
  - 2.13.0: 8 Jul 2026.
  - 2.14.0: 2 Sep 2026.
  - 2.14.1: 30 Sep 2026.
  - 2.15.0: 28 Oct 2026, the latest. Its `requires_python` is `>=3.11`, so Python 3.12 is fine.
  Sources: [GitHub releases](https://github.com/pytorch/pytorch/releases), [PyPI torch 2.15.0 JSON](https://pypi.org/pypi/torch/2.15.0/json)
  Support: direct
- **F4 [high] cu128 is not published from 2.12 onward.**
  - 2.12 blog: "Starting with PyTorch 2.12, the CUDA 12.8 binary wheel is deprecated and will no longer be published as part of the standard release matrix."
  - 2.12 keeps cu126, cu130 (the default) and an experimental cu132.
  - The 2.13 install matrix lists only cu126 and cu130 (plus ROCm 7.1 and 7.2, and CPU).
  - The 2.14.1 matrix lists cu132, ROCm 7.2 and CPU.
  - A September 2026 dev-discuss notice says CUDA 13.0 was removed from CI from the week of 28 Sep. A search snippet says 2.14 is the last release with any CUDA 12.x wheels and that cu126 stops in 2.15. I did not open that notice.
  - 2.15 uses CUDA 13.2 as the default, with 13.4 alongside it. I did not verify the 2.15 index tags (likely cu132 and a cu134-style tag).
  Sources: [2.12 release blog](https://pytorch.org/blog/pytorch-2-12-release-blog/), [previous-versions](https://pytorch.org/get-started/previous-versions/), [GitHub releases](https://github.com/pytorch/pytorch/releases), [dev-discuss CUDA 13.0 removal](https://dev-discuss.pytorch.org/t/removing-cuda-13-0-builds-from-pytorch-ci-cd-starting-the-week-of-sept-28/3447)
  Support: direct for 2.12 to 2.14.1; the 2.15 tags and the cu126 end date are low confidence.
- **F5 [high] Blackwell users are told to use the CUDA 13.0 or newer wheels, with driver 580.65.06 or newer on Linux.** Your driver 580 supports CUDA 13.0, so cu130 (2.12 and 2.13) is the lowest-friction step. cu132 and 13.4 may need a newer driver. I did not check the NVIDIA minimum for CUDA 13.2 or 13.4. I found no explicit sm_120 arch-list statement for the 2.12+ wheels, but Blackwell is the stated target of the cu13x wheels.
  Sources: [2.12 release blog](https://pytorch.org/blog/pytorch-2-12-release-blog/)
  Support: direct for the driver minimum; interpretation for sm_120.
- **F6 [medium] The wheels pull CUDA from PyPI packages.** The 2.15.0 metadata pins `cuda-toolkit[...]==13.2.2`, `nvidia-cudnn-cu13==9.26.0.51` and `nvidia-nccl-cu13==2.30.7`, and `triton~=3.9.0`. 2.12.1 pins CUDA 13.0.2 and cuDNN 9.20. The runtime CUDA libraries therefore ship inside the wheels. The Docker base image's CUDA version matters only for the driver-compatibility check, not for the torch libraries (inference).
  Sources: [PyPI torch 2.15.0 JSON](https://pypi.org/pypi/torch/2.15.0/json), [PyPI torch 2.12.1 JSON](https://pypi.org/pypi/torch/2.12.1/json)
  Support: direct for the pins; inference for the Docker conclusion.
- **F7 [medium] aarch64 cp312 wheels exist on PyPI for 2.15.0 (`manylinux_2_28_aarch64`).** Those PyPI wheels carry the Linux CUDA requires_dist (cuda-toolkit and so on), so an aarch64 install from PyPI may pull CUDA packages. Your explicit `/whl/cpu` index is what avoids that. I did not confirm that `download.pytorch.org/whl/cpu` has `+cpu` aarch64 cp312 wheels for 2.12 to 2.15. The index page fetch was truncated. The install matrices list a "CPU only" option for 2.12 to 2.14.
  Sources: [PyPI torch 2.15.0 JSON](https://pypi.org/pypi/torch/2.15.0/json), [previous-versions](https://pytorch.org/get-started/previous-versions/)
  Support: direct for PyPI; unverified for the CPU index.
- **F8 [medium] Breaking changes relevant to a train/infer codebase (from the GitHub releases summary, which was truncated):**
  - 2.12: CUDA 12.6 is the minimum to build from source. C++20 is required. `nn.functional` distributed ops raise under `torch.compile`. The `torchrun` default port changed. Planned torchcomms changes to ProcessGroups arrive in 2.13+ (eager init, single backend device).
  - 2.13: named tensors removed. NCCL 2.23+ required. CPython 3.13t wheels dropped.
  - 2.14: `torch.cholesky` and `torch.qr` removed. Profiler `use_cuda` removed. Clamp subgradient changes. Custom `new_group()` must accept `backend`.
  - 2.15: nested tensors removed from the default `weights_only` allowlist for `torch.load`. `import torch` cannot be re-executed. NCCL backend rewritten. Checkpoint metadata format changed. FSDP2 gradient dtype change.
  - I found no change to the `torch.load` `weights_only=True` default itself (it has been the default since 2.6).
  Sources: [GitHub releases](https://github.com/pytorch/pytorch/releases), [2.12 release blog](https://pytorch.org/blog/pytorch-2-12-release-blog/)
  Support: interpretation
- **F9 [low] Setuptools advisory fit:** 2.13+ no longer blocks setuptools 83.0.0 or 84.1.0, so both advisories can be cleared by the upgrade (inference from F1).

## Contradictions
- The 2.13 pre-release feature list (secondary) says the CUDA 12.8 and 12.9 builds were removed in 2.13. The 2.12 blog says cu128 was deprecated and dropped from the standard matrix from 2.12. The install matrices agree with the 2.12 blog. I treat cu128 as gone from 2.12 onward, though 2.12.0 might still have a lingering cu128 index. I did not check.

## Missing evidence
- Compatibility statements for transformers 5.24, sentence-transformers 6.3, peft 0.22, trl 1.19 and bitsandbytes 0.50 with torch 2.12 to 2.15. I did not research these (quick depth). No result was fetched.
- The 2.14.0 and 2.14.1 setuptools pins, and which cu tags 2.15 publishes on download.pytorch.org.
- The explicit sm_120 arch list of the cu130 and cu132 wheels.
- aarch64 `+cpu` cp312 wheels on the CPU index for each release.
- The NVIDIA minimum driver for CUDA 13.2 and 13.4.

## Sources
- Kept: PyPI JSON for torch 2.12.1, 2.13.0 and 2.15.0 (https://pypi.org/pypi/torch/2.15.0/json): primary metadata.
- Kept: PyTorch 2.12 release blog (https://pytorch.org/blog/pytorch-2-12-release-blog/): primary, dated 2026-05-19.
- Kept: GitHub releases page and the previous-versions page: primary, but the extracts were truncated.
- Kept (corroboration only): dev-discuss CUDA 13.0 removal notice and search snippets.
- Rejected: third-party summaries (alternativeto, newreleases, wiki mirrors): secondary and redundant.

## Not covered
- Library compatibility, which is the main remaining task. Check each library's pyproject or release notes for torch bounds.
- Check the lock and uv config against the code: `[tool.uv.sources]` and index definitions, the Docker base image, and any `torch<2.12` caps in dependencies.
- Whether the code uses `torch.load` with nested tensors, `torch.cholesky`, `torch.qr` or named tensors.

## Open questions
- [decision] The target version. 2.13 is the smallest jump that drops the setuptools pin and keeps cu130, and it fits driver 580. 2.15 is the latest but needs CUDA 13.2 or newer wheels and has more breaking changes.
- [unverified] Library compatibility with torch 2.13 to 2.15.
- [unverified] aarch64 CPU index wheels for cp312, and the 2.15 cu tags.

## What the upgrade would require (inference)
- Change the torch pin, for example `torch>=2.13,<2.14`.
- Change the explicit index from `.../whl/cu128` to `.../whl/cu130`. cu130 is published for 2.12 and 2.13. For 2.14 and later use the cu132 index.
- Switch the Docker base image from `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04` to a CUDA 13.x image, or keep it, since the wheels bundle their own CUDA libraries. The host driver must be 580.65.06 or newer.
- Drop or raise the `setuptools<82` constraint, then re-lock with `uv lock`.
- Verify `/whl/cpu` has cp312 aarch64 wheels for the chosen version.
- Re-check bitsandbytes 0.50 for a cu13x build.
