Optimized Group Relative Policy Optimization for RLHF Inference (O-GRPO)
========================================================================

Basic Information
-----------------

- **Project Name:** Optimized Group Relative Policy Optimization for RLHF
  Inference (O-GRPO)
- **GitHub Repository:** https://github.com/chu-siang/O-GRPO.git
- **About:** Inference-optimized RLHF training using *Group Relative Policy
  Optimization (GRPO)* in Hugging Face TRL, targeted at NVIDIA V100 GPUs, with
  Python-first APIs and optional C++ fused kernels for hot paths. See TRL repo
  and docs for the official GRPO and trainer interfaces.

Problem to Solve
----------------

**Field/Industry.** LLM post-training / alignment (RLHF). TRL provides
PPO/GRPO trainers and reward modeling tools to run RLHF pipelines on top of
Transformers.

**Background.** GRPO = *Group Relative Policy Optimization*, an RL fine-tuning
algorithm supported by TRL as ``GRPOTrainer``. It is highlighted in TRL docs as
a core method for post-training.

**Physics/Mathematics.** The objective maximizes expected reward of generated
completions with a KL regularization to a frozen reference policy
(policy-gradient family; GRPO is PPO-like but uses group-relative advantages).

**Problem.** On **V100 (32GB, FP16/FP32)**, RLHF loops are dominated by repeated
``.generate()`` calls (policy sampling), reward evaluation, and per-token
KL/logprob computation. Goals are to:

1. Reduce generation latency/overheads
2. Keep memory bounded
3. Maintain compatibility with TRL’s ``GRPOTrainer`` API and training loop

Prospective Users
-----------------

- **LLM researchers** benchmarking GRPO on constrained GPUs (V100).
- **HPC/ML engineers** needing reproducible, scalable RLHF pipelines on on-prem
  clusters.
- **Students/practitioners** learning TRL GRPO end-to-end from runnable code
  (Python) and selectively accelerating hot spots (C++ kernels) without breaking
  public APIs.

System Architecture
-------------------

Workflow
~~~~~~~~

1. Dataset yields prompts.
2. Policy (Causal LM) generates completions (batched, FP16, cache-enabled).
3. Reward function/model scores completions (vectorized; ``no_grad``).
4. Reference model forward (for KL/reg); compute per-token logprobs/advantages.
5. ``GRPOTrainer`` applies updates; log metrics.
6. Repeat.

Constraints (V100-aware)
~~~~~~~~~~~~~~~~~~~~~~~~

- Precision: FP16 acceleration; BF16/FP8 not available on V100.
- Memory: 32GB; sequence length typically kept ≤ ~4k; prefer micro-batches +
  frequent steps.
- Stability: ensure generation in ``eval()`` + ``no_grad()``; training in
  ``train()``.

Modularization
~~~~~~~~~~~~~~

- **Policy/Ref/Reward**: HF Transformers models (policy trainable; ref frozen;
  reward eval-only).
- **Trainer**: TRL ``GRPOTrainer`` controls loop.
- **Profiler**: timing + peak VRAM; per-stage metrics (gen/reward/KL/step).
- **Optional C++ extension**: fused sampling (temperature/top-p/repetition) to
  cut kernel launches.

API Description
---------------

**Programming model.** Python-first (TRL + Transformers). TRL provides the
official ``GRPOTrainer`` with public constructor/arguments and README-level
examples; PPO/RewardTrainer APIs are also documented.

Python: Minimal, runnable GRPO (V100-optimized)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # file: train_grpo_v100.py
   import os, time, torch
   from datasets import load_dataset
   from transformers import AutoTokenizer, AutoModelForCausalLM
   from trl import GRPOTrainer
   from contextlib import nullcontext

   MODEL = os.environ.get("MODEL", "Qwen/Qwen2-0.5B")
   MAX_NEW = int(os.environ.get("MAX_NEW_TOKENS", "128"))
   BATCH   = int(os.environ.get("BATCH", "8"))
   MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "512"))

   def reward_unique_chars(completions, **kwargs):
       return [float(len(set(c))) for c in completions]

   def main():
       device = "cuda" if torch.cuda.is_available() else "cpu"
       tok = AutoTokenizer.from_pretrained(MODEL)
       if tok.pad_token is None:
           tok.pad_token = tok.eos_token

       model = AutoModelForCausalLM.from_pretrained(
           MODEL,
           torch_dtype=torch.float16 if device == "cuda" else None,
           device_map="auto" if device == "cuda" else None,
       )

       ds = load_dataset("trl-lib/chatbot_arena_completions", split="train")
       if MAX_SAMPLES > 0:
           ds = ds.select(range(min(MAX_SAMPLES, len(ds))))

       gen_kwargs = dict(
           do_sample=True, top_p=0.9, temperature=0.7,
           repetition_penalty=1.05, max_new_tokens=MAX_NEW, use_cache=True
       )

       trainer = GRPOTrainer(
           model=model,
           reward_funcs=reward_unique_chars,
           train_dataset=ds,
           processing_class=tok,
           gen_kwargs=gen_kwargs,
       )

       class Timed:
           def __init__(self, name): self.name = name
           def __enter__(self):
               if device == "cuda":
                   torch.cuda.reset_peak_memory_stats()
               self.t0 = time.time()
               return self
           def __exit__(self, *a):
               dt = time.time() - self.t0
               if device == "cuda":
                   peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
                   print(f"[{self.name}] {dt:.2f}s, peak={peak:.2f}GB")
               else:
                   print(f"[{self.name}] {dt:.2f}s")

       model.eval()
       with torch.no_grad():
           autocast = (
               torch.cuda.amp.autocast(dtype=torch.float16)
               if device == "cuda" else nullcontext()
           )
           with Timed("train(grpo)"), autocast:
               trainer.train()

   if __name__ == "__main__":
       main()

C++: Fused sampling (temperature + softmax + multinomial) skeleton
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   // file: fused_sample.cpp
   #include <torch/extension.h>

   torch::Tensor fused_sample(torch::Tensor logits, double temperature) {
       TORCH_CHECK(logits.is_cuda(), "logits must be CUDA");
       auto scaled = logits / temperature;
       auto probs  = torch::softmax(scaled, -1);
       auto ids = torch::multinomial(probs, 1);
       return ids;
   }

   PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
       m.def("fused_sample", &fused_sample,
             "Fused temperature-softmax-sample");
   }

Build script (setup.py)
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from setuptools import setup
   from torch.utils.cpp_extension import CppExtension, BuildExtension

   setup(
       name="fused_sample_ext",
       ext_modules=[
           CppExtension("fused_sample_ext", ["fused_sample.cpp"]),
       ],
       cmdclass={"build_ext": BuildExtension},
   )

Engineering Infrastructure
--------------------------

- **Automatic build system**: Python env (``requirements.txt``), Dockerfile
  (CUDA 12.x + PyTorch + TRL), C++ extension auto-build.
- **Version control**: ``main`` for stable, ``feat/*`` for experiments, PRs
  gated by lint/tests.
- **Testing**: CPU smoke tests, GPU V100 short GRPO runs, tokens/sec + VRAM
  benchmarks.
- **Documentation**: ``README.md``, ``docs/`` (Sphinx), inline docstrings.
- **CI**: GitHub Actions (CPU), optional self-hosted GPU runner.
- **Optimization playbook**:

  - Python: ``model.eval()``, ``no_grad``, FP16 autocast, length bucketing,
    vectorized reward.
  - C++: fused sampling kernel, CUDA Graphs, KV pooling.
  - Numerical: log-sum-exp for KL, clamped probs.
  - System: stage profiling, Accelerate/DeepSpeed A/B tests.

Schedule
--------

- **Planning phase (8 weeks from 09/22 to 11/25)**
- Week 1 (09/22): Repo generate, baseline GRPO run on V100.
- Week 2 (09/30): Profiling hooks, enforce FP16 inference.
- Week 3 (10/19): Baseline Profiling and Build Setup
  - Verify that the Qwen2.5-VL model and dependencies run correctly on the V100 GPU.
  - Profile the baseline attention module using Nsight Systems to confirm
    that kernel launch overhead dominates runtime.
  - Create a new directory ``cpp_ext/`` and set up a working ``setup.py`` build
    script for compiling custom CUDA extensions.
  - Ensure the environment builds successfully with dummy kernels and imports
    correctly in Python.

- Week 4 (10/28): Implement ``h2d_once.cu`` and PyBind Interface
  - Design and implement ``h2d_once.cu`` to batch all model tensors (Q, K, V, mask)
    into one contiguous host buffer.
  - Use ``cudaMemcpyAsync`` to copy all batched data in a single transfer per batch.
  - Build and import this kernel in Python via the PyBind interface
    (``gpuops.h2d_once``).
  - Compare transfer timing against PyTorch’s default copy behavior to confirm
    reduced H2D overhead.

- Week 5 (11/05): Implement ``masked_softmax.cu``
  - Implement a fused CUDA kernel combining scaling, causal masking, and softmax
    into a single operation.
  - Use FP16 for input/output and FP32 for computation to maintain numerical accuracy.
  - Build and test the function via PyBind (``gpuops.masked_softmax``), verifying
    numerical correctness against ``torch.nn.functional.softmax``.

- Week 6 (11/12): Integration and Functional Verification
  - Modify ``transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`` to replace
    internal attention function calls with the new fused kernels.
  - Confirm that the model produces correct output compared to the original
    implementation.
  - Verify numerical stability and correctness using a small inference test.

- Week 7 (11/19): Performance Comparison and Profiling Visualization
  - Run performance comparison and collect metrics: kernel-launch count,
    per-token latency, total inference time, and GPU utilization.
  - Visualize profiling data (e.g., Nsight Systems timeline) to demonstrate reduced
    kernel fragmentation and fewer synchronization events.

- Week 8 (11/25): Final Documentation
  - Write a concise technical report (``docs/opt_report.md``) summarizing:
    - profiling observations,
    - kernel design and integration steps,
    - benchmark and profiling results.

References
----------

1. TRL GitHub repository: https://github.com/huggingface/trl
2. TRL Docs Index: https://huggingface.co/docs/trl/en/index
3. PPO Trainer docs: https://huggingface.co/docs/trl/main/en/ppo_trainer
4. Examples overview: https://huggingface.co/docs/trl/main/en/example_overview
5. TRL Issues (Accelerate/vLLM):
   https://github.com/huggingface/trl/issues/3881
