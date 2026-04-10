# Subgroup I3: smolVLA for Lightweight Embodied Task Execution

## Assignment 3: smolVLA Benchmark for Robotic Manipulation Planning (SIMULATION)

What to do: Evaluate Hugging Face's smolVLA as a lightweight Vision‑Language‑Action model for robotic manipulation tasks, comparing it against larger VLA models (OpenVLA, RT‑2) in terms of task success, inference speed, and resource consumption, with the goal of assessing edge‑deployment feasibility
1) Set up smolVLA from Hugging Face in a local inference pipeline with a simulated robot environment
2) Design a set of 10 tabletop manipulation tasks of increasing complexity (pick, place, stack, sort by color, pour, push, open drawer, tool use, multi‑step, react to change) in a simulator (Gazebo or SIMPLER)
3) Define a standardized evaluation protocol: task success rate, planning accuracy, number of re‑plans needed, total execution time
4) Run the same tasks with OpenVLA and/or RT‑2 (or available open VLA alternatives) as comparison baselines
5) Measure resource consumption: GPU VRAM, inference latency, CPU usage for each model
6) Test smolVLA with different input modalities: RGB only, RGB + depth, RGB + semantic mask from SAM
7) Analyze where smolVLA fails compared to larger models and identify task complexity thresholds

Software needed: Hugging Face Transformers, smolVLA, OpenVLA, PyTorch, Gazebo or SIMPLER env, ROS2 Humble, Python profiling tools (torch.profiler, nvidia‑smi)
Research needed: VLA model architectures (openpi, OpenVLA, smolVLA), embodied AI benchmarks, model compression and edge deployment, vision‑language grounding, action tokenization methods
Deliverables: smolVLA evaluation pipeline, benchmark results across all tasks and models, resource consumption analysis, recommendation report on edge deployment viability, identification of failure modes and task complexity limits
