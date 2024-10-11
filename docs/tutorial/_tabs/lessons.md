---
layout: post
icon: fas fa-graduation-cap
title: Lessons Learned
date: 2024-10-02
toc: true
order: 2
---

# From Training Troubleshooting to Cluster Issues

Before diving into training the model from scratch, we wanted to share some valuable lessons we learned along the way. In this section, we'll address problems we came across that you might encounter too, and explain how to solve them. These range from model divergence issues to debugging on a cluster and tackling cluster-related configuration problems that can be challenging to debug.

## Debugging Model Convergence Issues
> **TL;DR:** We experienced model divergence when changing datasets mid-training due to mismatched data loader states, leading to repeated batches and divergence. By fixing the data loader to match the new dataset and smoothing transitions between data chunks, we resolved the convergence issues.

When your model isn't converging as expected, it can be frustrating. In an earlier experiment, we tried splitting our data into five smaller chunks of increasing difficulty, each to be trained for two epochs. While this approach isn't part of the main tutorial, we thought it would be helpful to explain what went wrong and how we resolved the issues.

The loss and weight norm curves for the full 10 epochs of training looked like this:

![Loss Curve](./assets/fails_loss.png) ![Weight Norm Curve](./assets/fails_weight_norm.png)

As you might notice, there are three divergence points where we needed to roll back to an earlier checkpoint and restart training, even though we hadn't changed any hyperparameters.

**Problems We Found During Training**
The key issue stemmed from how we swapped the sub-datasets. When we switched to a new data chunk, the number of batches changed. We continued training using the `--load` flag (see [Training Section](../training.md)), which meant that the saved data loader state did not match the new dataset. This mismatch led to some batches being shown multiple times in succession.

Since the data loader state was assuming a dataset length different from the actual one, we encountered divergences:

- **Blue Curve**: When the current index of the pre-calculated batches (from a different number of samples) was greater than expected, the network diverged. We fixed this in an earlier implementation and resumed training successfully (shown in the brown curve).
- **Violet Curve**: We still had an off-by-one bug in our fix for the data loader. Interestingly, this didn't lead to divergence, but we saw a sudden but slight decrease in output quality with each new epoch. The outputs became brighter, and we observed small jumps in the average weight norm of the network, happening at the last batch of each epoch. We fixed that off-by-one error and continued training again (shown in the mint curve).
- **Purple Curve**: Since we tried to increase training speed by splitting data into levels of increasing quality (using the number of nouns in the video descriptions as an indicator), we found that the last jump — from difficulty level "4 of 5" to "5 of 5" — was too drastic. This significant drift in data has led to the divergence seen in the purple curve. Removing the hardest `0.01%` of that data solved the problem, and training finished successfully.

## Debugging on a Cluster
> **TL;DR:** Our training processes were freezing randomly without errors. Using py-spy, we discovered that garbage collection issues in PyAV within our data loader were causing the freezes. Refactoring the code to use a single PyAV instance eliminated the problem.

When training on a cluster, you might run into issues that are hard to diagnose. Particularly data loader code is inherently highly parallel: on every node in the cluster, for every training job (one per GPU), we have multiple workers (in our case, 16 per training process/GPU) that read data to feed into the training script as efficiently as possible.  
Unfortunately, we found that our training froze unexpectedly without any errors — and worse, it happened randomly every 2 to 6 hours. Checking memory usage, CPU usage, and other metrics didn't reveal any issues.

Training on a large cluster is expensive, so while [simple solutions](https://xkcd.com/1495/) like restarting everything recurrently might seem effective in the short term, we needed a more sustainable fix since stopping and starting the cluster also incurs costs: The training startup is slow due to launching jobs on all nodes, loading checkpoint states, and any just-in-time compilation that might need to happen beforehand.

**So, how can we debug a problem like random data loader freezes?**  
We found that [py-spy](https://github.com/benfred/py-spy) was invaluable for this task. `py-spy` has several modes, and we were interested in using the `dump` feature, which prints out the current process stack trace for a given PID.

We used the following command to run `py-spy dump` for all workers on all our machines from our head node:
```bash
parallel -a nodes.txt -S {} "ssh {} 'pgrep -f python.*train.py | xargs -I {{}} py-spy dump --pid {{}} > ./dumps/{{}}.out'"
```

We then looked for unusual methods where our processes were spending a lot of time by examining active threads (most other threads in the dumps were idle).
By re-evaluating after several minutes, we saw that the same processes were still stuck on the same lines—nothing had changed.
The line standing in the dump is out is the `_read_from_stream` method.

![](./assets/pyspy_dump.png)

Looking at the [specific line in the code](https://github.com/davidh-lambda/Open-Sora/blob/73c77fbb6eb5dac72fc8327a3797fd91894d25cd/opensora/datasets/read_video.py#L211), we found the following linke causing the problem.

```python
210 result = video_frames[start_ptr:end_ptr].copy()
211 return result
```

It seemed to be related to garbage collection.

Since PyAV is used under the hood, we checked their documentation and discovered their notes on [Caveats related to Garbage Collection](https://pyav.org/docs/stable/overview/caveats.html#garbage-collection), possibly causing freezes due to opening and closing instances frequently. We refactored [the code to use only a single PyAV instance](https://github.com/davidh-lambda/Open-Sora/commit/3e935bcb46c03d30330815244a1aff4e67d42fd0), which solved the problem!


## Random NCCL Errors

> **TL;DR:** We faced frequent NCCL errors causing training crashes, traced back to clock synchronization issues across cluster nodes. Implementing proper time synchronization using Chrony resolved these errors and stabilized our training runs.

Another issue we encountered was random crashes due to NCCL errors. These crashes happened more regularly, sometimes every 30 minutes, and also required us to manually restart training.

The key error message was:
```
Some NCCL operations have failed or timed out. Due to the asynchronous nature of CUDA kernels, subsequent GPU operations might run on corrupted/incomplete data.
```

After some investigation, we realized that the issue was caused by clock synchronization problems across the cluster nodes. One of our team members observed that some nodes' clocks were moving backwards:

> "I noticed that one node's clock jumped backwards—the last line of `dmesg` was a few seconds ahead of the current time on that node. It appears that the polling interval is 34 minutes, and the upstream NTP server is behind this node, causing the clock to slew backwards. The freeze periodicity seems to match this interval."

The solution was to address the clock synchronization issues by switching the NTP client to Chrony on the cluster head node. This change resolved the hangs caused by NCCL.

<br/>

---

**What Next?**
With these lessons learned, you can now proceed to the [Setup](../setup.md) section to set up the codebase and begin training.

In the next part, we'll guide you through cloning the repository, installing dependencies, and configuring your cluster. You'll learn how to create a shared folder, install Miniconda on all nodes, clone the required codebase, and ensure all nodes have consistent environments and access to necessary files. Maintaining uniformity across all nodes is essential, as inconsistencies can lead to challenging bugs during the training process.

---