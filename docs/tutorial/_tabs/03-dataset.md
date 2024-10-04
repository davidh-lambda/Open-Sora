---
layout: post
icon: fas fa-database
title: Dataset
date: 2024-10-02
toc: true
---


### [**Dataset** — Downloading & Preprocessing](../03-dataset)
Let's face it: Video data is not easily accessible, and there aren't many publicly available sources.  
We'll
- download [**OpenVid**](https://github.com/NJU-PCALab/OpenVid-1M) and [**MiraData**](https://github.com/mira-space/MiraData) for our reproduction experiment
- walk through the necessary steps to **preprocess the datasets**.
- Since preprocessing takes time even at this scale, we'll also talk about how to use your cluster to **parallelize** such preprocessing tasks conveniently without writing any code - call it a little love letter to what Unix already brings to the table if you want <3.

