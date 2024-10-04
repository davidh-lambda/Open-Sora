---
layout: post
icon: fas fa-database
title: Dataset
date: 2024-10-02
toc: true
---


### [**Dataset** — Downloading & Preprocessing](../03-dataset)
Let's face it: Video data is not easily accessible, and there aren't many publicly available sources.  
- download [**OpenVid**](https://github.com/NJU-PCALab/OpenVid-1M) and [**MiraData**](https://github.com/mira-space/MiraData) for our reproduction experiment. For MiraData, we follow the guidance from the author's [repo
](https://github.com/mira-space/MiraData/tree/v1?tab=readme-ov-file#download) to download the 330K version of the dataset. OpenVid has 1M video clips and captions that can be downloaded from their Huggingface [link](https://huggingface.co/datasets/nkp37/OpenVid-1M).  
- Since both OpenVid and MiraData come with clips and captions, we can skip most of the [preprocess steps](https://github.com/hpcaitech/Open-Sora/blob/main/docs/data_processing.md), except for adding missing meta data to the csv file for training. The full list of required colmuns by the csv for training opensora model: `path,text,num_frames,height,width,aspect_ratio,fps,resolution,file_size`, and suming you've got the csv with the columns path,text for your data set ready, all you need to do to fill the rest is: 
```
python -m tools.datasets.datautil dataset.csv --info --remove-empty-caption
```
- We remove video clip that are larger than 50MB since they are expensive to load during training.
- We found certain video clips will cause FFMPEG error and crash the training, so have to filter out files with FFMPEG errors from the dataset before training. To do so, we check for each video clip whether FFMPEG can decode the video. This process is CPU-intensive and to speed up we parallelize it across multiple servers. And the steps to do so are
```
# Filter out FFMPEG errors.
1. create an `all.txt` with all file names (essentially only the column `path` from our dataset.csv.
2. assuming we have 24 nodes available for checking, `split -n l/24 all.txt` to create the sub-lists xa ... xz
3. adapt the parameters in `tools/ffmpeg_check_parallel.sh` and run it on all nodes
4. this will create .err files along all files in all.txt. if the file is empty, it's good, if it's non-empty there was an error
5. we will filter out all non-zero err files from all.txt using `tools/ffmpeg_filter_without_errors.py`
```

We store the dataset on a shared storage so they are available for all compute nodes. 
