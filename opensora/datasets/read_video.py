import gc
import math
import os
import re
import warnings
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple, Union

import av
import cv2
import numpy as np
import torch
from torchvision import get_video_backend
from torchvision.io.video import _check_av_available

MAX_NUM_FRAMES = 2500


def read_video_av(
    filename: str,
    start_pts: Union[float, Fraction] = 0,
    end_pts: Optional[Union[float, Fraction]] = None,
    pts_unit: str = "pts",
    output_format: str = "THWC",
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Reads a video from a file, returning both the video frames and the audio frames

    This method is modified from torchvision.io.video.read_video, with the following changes:

    1. will not extract audio frames and return empty for aframes
    2. remove checks and only support pyav
    3. add container.close() and gc.collect() to avoid thread leakage
    4. try our best to avoid memory leak

    Args:
        filename (str): path to the video file
        start_pts (int if pts_unit = 'pts', float / Fraction if pts_unit = 'sec', optional):
            The start presentation time of the video
        end_pts (int if pts_unit = 'pts', float / Fraction if pts_unit = 'sec', optional):
            The end presentation time
        pts_unit (str, optional): unit in which start_pts and end_pts values will be interpreted,
            either 'pts' or 'sec'. Defaults to 'pts'.
        output_format (str, optional): The format of the output video tensors. Can be either "THWC" (default) or "TCHW".

    Returns:
        vframes (Tensor[T, H, W, C] or Tensor[T, C, H, W]): the `T` video frames
        aframes (Tensor[K, L]): the audio frames, where `K` is the number of channels and `L` is the number of points
        info (Dict): metadata for the video and audio. Can contain the fields video_fps (float) and audio_fps (int)
    """
    # format
    output_format = output_format.upper()
    if output_format not in ("THWC", "TCHW"):
        raise ValueError(f"output_format should be either 'THWC' or 'TCHW', got {output_format}.")
    # file existence
    if not os.path.exists(filename):
        raise RuntimeError(f"File not found: {filename}")
    # backend check
    assert get_video_backend() == "pyav", "pyav backend is required for read_video_av"
    _check_av_available()

    assert pts_unit == "pts" and start_pts is not None and end_pts is not None

    # end_pts check
    if end_pts is None:
        end_pts = float("inf")
    if end_pts < start_pts:
        raise ValueError(f"end_pts should be larger than start_pts, got start_pts={start_pts} and end_pts={end_pts}")

    info = {}

    # == read ==
    try:
        # TODO: The reading has memory leak (4G for 8 workers 1 GPU)
        container = av.open(filename, metadata_errors="ignore")
        assert container.streams.video is not None

        video_fps = container.streams.video[0].average_rate
        if video_fps is not None:
            info["video_fps"] = float(video_fps)
        height = container.streams.video[0].height
        width = container.streams.video[0].width
        total_frames = end_pts - start_pts + 1

        vframes = torch.zeros((total_frames, height, width, 3), dtype=torch.uint8)

        for idx, frame in enumerate(container.decode(video=0)):
            if idx < start_pts or idx > end_pts:
                continue
            rgb_frame = frame.to_rgb() # NOTE: may or may not allocate depending on current format
            np_frame = rgb_frame.to_ndarray() # NOTE: may or may not allocate depending on current format
            # NOTE: torch.from_numpy will attempt to share memory
            vframes[idx - start_pts] = torch.from_numpy(np_frame).clone()
            del np_frame
            del rgb_frame
            del frame
    finally:
        # garbage collection for thread leakage
        container.close()
        del container

    if output_format == "TCHW":
        # [T,H,W,C] --> [T,C,H,W]
        vframes = vframes.permute(0, 3, 1, 2)

    aframes = torch.empty((1, 0), dtype=torch.float32)
    return vframes, aframes, info


def read_video_cv2(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        # print("Error: Unable to open video")
        raise ValueError
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        vinfo = {
            "video_fps": fps,
        }

        frames = []
        while True:
            # Read a frame from the video
            ret, frame = cap.read()

            # If frame is not read correctly, break the loop
            if not ret:
                break

            frames.append(frame[:, :, ::-1])  # BGR to RGB

            # Exit if 'q' is pressed
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break

        # Release the video capture object and close all windows
        cap.release()
        cv2.destroyAllWindows()

        frames = np.stack(frames)
        frames = torch.from_numpy(frames)  # [T, H, W, C=3]
        frames = frames.permute(0, 3, 1, 2)
        return frames, vinfo


def read_video(video_path, start_pts=0, end_pts=None, pts_unit="pts", backend="av"):
    #if False and backend == "cv2":
    #    vframes, vinfo = read_video_cv2(video_path)
    #elif True or backend == "av":
    if True:
        vframes, _, vinfo = read_video_av(filename=video_path, start_pts=start_pts, end_pts=end_pts, pts_unit=pts_unit, output_format="TCHW")
    else:
        raise ValueError

    return vframes, vinfo
