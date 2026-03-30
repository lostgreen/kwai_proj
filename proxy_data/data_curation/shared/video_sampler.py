"""
轻量级视频抽帧工具 — 用于 VLM 视觉校验。

功能：
  - 从视频中均匀抽取 N 帧（默认 6 帧）
  - 舍弃开头前 5% 和结尾后 5% 以避开片头片尾
  - 压缩分辨率至最长边 512px
  - 返回 base64 编码的 JPEG 图片列表

依赖: pip install decord Pillow

用法:
    from shared.video_sampler import sample_frames_base64

    frames_b64 = sample_frames_base64("/path/to/video.mp4", n_frames=6)
    # frames_b64: list[str]  — 每个元素是 JPEG 的 base64 字符串
"""

import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image


def _resize_frame(img: Image.Image, max_side: int = 512) -> Image.Image:
    """Resize so the longest side is at most max_side, preserving aspect ratio."""
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    return img.resize((new_w, new_h), Image.LANCZOS)


def _frame_to_base64(img: Image.Image, quality: int = 85) -> str:
    """Encode a PIL Image as JPEG base64 string."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def sample_frames_base64(
    video_path: str,
    n_frames: int = 6,
    skip_ratio: float = 0.05,
    max_side: int = 512,
    jpeg_quality: int = 85,
) -> list[str]:
    """从视频中均匀抽取 N 帧，返回 base64 编码的 JPEG 列表。

    Args:
        video_path: 视频文件路径
        n_frames: 抽取帧数（默认 6）
        skip_ratio: 跳过开头/结尾的比例（默认 5%）
        max_side: 图片最长边像素（默认 512）
        jpeg_quality: JPEG 压缩质量（默认 85）

    Returns:
        list[str]: base64 编码的 JPEG 字符串列表
    """
    try:
        from decord import VideoReader, cpu
    except ImportError:
        raise ImportError(
            "decord is required for video frame extraction: "
            "pip install decord"
        )

    vr = VideoReader(str(video_path), ctx=cpu(0))
    total_frames = len(vr)

    if total_frames == 0:
        return []

    # Skip first/last skip_ratio of the video
    start_frame = int(total_frames * skip_ratio)
    end_frame = int(total_frames * (1 - skip_ratio))

    # Ensure valid range
    if end_frame <= start_frame:
        start_frame = 0
        end_frame = total_frames - 1

    # Uniformly sample N frame indices within the effective range
    if n_frames >= (end_frame - start_frame):
        indices = list(range(start_frame, end_frame))
    else:
        indices = np.linspace(start_frame, end_frame - 1, n_frames, dtype=int).tolist()

    # Decode frames
    frames = vr.get_batch(indices).asnumpy()  # (N, H, W, 3)

    results = []
    for frame in frames:
        img = Image.fromarray(frame)
        img = _resize_frame(img, max_side)
        results.append(_frame_to_base64(img, jpeg_quality))

    return results


def sample_frames_pil(
    video_path: str,
    n_frames: int = 6,
    skip_ratio: float = 0.05,
    max_side: int = 512,
) -> list[Image.Image]:
    """与 sample_frames_base64 相同，但返回 PIL Image 列表（方便调试可视化）。"""
    try:
        from decord import VideoReader, cpu
    except ImportError:
        raise ImportError(
            "decord is required for video frame extraction: "
            "pip install decord"
        )

    vr = VideoReader(str(video_path), ctx=cpu(0))
    total_frames = len(vr)

    if total_frames == 0:
        return []

    start_frame = int(total_frames * skip_ratio)
    end_frame = int(total_frames * (1 - skip_ratio))

    if end_frame <= start_frame:
        start_frame = 0
        end_frame = total_frames - 1

    if n_frames >= (end_frame - start_frame):
        indices = list(range(start_frame, end_frame))
    else:
        indices = np.linspace(start_frame, end_frame - 1, n_frames, dtype=int).tolist()

    frames = vr.get_batch(indices).asnumpy()

    results = []
    for frame in frames:
        img = Image.fromarray(frame)
        img = _resize_frame(img, max_side)
        results.append(img)

    return results
