#

# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import random
from collections import defaultdict
from io import BytesIO
from typing import Any, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from qwen_vl_utils.vision_process import fetch_video
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from . import torch_functional as VF


_VIDEO_DEBUG_ENABLED = os.environ.get("EASYR1_DEBUG_VIDEO_FRAMES", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}
_VIDEO_DEBUG_MAX_LOGS = int(os.environ.get("EASYR1_DEBUG_VIDEO_FRAMES_MAX_LOGS", "200"))
_video_debug_log_count = 0

_VISUAL_TOKEN_DEBUG_ENABLED = os.environ.get(
    "EASYR1_DEBUG_VISUAL_TOKENS",
    os.environ.get("EASYR1_DEBUG_VIDEO_FRAMES", "0"),
).strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}
_VISUAL_TOKEN_DEBUG_MAX_LOGS = int(
    os.environ.get(
        "EASYR1_DEBUG_VISUAL_TOKENS_MAX_LOGS",
        os.environ.get("EASYR1_DEBUG_VIDEO_FRAMES_MAX_LOGS", "200"),
    )
)
_visual_token_debug_log_count = 0


QUESTION_TEMPLATE = (
    "{Question}\n"
    "Please answer this question based on the visual content."
    "Provide your thinking process between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags."
    "At the end, you must output the final answer in the format:\n"
    "<answer><your_answer_here></answer>\n"
)

TYPE_TEMPLATE = {
    "multiple choice": (
        "Please provide only the single option letter (e.g., A, B, C, D, etc.) "
        "within the <answer>...</answer> tags.\n"
        "Example:\n<answer>A</answer>"
    ),
    "numerical": (
        "Please provide only the numerical value within the <answer>...</answer> tags.\n"
        "Example:\n<answer>3.14</answer>"
    ),
    "OCR": (
        "Please provide only the transcribed text within the <answer>...</answer> tags.\n"
        "Example:\n<answer>Hello World</answer>"
    ),
    "open-ended": (
        "Please provide only your text answer within the <answer>...</answer> tags.\n"
        "Example:\n<answer>The capital of France is Paris.</answer>"
    ),
    "regression": (
        "Please provide only the numerical value within the <answer>...</answer> tags.\n"
        "Example:\n<answer>42.7</answer>"
    ),
    "math": (
        "Please provide only the final result (a number or LaTeX formula) within the <answer>...</answer> tags.\n"
        "Example:\n<answer>$$-\\dfrac{3}{2}$$</answer>"
    ),
    "temporal grounding": (
        "Please provide only the time span in seconds as JSON within the <answer>...</answer> tags.\n"
        "Example:\n<answer>{\"time\": [12.3, 25.7]}</answer>"
    ),
    "spatial grounding": (
        "Please provide only the bounding box as JSON with key 'boxes' within the <answer>...</answer> tags.\n"
        "Example:\n<answer>{\"boxes\": [35, 227, 437, 932]}</answer>"
    ),
    "spatial-temporal grounding": (
        "Please provide only the time span in seconds and bounding boxes as JSON within the <answer>...</answer> tags.\n"
        "You MUST output one bounding box for every integer second within the given time span (inclusive).\n"
        "Example:\n"
        "<answer>{\"time\": [8.125, 13.483], \"boxes\": {\"9\": [317, 422, 582, 997], "
        "\"10\": [332, 175, 442, 369], \"11\": [340, 180, 450, 370]}}</answer>\n"
        "Note: Each key in 'boxes' must be an integer second within the span, and its value must be a 4-number bounding box [x1, y1, x2, y2]."
    ),
    "tracking": (
        "Please track the target object throughout the video and provide one bounding box per second, "
        "ONLY up to 32 seconds, within the <answer>...</answer> tags.\n"
        "Example:\n"
        "<answer>{\"boxes\": {\"1\": [405, 230, 654, 463], \"2\": [435, 223, 678, 446], ..., "
        "\"32\": [415, 203, 691, 487]}}</answer>\n"
        "Note: Each key in 'boxes' must correspond to a second (1, 2, 3, ..., 32) and contain a 4-number bounding box [x1, y1, x2, y2]."
    ),
    "segmentation_image": (
        "This task prepares inputs for image object segmentation with a specialized model (e.g., SAM2).\n"
        "Please provide ONE bounding box, 3 positive points (clearly INSIDE the object), and 3 negative points "
        "(clearly OUTSIDE the object) within the <answer>...</answer> tags.\n"
        "Choose informative points that help distinguish object vs. background. Prefer negatives on clear non-object "
        "pixels INSIDE the box when safe; otherwise place them just outside on obvious background. "
        "Negatives must NEVER be on the object or on its boundary.\n"
        "Example:\n"
        "<answer>{\"boxes\": [x1, y1, x2, y2], \"positive_points\": [[x,y],[x,y],[x,y]], "
        "\"negative_points\": [[x,y],[x,y],[x,y]]}</answer>"
    ),
    "segmentation_video": (
        "This task prepares inputs for video object segmentation with a specialized model (e.g., SAM2).\n"
        "Please select ONE representative time (in seconds), and provide ONE bounding box, "
        "3 positive points (clearly INSIDE the object), and 3 negative points (clearly OUTSIDE the object) "
        "within the <answer>...</answer> tags.\n"
        "Choose informative points that help distinguish object vs. background. Prefer negatives on clear non-object "
        "pixels INSIDE the box when safe; otherwise place them just outside on obvious background. "
        "Negatives must NEVER be on the object or on its boundary.\n"
        "Example:\n"
        "<answer>{\"time\": <time_in_seconds>, \"boxes\": [x1, y1, x2, y2], "
        "\"positive_points\": [[x,y],[x,y],[x,y]], \"negative_points\": [[x,y],[x,y],[x,y]]}</answer>"
    )
}


def collate_fn(features: list[dict[str, Any]]) -> dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


def process_image(
    image: Union[dict[str, Any], ImageObject, str], min_pixels: Optional[int], max_pixels: Optional[int]
) -> ImageObject:
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    # print(max_pixels)

    image.load()  # avoid "Too many open files" errors
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if min_pixels is not None and (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def process_video(
    video: str, min_pixels: int = 4*32*32, max_pixels: int = 48*32*32, max_frames: int = 256, video_fps: float = 2, min_frames: int = 0, return_fps: bool = False
):
    vision_info = {"video": video, "min_pixels": min_pixels, "max_pixels": max_pixels, "max_frames": max_frames, "fps": video_fps}
    result = fetch_video(vision_info, image_patch_size=16, return_video_sample_fps=return_fps, return_video_metadata=return_fps)

    # If min_frames is set and the result has fewer frames, retry with higher fps
    if min_frames > 0:
        n_frames = _count_video_frames(result)
        if n_frames is not None and n_frames < min_frames and n_frames > 0:
            # Estimate the video duration from current sampling: duration ≈ n_frames / video_fps
            # Compute the fps needed to get min_frames: new_fps = min_frames / duration
            est_duration = n_frames / max(video_fps, 0.1)
            new_fps = min(min_frames / max(est_duration, 0.1), max_frames / max(est_duration, 0.1))
            new_fps = max(new_fps, video_fps)  # never go below the original fps
            if new_fps > video_fps:
                vision_info_retry = {"video": video, "min_pixels": min_pixels, "max_pixels": max_pixels, "max_frames": max_frames, "fps": new_fps}
                result = fetch_video(vision_info_retry, image_patch_size=16, return_video_sample_fps=return_fps, return_video_metadata=return_fps)

    _maybe_log_video_debug(
        video=video,
        result=result,
        target_fps=video_fps,
        max_frames=max_frames,
        return_fps=return_fps,
    )
    return result


def _count_video_frames(video_data: Any) -> Optional[int]:
    # Typical cases:
    # - np.ndarray/torch.Tensor: (T, H, W, C) or (T, C, H, W)
    # - tuple: (video_frames, metadata)
    # - list: [frame0, frame1, ...]
    if video_data is None:
        return None

    if isinstance(video_data, tuple):
        if len(video_data) == 0:
            return 0
        # Prefer the first item for (frames, metadata)-style payloads.
        return _count_video_frames(video_data[0])

    if isinstance(video_data, list):
        return len(video_data)

    if isinstance(video_data, np.ndarray):
        return int(video_data.shape[0]) if video_data.ndim >= 4 else None

    if isinstance(video_data, torch.Tensor):
        return int(video_data.shape[0]) if video_data.ndim >= 4 else None

    shape = getattr(video_data, "shape", None)
    if shape is not None and len(shape) >= 4:
        return int(shape[0])

    return None


def _maybe_log_video_debug(video: Any, result: Any, target_fps: float, max_frames: int, return_fps: bool) -> None:
    global _video_debug_log_count
    if not _VIDEO_DEBUG_ENABLED:
        return
    if _video_debug_log_count >= _VIDEO_DEBUG_MAX_LOGS:
        return

    sampled_fps = None
    video_payload = result
    if return_fps and isinstance(result, tuple) and len(result) >= 2:
        video_payload = result[0]
        sampled_fps = result[1]

    sampled_frames = _count_video_frames(video_payload)
    rank = os.environ.get("RANK", "0")
    src = video if isinstance(video, str) else f"<{type(video).__name__}>"
    print(
        "[VIDEO_DEBUG][process_video] "
        f"rank={rank} src={src} target_fps={target_fps} sampled_fps={sampled_fps} "
        f"max_frames={max_frames} sampled_frames={sampled_frames}"
    )
    _video_debug_log_count += 1


def _count_token(input_ids: torch.Tensor, token_id: Optional[int]) -> int:
    if token_id is None or not isinstance(input_ids, torch.Tensor):
        return 0
    if token_id < 0:
        return 0
    return int((input_ids == token_id).sum().item())


def _grid_to_list(grid: Optional[torch.Tensor]) -> Optional[list[list[int]]]:
    if grid is None:
        return None
    if not isinstance(grid, torch.Tensor):
        return None
    if grid.numel() == 0:
        return []
    return [[int(v) for v in row] for row in grid.detach().cpu().tolist()]


def _maybe_log_visual_token_debug(
    *,
    index: int,
    prompt_len_before_pad: int,
    prompt_len_after_pad: int,
    video_tokens_before_pad: int,
    video_tokens_after_pad: int,
    image_tokens_before_pad: int,
    image_tokens_after_pad: int,
    raw_prompt_len: int,
    max_prompt_length: int,
    video_grid_thw: Optional[torch.Tensor],
    image_grid_thw: Optional[torch.Tensor],
) -> None:
    global _visual_token_debug_log_count
    if not _VISUAL_TOKEN_DEBUG_ENABLED:
        return
    if _visual_token_debug_log_count >= _VISUAL_TOKEN_DEBUG_MAX_LOGS:
        return

    rank = os.environ.get("RANK", "0")
    print(
        "[VIDEO_DEBUG][prompt_tokens] "
        f"rank={rank} idx={index} "
        f"prompt_len_before={prompt_len_before_pad} prompt_len_after={prompt_len_after_pad} "
        f"video_tokens_before={video_tokens_before_pad} video_tokens_after={video_tokens_after_pad} "
        f"image_tokens_before={image_tokens_before_pad} image_tokens_after={image_tokens_after_pad} "
        f"raw_prompt_len={raw_prompt_len} max_prompt_length={max_prompt_length} "
        f"video_grid_thw={_grid_to_list(video_grid_thw)} image_grid_thw={_grid_to_list(image_grid_thw)}"
    )
    _visual_token_debug_log_count += 1


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        video_key: str = "videos",
        image_dir: Optional[str] = None,
        video_fps: float = 2.0,
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        max_frames: int = 256,
        min_frames: int = 0,
        filter_overlong_prompts: bool = True,
        filter_overlong_prompts_workers: int = 16,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.video_key = video_key
        self.image_dir = image_dir
        self.video_fps = video_fps
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.max_frames = max_frames
        self.min_frames = min_frames

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            # when we use dataset builder, we should always refer to the train split
            file_type = os.path.splitext(os.listdir(data_path)[0])[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_dir=data_path, split=data_split)
        elif os.path.isfile(data_path):
            file_type = os.path.splitext(data_path)[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_files=data_path, split=data_split)
        else:
            # load remote dataset from huggingface hub
            self.dataset = load_dataset(data_path, split=data_split)

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        if filter_overlong_prompts:
            self.dataset = self.dataset.filter(
                self._filter_overlong_prompts,
                desc="Filtering overlong prompts",
                num_proc=filter_overlong_prompts_workers,
            )


    def _build_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        # ── 直接使用数据中预构建的 messages（跳过 QUESTION_TEMPLATE 包装）──
        if "messages" in example and isinstance(example["messages"], list) and len(example["messages"]) > 0:
            # 从 messages 列表中取最后一个 user 角色的 content
            prompt_str = ""
            for msg in example["messages"]:
                if msg.get("role") == "user":
                    prompt_str = msg.get("content", "")

            # 处理 <video> / <image> 占位符，转成多模态 token
            if self.video_key in example and isinstance(example.get(self.video_key), list) and len(example.get(self.video_key)) > 0:
                content_list = []
                for i, content in enumerate(prompt_str.split("<video>")):
                    if i != 0:
                        content_list.append({"type": "video"})
                    if content:
                        content_list.append({"type": "text", "text": content})
                return [{"role": "user", "content": content_list}]
            elif self.image_key in example and isinstance(example.get(self.image_key), list) and len(example.get(self.image_key)) > 0:
                content_list = []
                for i, content in enumerate(prompt_str.split("<image>")):
                    if i != 0:
                        content_list.append({"type": "image"})
                    if content:
                        content_list.append({"type": "text", "text": content})
                return [{"role": "user", "content": content_list}]
            else:
                return [{"role": "user", "content": prompt_str}]

        prompt_str: str = example[self.prompt_key]
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            prompt_str = format_prompt.render(content=prompt_str)

        data_type = (example.get("data_type") or "").strip().lower()
        pt = example.get("problem_type") or ""
        question = prompt_str

        if (pt == "multiple choice") and isinstance(example.get("options"), list) and example["options"]:
            opts = "\n".join(example["options"])
            question = f"{question}\nOptions:\n{opts}"

        if pt == "segmentation":
            type_key = "segmentation_video" if data_type == "video" else "segmentation_image"
        else:
            type_key = pt

        tail = TYPE_TEMPLATE.get(type_key, "")
        prompt_str = QUESTION_TEMPLATE.format(Question=question) + tail

        if self.image_key in example and isinstance(example.get(self.image_key), list) and len(example.get(self.image_key)) > 0:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})

            # print(content_list)

            return [{"role": "user", "content": content_list}]
        elif self.video_key in example and isinstance(example.get(self.video_key), list) and len(example.get(self.video_key)) > 0:
            content_list = []
            for i, content in enumerate(prompt_str.split("<video>")):
                if i != 0:
                    content_list.append({"type": "video"})

                if content:
                    content_list.append({"type": "text", "text": content})

            # print(content_list)

            return [{"role": "user", "content": content_list}]
        else:
            return [{"role": "user", "content": prompt_str}]


    def _filter_overlong_prompts(self, example: dict[str, Any]) -> bool:
        messages = self._build_messages(example)
        if self.image_key in example and isinstance(example.get(self.image_key), list) and len(example.get(self.image_key)) > 0:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example[self.image_key]
            try:
                if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                    images = [os.path.join(self.image_dir, image) for image in images]

            except Exception:
                print(f"images type: {type(images)} | value: {images}")
                print("full example:", example)



            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")

            print(images, model_inputs["input_ids"].size(-1))
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        elif self.video_key in example and isinstance(example.get(self.video_key), list) and len(example.get(self.video_key)) > 0:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example[self.video_key]
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            _meta = example.get("metadata") or {}
            _eff_fps = _meta.get("l1_fps", self.video_fps) if _meta.get("level") == 1 else self.video_fps
            for video in videos:
                processed_videos.append(process_video(video, min_pixels=self.min_pixels, max_pixels=self.max_pixels, max_frames=self.max_frames, min_frames=self.min_frames, video_fps=_eff_fps))

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            # print(videos, model_inputs["input_ids"].size(-1))
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        else:
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            return len(input_ids) <= self.max_prompt_length

    def __len__(self):
        return len(self.dataset)

    _BAD_VIDEO_MAX_RETRIES = 5

    def __getitem__(self, index):
        for _attempt in range(self._BAD_VIDEO_MAX_RETRIES + 1):
            try:
                return self._getitem_inner(index)
            except Exception as e:
                err_msg = str(e)
                # Log the bad sample
                _log_path = os.environ.get("BAD_SAMPLES_LOG", "bad_samples.txt")
                try:
                    with open(_log_path, "a") as f:
                        f.write(f"[BAD_SAMPLE] index={index} attempt={_attempt} error={err_msg}\n")
                except OSError:
                    pass
                print(f"[dataset] WARNING: __getitem__ failed for index={index} "
                      f"(attempt {_attempt+1}/{self._BAD_VIDEO_MAX_RETRIES+1}): {err_msg}")
                # Pick a different random index for retry
                index = random.randint(0, len(self) - 1)
        raise RuntimeError(
            f"[dataset] All {self._BAD_VIDEO_MAX_RETRIES+1} attempts failed in __getitem__. "
            f"Last index={index}"
        )

    def _getitem_inner(self, index):
        example: dict = self.dataset[index]
        messages = self._build_messages(example)
        example.pop(self.prompt_key, None)
        token_stats_grid_video = None
        token_stats_grid_image = None
        prompt_len_before_pad = 0
        video_tokens_before_pad = 0
        image_tokens_before_pad = 0

        if self.image_key in example and isinstance(example.get(self.image_key), list) and len(example.get(self.image_key)) > 0:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example.pop(self.image_key)
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"images": images}
            token_stats_grid_image = model_inputs.get("image_grid_thw", None)
        elif self.video_key in example and isinstance(example.get(self.video_key), list) and len(example.get(self.video_key)) > 0:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example.pop(self.video_key)
            if self.image_dir is not None and len(videos) != 0:
                if isinstance(videos[0], str):  # video paths
                    videos = [os.path.join(self.image_dir, video) for video in videos]
                elif isinstance(videos[0], list) and len(videos[0]) > 0 and isinstance(videos[0][0], str): # list of frame paths
                    videos = [[os.path.join(self.image_dir, frame) for frame in video] for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            video_fps_list = []
            all_frames = []       # 每个视频的 tensor frames
            all_metadatas = []    # 每个视频的 metadata dict
            # 多视频时，将 max_frames 均匀分配给每个视频以防止 OOM
            n_videos = len(videos)
            max_frames_per_video = max(1, self.max_frames // n_videos) if n_videos > 1 else self.max_frames
            # Per-record fps: L1 clips are resampled at l1_fps (e.g. 1fps)
            _meta = example.get("metadata") or {}
            _eff_fps = _meta.get("l1_fps", self.video_fps) if _meta.get("level") == 1 else self.video_fps

            for video in videos:
                processed_video, video_fps = process_video(
                    video, min_pixels=self.min_pixels, max_pixels=self.max_pixels, max_frames=max_frames_per_video, min_frames=self.min_frames, video_fps=_eff_fps, return_fps=True
                )
                video_kwargs = {"do_sample_frames": False}
                processed_videos.append(processed_video)
                video_fps_list.append(video_fps)

                # 每个视频的 process_video 返回 (frames, metadata) 或 None
                if processed_video is not None:
                    frames, meta = processed_video
                    all_frames.append(frames)
                    all_metadatas.append(meta)

            # video_metadata 长度必须与视频数量一致，否则处理器报 IndexError
            video_metadatas = all_metadatas if all_metadatas else None
            videos_input = all_frames if all_frames else None
            model_inputs= self.processor(text=[prompt], videos=videos_input, add_special_tokens=False, video_metadata=video_metadatas, return_tensors="pt", do_resize=False, **video_kwargs)

            # print(videos, model_inputs["input_ids"].size(-1))


            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {
                "videos": videos,
                "min_pixels": self.min_pixels,
                "max_pixels": self.max_pixels,
                "max_frames": max_frames_per_video,
                "min_frames": self.min_frames,
                "video_fps": _eff_fps
            }
            token_stats_grid_video = model_inputs.get("video_grid_thw", None)
        else:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]


        if "images" in example:
            example.pop("images")
        elif "videos" in example:
            example.pop("videos")

        # print(example)

        # print(example.keys())


        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen-vl mrope
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from ..models.transformers.qwen3_vl import get_rope_index



            else:
                from ..models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                attention_mask=attention_mask,
            )  # (3, seq_length)
            text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)  # (1, seq_length)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)  # (4, seq_length)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        prompt_len_before_pad = int(attention_mask.sum().item())
        video_token_id = getattr(self.processor, "video_token_id", None) if self.processor is not None else None
        image_token_id = getattr(self.processor, "image_token_id", None) if self.processor is not None else None
        video_tokens_before_pad = _count_token(input_ids, video_token_id)
        image_tokens_before_pad = _count_token(input_ids, image_token_id)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        raw_prompt_len_before_trunc = len(raw_prompt_ids)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        prompt_len_after_pad = int(attention_mask.sum().item())
        video_tokens_after_pad = _count_token(input_ids, video_token_id)
        image_tokens_after_pad = _count_token(input_ids, image_token_id)
        visual_tokens_after_pad = video_tokens_after_pad + image_tokens_after_pad

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids
        example["prompt_token_count"] = torch.tensor(prompt_len_after_pad, dtype=torch.long)
        example["video_visual_token_count"] = torch.tensor(video_tokens_after_pad, dtype=torch.long)
        example["image_visual_token_count"] = torch.tensor(image_tokens_after_pad, dtype=torch.long)
        example["visual_token_count"] = torch.tensor(visual_tokens_after_pad, dtype=torch.long)
        example["ground_truth"] = example.pop(self.answer_key)

        _maybe_log_visual_token_debug(
            index=index,
            prompt_len_before_pad=prompt_len_before_pad,
            prompt_len_after_pad=prompt_len_after_pad,
            video_tokens_before_pad=video_tokens_before_pad,
            video_tokens_after_pad=video_tokens_after_pad,
            image_tokens_before_pad=image_tokens_before_pad,
            image_tokens_after_pad=image_tokens_after_pad,
            raw_prompt_len=raw_prompt_len_before_trunc,
            max_prompt_length=self.max_prompt_length,
            video_grid_thw=token_stats_grid_video,
            image_grid_thw=token_stats_grid_image,
        )

        # print(example)
        # print(input_ids.shape)


        # print(example)
        return example
