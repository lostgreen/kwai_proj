#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from PIL import Image
import numpy as np

try:
    from decord import VideoReader, cpu
except ImportError:
    print("Warning: 未检测到 decord，请通过 `pip install decord` 安装以实现高速多线程抽帧。")
    import sys
    sys.exit(1)


def process_single_video(video_path, output_dir, video_fps, max_frames, resize_min=None, resize_max=None):
    """
    抽取单一视频的帧并存为相对应的序列图
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    # 创建以视频命名的子文件夹
    video_out_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_out_dir, exist_ok=True)
    
    out_prefix = os.path.join(video_out_dir, "frame")
    
    # 避免重复抽取
    # 如果已经存在某种特征，这里可以使用缓存。简单起见每次覆盖/探测
    # 如果需要极其严谨的跳过，可以检查是否已经生成了最后一帧
    
    saved_frames = []
    
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        original_fps = vr.get_avg_fps()
        total_frames = len(vr)
        
        # 计算需要采样的帧号 (按指定 FPS 抽取)
        if video_fps > 0:
            step = original_fps / video_fps
            indices = [int(i * step) for i in range(int(total_frames / step))]
        else:
            indices = list(range(total_frames))
            
        # 截断超长视频
        if len(indices) > max_frames:
            indices = indices[:max_frames]
            
        frames = vr.get_batch(indices).asnumpy()
        
        # 将抽出的 numpy array 帧保存为 jpeg
        for i, frame_np in enumerate(frames):
            img_path = f"{out_prefix}_{i:04d}.jpg"
            img = Image.fromarray(frame_np)
            
            # (可选) 在脱机阶段直接做图像 Resize 以极致压缩磁盘
            if resize_max is not None and (img.width * img.height) > resize_max:
                import math
                resize_factor = math.sqrt(resize_max / (img.width * img.height))
                new_w, new_h = int(img.width * resize_factor), int(img.height * resize_factor)
                img = img.resize((new_w, new_h), Image.LANCZOS)
                
            img.save(img_path, quality=90)
            saved_frames.append(img_path)
            
        return saved_frames
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return None


def worker(item_string, output_dir, video_fps, max_frames, max_pixels):
    item = json.loads(item_string)
    videos = item.get("videos", [])
    if not videos:
        return item
    
    videos_out = []
    for vid in videos:
        extracted = process_single_video(
            video_path=vid, 
            output_dir=output_dir, 
            video_fps=video_fps, 
            max_frames=max_frames,
            resize_max=max_pixels
        )
        if extracted is None:
            return None 
            
        videos_out.append(extracted) # A list of image paths represents one video
    
    # 保持为 videos 字段，但内容变成了帧路径的列表，触发 qwen_vl_utils 无解码模式
    item["videos"] = videos_out
    return item

def main():
    parser = argparse.ArgumentParser(description="将基于 Video 的 JSONL 数据集转换为全 Image List 数据集")
    parser.add_argument("--input_jsonl", required=True, help="输入的包含 videos 字段的 jsonl 文件")
    parser.add_argument("--output_jsonl", required=True, help="转化后的输出 jsonl 文件")
    parser.add_argument("--output_frame_dir", required=True, help="离线提取保存 JPG 的宿主目录")
    parser.add_argument("--video_fps", type=float, default=2.0)
    parser.add_argument("--max_frames", type=int, default=256)
    parser.add_argument("--max_pixels", type=int, default=12288, help="提前根据模型设定进行裁剪节约磁盘空间")
    parser.add_argument("--num_workers", type=int, default=16, help="并发处理线程数")
    args = parser.parse_args()

    os.makedirs(args.output_frame_dir, exist_ok=True)
    
    with open(args.input_jsonl, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    print(f"Total entries to process: {len(lines)}")
    
    results = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(
            worker, line, args.output_frame_dir, args.video_fps, args.max_frames, args.max_pixels
        ): i for i, line in enumerate(lines)}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting Videos"):
            res = future.result()
            if res is not None:
                results.append(res)
                
    # 保存替换后的新 jsonl
    with open(args.output_jsonl, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
            
    print(f"Conversion complete! Original lines: {len(lines)}, Filtered healthy lines: {len(results)}")
    print(f"Check your novel JSONL dataset at: {args.output_jsonl}")

if __name__ == '__main__':
    main()
