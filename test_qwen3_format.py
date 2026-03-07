import sys
import os
import json
import torch
import argparse
from transformers import AutoProcessor
from qwen_vl_utils.vision_process import fetch_video

# 根据用户的模型路径调整
MODEL_PATH = "/home/xuboshen/models/Qwen3-VL-4B-Instruct"

def run_single_inference(jsonl_path, index=0, max_frames=256, video_fps=2.0, max_pixels=12288, min_pixels=4096):
    print(f"=== 开始单样本推理测试 ===")
    
    # 1. 读取 JSONL 单条数据
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        if index >= len(lines):
            print(f"Error: index {index} 超出数据范围 (0-{len(lines)-1})")
            return
        
        data = json.loads(lines[index])
        print(f"成功读取数据第 {index} 条。")
    except Exception as e:
        print(f"读取数据失败: {e}")
        return

    # 2. 提取 prompt 和视频路径
    prompt = data.get("prompt", "")
    videos = data.get("videos", [])
    ground_truth = data.get("answer", "")
    
    if not videos:
        print("未检测到视频输入。")
        return
        
    video_path = videos[0] # 取主视频
    print(f"\n[输入信息]")
    print(f"Video: {video_path}")
    print(f"Prompt: \n{prompt[:300]}...\n(截断显示, 长度={len(prompt)})")
    print(f"Ground Truth:\n{ground_truth}")

    # 3. 处理视频 (跟 Dataset 阶段一致)
    vision_info = {
        "video": video_path, 
        "min_pixels": min_pixels, 
        "max_pixels": max_pixels, 
        "max_frames": max_frames, 
        "fps": video_fps
    }
    
    print("\n[加载模型与处理器...]")
    try:
        processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
        # 根据 verl/workers/fsdp_workers.py 中真实加载逻辑
        from transformers import AutoConfig, AutoModelForImageTextToText
        
        # 1. 先用 AutoConfig 拉取配置
        model_config = AutoConfig.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            bos_token_id=processor.tokenizer.bos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id
        )
        
        # 2. Qwen3-VL 的映射应该是 AutoModelForImageTextToText，而不是 CausalLM 
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_PATH, 
            config=model_config,
            torch_dtype=torch.bfloat16, 
            attn_implementation="flash_attention_2",
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()
        
    except Exception as e:
        print(f"模型加载报错: {e}")
        return

    print("\n[处理多模态输入...]")
    try:
        processed_video, return_fps = fetch_video(
            vision_info, image_patch_size=16, return_video_sample_fps=True, return_video_metadata=True
        )
        
        video_metadata = None
        if isinstance(processed_video, tuple):
            processed_video, video_metadata = processed_video
            
        messages = [
            {"role": "user", "content": [
                {"type": "video"}, 
                {"type": "text", "text": prompt.replace("<video>", "")} # 注意替换为标准的 Qwen message 格式
            ]}
        ]
        
        text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        
        inputs = processor(
            text=[text_prompt], 
            videos=[processed_video], 
            video_metadata=[video_metadata] if video_metadata else None,
            return_tensors="pt",
            do_resize=False,
            do_sample_frames=False
        )
        
        # 移动到模型对应的设备
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        print(f"Input IDs shape: {inputs['input_ids'].shape}")
        
    except Exception as e:
        print(f"输入处理报错: {e}")
        return

    print("\n[生成输出...]")
    try:
        # 准备生成参数，设置典型的 RLHF 搜索参数 (如 temperature)
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )
            
        # 裁掉Prompt部分，只保留生成的答案
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        print("\n================ 模型输出 ================\n")
        print(output_text)
        print("\n==========================================")
        
        # 简单检查格式
        has_tag = "<events>" in output_text and "</events>" in output_text
        print(f"\n[格式分析] 包含 <events> 标签: {has_tag}")
        
    except Exception as e:
        print(f"生成报错: {e}")
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, default="/home/xuboshen/zgw/OneThinker/EasyR1/proxy_data/youcook2_train_easyr1.jsonl", help="数据路径")
    parser.add_argument("--index", type=int, default=0, help="提取第几条数据进行测试")
    args = parser.parse_args()
    
    run_single_inference(args.jsonl, args.index)
