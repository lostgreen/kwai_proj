#!/usr/bin/env python3
"""诊断 GPU filler 进程检测 — 在训练运行中执行，排查 pynvml 是否正常。

用法:
    python diagnose_filler.py          # 检查所有 GPU
    python diagnose_filler.py --gpu 0  # 检查指定 GPU
"""

import os
import subprocess
import sys

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("GPU Filler 诊断")
    print("=" * 70)

    # 1. 环境变量
    print("\n[1] 环境变量")
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "(未设置)")
    print(f"  CUDA_VISIBLE_DEVICES = {cvd}")
    print(f"  VERL_GPU_SIGNAL_PATH = {os.environ.get('VERL_GPU_SIGNAL_PATH', '/tmp/verl_gpu_phase (default)')}")

    # 2. 信号文件
    signal_path = os.environ.get("VERL_GPU_SIGNAL_PATH", "/tmp/verl_gpu_phase")
    print(f"\n[2] 信号文件: {signal_path}")
    if os.path.exists(signal_path):
        try:
            mtime = os.path.getmtime(signal_path)
            import time
            age = time.time() - mtime
            with open(signal_path) as f:
                content = f.read().strip()
            print(f"  内容: '{content}'")
            print(f"  修改时间: {time.ctime(mtime)} ({age:.1f}s ago)")
        except Exception as e:
            print(f"  读取失败: {e}")
    else:
        print("  文件不存在 (训练可能未运行或已退出)")

    # 3. pynvml 初始化
    print("\n[3] pynvml 初始化")
    try:
        import pynvml
        pynvml.nvmlInit()
        driver = pynvml.nvmlSystemGetDriverVersion()
        device_count = pynvml.nvmlDeviceGetCount()
        print(f"  驱动版本: {driver}")
        print(f"  设备数量: {device_count}")
    except Exception as e:
        print(f"  pynvml 初始化失败: {e}")
        print("  >>> 这就是问题所在！pynvml 无法工作。")
        sys.exit(1)

    # 4. NVML index 映射 (同 gpu_filler.py 的 resolve_nvml_indices)
    print("\n[4] NVML Index 映射")
    cvd_val = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd_val:
        physical = [int(x.strip()) for x in cvd_val.split(",") if x.strip()]
        print(f"  CUDA_VISIBLE_DEVICES 映射: CUDA idx → NVML idx")
        for i, p in enumerate(physical):
            print(f"    CUDA {i} → NVML {p}")
    else:
        print("  CUDA_VISIBLE_DEVICES 未设置 → CUDA idx = NVML idx")

    # 5. 检查 filler 进程
    print("\n[5] 当前 filler 进程")
    result = subprocess.run(["pgrep", "-af", "gpu_filler.py"], capture_output=True, text=True)
    if result.stdout.strip():
        for line in result.stdout.strip().split("\n"):
            print(f"  {line}")
        filler_pids = [int(line.split()[0]) for line in result.stdout.strip().split("\n") if line.strip()]
    else:
        print("  没有找到 filler 进程")
        filler_pids = []

    # 6. 逐 GPU 进程检测
    print("\n[6] 逐 GPU 进程检测 (pynvml)")
    my_pid = os.getpid()
    gpus = [args.gpu] if args.gpu is not None else list(range(device_count))

    for nvml_idx in gpus:
        print(f"\n  --- NVML GPU {nvml_idx} ---")
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(nvml_idx)
            name = pynvml.nvmlDeviceGetName(h)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            print(f"  设备: {name}")
            print(f"  利用率: GPU={util.gpu}%, Memory={util.memory}%")
            print(f"  显存: {mem.used / 1e9:.2f} / {mem.total / 1e9:.2f} GB")
        except Exception as e:
            print(f"  获取设备信息失败: {e}")

        # 进程列表
        try:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
            if not procs:
                print(f"  CUDA 进程: 无 <<<< 如果训练正在运行，这就是问题！")
            else:
                print(f"  CUDA 进程 ({len(procs)}):")
                for p in procs:
                    alive = True
                    try:
                        os.kill(p.pid, 0)
                    except ProcessLookupError:
                        alive = False
                    except PermissionError:
                        alive = True  # 进程存在但无权发信号

                    is_filler = p.pid in filler_pids
                    is_me = p.pid == my_pid
                    mem_mb = (p.usedGpuMemory or 0) / 1e6
                    labels = []
                    if is_filler:
                        labels.append("FILLER")
                    if is_me:
                        labels.append("THIS_SCRIPT")
                    if not alive:
                        labels.append("ZOMBIE")

                    # 尝试获取进程命令行
                    try:
                        cmdline = open(f"/proc/{p.pid}/cmdline", "rb").read().decode(errors="replace").replace("\0", " ")[:80]
                    except Exception:
                        cmdline = "(无法读取)"

                    label_str = f" [{', '.join(labels)}]" if labels else ""
                    print(f"    PID {p.pid:>7}: {mem_mb:.0f} MB{label_str}")
                    print(f"      cmd: {cmdline}")
        except pynvml.NVMLError as e:
            print(f"  获取进程列表失败: {e}")
            print(f"  >>> 这可能是权限问题或驱动 bug！")

    # 7. 对比 nvidia-smi 的进程列表
    print("\n[7] nvidia-smi 进程列表 (对比)")
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        lines = result.stdout.strip().split("\n")
        if lines and lines[0]:
            for line in lines:
                print(f"  {line}")
        else:
            print("  nvidia-smi 没有找到计算进程")
    else:
        print(f"  nvidia-smi 执行失败: {result.stderr}")

    # 8. torch CUDA 检测
    print("\n[8] torch.cuda 检测")
    try:
        import torch
        print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
        print(f"  torch.cuda.device_count(): {torch.cuda.device_count()}")
        for i in range(min(torch.cuda.device_count(), len(gpus) if args.gpu is not None else 8)):
            print(f"  torch.cuda.get_device_name({i}): {torch.cuda.get_device_name(i)}")
    except Exception as e:
        print(f"  torch 异常: {e}")

    print("\n" + "=" * 70)
    print("诊断完成。如果 [6] 显示训练 GPU 上'无 CUDA 进程'但 [7] nvidia-smi")
    print("能看到进程，说明 pynvml 的 nvmlDeviceGetComputeRunningProcesses 在")
    print("此环境下无法正常工作（可能是容器/cgroup/MIG 隔离导致）。")
    print("=" * 70)


if __name__ == "__main__":
    main()
