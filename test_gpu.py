import torch

# 总显存
total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # 约8GB

# 当前已分配显存
allocated = torch.cuda.memory_allocated() / 1024**3

# 最大可用显存（减去驱动预留）
available = total_memory - 1.5  # 减去驱动占用的约1.5GB

print(f"总显存: {total_memory:.2f} GB")
print(f"可用显存: {available:.2f} GB")