import torch
import os

# 检查点文件路径
checkpoint_path = '/home/teai/gwf_file/YOLOv11-RGBT/runs/M3FD/M3FD-yolo11n-RGBT-midfusion-3/weights/last.pt'

# 加载检查点文件
ckpt = torch.load(checkpoint_path, weights_only=False)

# 修改epoch参数为149
ckpt['epoch'] = 99

# 修改epochs参数为300
ckpt['train_args']['epochs'] = 130

# 保存修改后的检查点文件
torch.save(ckpt, checkpoint_path)

print(f"成功将 {checkpoint_path} 中的epoch参数修改为99，epochs参数修改为130")
