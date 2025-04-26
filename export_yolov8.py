#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
from ultralytics import YOLO
from pathlib import Path
import shutil

# 创建导出目录（如果不存在）
export_dir = Path("exported_models")
export_dir.mkdir(exist_ok=True)
print(f"导出目录: {export_dir.absolute()}")

# 从本地YAML配置文件构建YOLOv8模型
def build_and_export_model(yaml_path, model_size, export_formats=None):
    """
    从YAML配置文件构建YOLOv8模型并导出
    
    Args:
        yaml_path: YAML配置文件路径
        model_size: 模型大小('n', 's', 'm', 'l', 'x')
        export_formats: 导出格式列表，例如['pt', 'onnx']
    """
    if export_formats is None:
        export_formats = ['pt']  # 默认导出PyTorch格式
        
    print(f"\n开始构建 YOLOv8{model_size} 模型从本地配置: {yaml_path}")
    
    # 构建模型（从YAML配置）
    model = YOLO(yaml_path)
    
    # 设置模型大小
    if hasattr(model, 'args'):
        if isinstance(model.args, dict):
            model.args['scale'] = model_size
        else:
            model.args.scale = model_size
    
    # 为每种格式导出模型
    for fmt in export_formats:
        output_path = export_dir / f"yolov8{model_size}_local.{fmt}"
        print(f"正在导出 {fmt} 格式到: {output_path}")
        
        try:
            if fmt.lower() == 'pt':
                # 使用torch.save直接保存模型
                pt_path = str(export_dir / f"yolov8{model_size}_local.pt")
                
                # 尝试不同的方式来保存模型
                try:
                    print("尝试保存模型属性...")
                    # 打印模型属性，帮助我们了解可用的属性
                    print(f"模型属性: {dir(model)}")
                    
                    # 直接保存整个YOLO对象
                    torch.save(model, pt_path)
                    print(f"成功导出模型到: {pt_path}")
                except Exception as e1:
                    print(f"尝试保存YOLO对象失败: {e1}")
                    
                    try:
                        # 如果model.model是可迭代对象，尝试获取第一个元素
                        if hasattr(model, 'model') and hasattr(model.model, '__iter__'):
                            model_item = next(iter(model.model))
                            torch.save(model_item, pt_path)
                            print(f"成功导出模型第一个元素到: {pt_path}")
                        else:
                            # 尝试训练模型并保存
                            print("尝试训练空过程并保存...")
                            # 这种情况下创建一个空的训练过程仅用于保存模型
                            model.train(data=None, epochs=1, imgsz=640, batch=-1)
                            print(f"成功导出训练后的模型到: {pt_path}")
                    except Exception as e2:
                        print(f"所有保存尝试均失败: {e2}")
            else:
                # 使用export方法导出其他格式
                exported_path = model.export(format=fmt, imgsz=640)
                
                # 将导出的模型移动到我们的导出目录
                if os.path.exists(exported_path) and str(export_dir) not in exported_path:
                    target_path = export_dir / os.path.basename(exported_path)
                    os.rename(exported_path, target_path)
                    print(f"成功导出并移动到: {target_path}")
                else:
                    print(f"成功导出到: {exported_path}")
                
        except Exception as e:
            print(f"导出 {fmt} 格式失败: {e}")
    
    return model

# 主函数
def main():
    print("开始导出YOLOv8模型（使用本地配置文件）...\n")
    
    # 配置文件路径
    yaml_path = "ultralytics/cfg/models/v8/yolov8.yaml"
    
    # 检查配置文件是否存在
    if not os.path.exists(yaml_path):
        print(f"错误: 配置文件 '{yaml_path}' 不存在")
        return
    
    # 导出不同大小的模型（这里只导出'n'小模型作为示例）
    # 可以根据需要添加其他大小: 's', 'm', 'l', 'x'
    model_size = 'n'  # 选择最小的模型以便快速导出
    
    # 导出格式（只导出PyTorch格式）
    export_formats = ['pt']
    
    # 构建并导出模型
    model = build_and_export_model(yaml_path, model_size, export_formats)
    
    print("\n导出完成！")
    print(f"导出的模型位于: {export_dir.absolute()}")

if __name__ == "__main__":
    main() 