#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ultralytics import YOLO
import torch
from pathlib import Path
import os
import shutil

def load_model(yaml_path,verbose=True):
    """
    加载YOLOv8模型并打印其参数信息
    
    Args:
        verbose: 是否打印详细信息
    
    Returns:
        加载的YOLO模型对象
    """

    
    # 检查YAML文件是否存在
    if not os.path.exists(yaml_path):
        print(f"错误: 配置文件 '{yaml_path}' 不存在")
        # 尝试推断正确的路径
        base_dir = Path(__file__).parent
        possible_paths = list(base_dir.glob("**/yolov8.yaml"))
        if possible_paths:
            yaml_path = str(possible_paths[0])
            print(f"找到可能的配置文件: {yaml_path}")
        else:
            print("无法找到配置文件")
            return None
    
    print(f"\n正在加载 YOLOv8 模型: {yaml_path}")
    
    # 加载模型（设置verbose=True显示详细信息）
    model = YOLO(yaml_path, verbose=verbose)
    
    if verbose:
        # 打印模型结构
        print("\n模型结构:")
        print(f"类型: {type(model)}")
        
        if hasattr(model, 'model'):
            print(f"模型类型: {type(model.model)}")
            # 打印模型参数数量
            model_params = sum(p.numel() for p in model.model.parameters())
            print(f"参数数量: {model_params:,}")
            
            # 打印模型的一些关键属性
            print("\n模型主要属性:")
            for attr in dir(model):
                if not attr.startswith('_') and attr not in ['model', 'predictor', 'trainer']:
                    try:
                        value = getattr(model, attr)
                        if not callable(value):
                            print(f"{attr}: {value}")
                    except:
                        pass
    
    return model

def save_model(model, save_path):
    """
    正确保存YOLOv8模型
    """
    save_dir = Path("exported_models")
    save_dir.mkdir(exist_ok=True)

    full_save_path = save_dir / save_path
    print(f"开始保存模型到: {full_save_path}")

    print("用 model.save() 正确保存...")
    model.save(str(full_save_path))  # 用model自带的save方法保存！
    print(f"成功保存模型到: {full_save_path}")
    return True


def main():
    # 加载模型并显示详细信息
    model = load_model(yaml_path = "ultralytics/cfg/models/v8/yolov8_cbam.yaml",verbose=True)
    
    if model is None:
        print("加载模型失败")
        return None
    
    print("\n模型已成功加载！")
    
    # 保存模型
    save_model_enabled = True
    if save_model_enabled:
        save_model(model, "yolov8_loaded_cbam.pt")
    
    # 返回加载的模型以便进一步使用
    return model

if __name__ == "__main__":
    model = main()
    
    if model is not None:
        # 这里可以对加载的模型做进一步操作
        # 例如: 进行预测、提取特征等
        print("\n模型加载完成，可以进行进一步操作") 