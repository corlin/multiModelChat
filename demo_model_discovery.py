#!/usr/bin/env python3
"""
模型发现和管理功能演示脚本
"""

import os
import tempfile
from pathlib import Path

from src.multi_llm_comparator.services import ModelManager, ModelFileScanner
from src.multi_llm_comparator.core.models import ModelType


def create_demo_models():
    """创建演示用的模型文件"""
    temp_dir = tempfile.mkdtemp(prefix="demo_models_")
    temp_path = Path(temp_dir)
    
    # 创建不同类型的模型文件
    (temp_path / "pytorch_model.bin").write_text("fake pytorch model content")
    (temp_path / "safetensors_model.safetensors").write_text("fake safetensors content")
    (temp_path / "gguf_model_q4_0.gguf").write_text("fake gguf model content")
    (temp_path / "another_model.pt").write_text("another pytorch model")
    
    # 创建子目录
    sub_dir = temp_path / "submodels"
    sub_dir.mkdir()
    (sub_dir / "sub_model.gguf").write_text("model in subdirectory")
    
    # 创建非模型文件（应该被忽略）
    (temp_path / "readme.txt").write_text("not a model file")
    (temp_path / "config.json").write_text('{"model_type": "test"}')
    
    return str(temp_path)


def demo_scanner():
    """演示模型扫描器功能"""
    print("=== 模型文件扫描器演示 ===")
    
    # 创建演示模型
    demo_dir = create_demo_models()
    print(f"创建演示模型目录: {demo_dir}")
    
    # 创建扫描器
    scanner = ModelFileScanner()
    
    # 扫描模型
    print("\n扫描模型文件...")
    result = scanner.scan_directory(demo_dir, recursive=True)
    
    print(f"扫描结果:")
    print(f"  - 扫描文件数: {result.scanned_files}")
    print(f"  - 有效模型数: {result.valid_models}")
    print(f"  - 错误数: {len(result.errors)}")
    
    if result.errors:
        print("  - 错误信息:")
        for error in result.errors:
            print(f"    * {error}")
    
    print(f"\n发现的模型:")
    for model in result.models:
        print(f"  - {model.name} ({model.model_type.value})")
        print(f"    路径: {model.path}")
        print(f"    大小: {model.size} bytes")
        print(f"    配置: {model.config}")
        print()
    
    # 清理
    import shutil
    shutil.rmtree(demo_dir)
    
    return result.models


def demo_manager():
    """演示模型管理器功能"""
    print("=== 模型管理器演示 ===")
    
    # 创建演示模型
    demo_dir = create_demo_models()
    print(f"创建演示模型目录: {demo_dir}")
    
    # 创建管理器
    manager = ModelManager(cache_file="demo_cache.json")
    
    # 扫描模型
    print("\n使用管理器扫描模型...")
    result = manager.scan_models([demo_dir])
    
    print(f"管理器中的模型数量: {len(manager.get_available_models())}")
    
    # 按类型分组显示
    pytorch_models = manager.get_models_by_type(ModelType.PYTORCH)
    gguf_models = manager.get_models_by_type(ModelType.GGUF)
    
    print(f"\nPyTorch模型 ({len(pytorch_models)}个):")
    for model in pytorch_models:
        print(f"  - {model.name}")
    
    print(f"\nGGUF模型 ({len(gguf_models)}个):")
    for model in gguf_models:
        print(f"  - {model.name}")
    
    # 选择模型进行比较
    available_models = manager.get_available_models()
    if len(available_models) >= 2:
        selected_ids = [model.id for model in available_models[:2]]
        print(f"\n选择前2个模型进行比较: {[model.name for model in available_models[:2]]}")
        
        manager.select_models(selected_ids)
        selected_models = manager.get_selected_models()
        
        print(f"已选择的模型:")
        for model in selected_models:
            print(f"  - {model.name} ({model.model_type.value})")
    
    # 显示统计信息
    stats = manager.get_model_statistics()
    print(f"\n模型统计信息:")
    print(f"  - 总模型数: {stats['total_models']}")
    print(f"  - PyTorch模型: {stats['pytorch_models']}")
    print(f"  - GGUF模型: {stats['gguf_models']}")
    print(f"  - 已选择模型: {stats['selected_models']}")
    print(f"  - 最大选择数: {stats['max_selection']}")
    print(f"  - 总大小: {stats['total_size_mb']:.2f} MB")
    
    # 清理
    import shutil
    shutil.rmtree(demo_dir)
    
    # 清理缓存文件
    cache_file = Path("demo_cache.json")
    if cache_file.exists():
        cache_file.unlink()


def main():
    """主函数"""
    print("多LLM模型比较器 - 模型发现和管理功能演示\n")
    
    try:
        # 演示扫描器
        demo_scanner()
        
        print("\n" + "="*50 + "\n")
        
        # 演示管理器
        demo_manager()
        
        print("\n演示完成！")
        
    except Exception as e:
        print(f"演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()