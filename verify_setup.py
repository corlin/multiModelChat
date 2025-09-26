#!/usr/bin/env python3
"""
验证uv环境和项目设置的脚本
"""

import subprocess
import sys
import importlib
from pathlib import Path


def check_uv():
    """检查uv是否可用"""
    print("🔍 检查uv...")
    try:
        result = subprocess.run(["uv", "--version"], capture_output=True, text=True, check=True)
        print(f"✅ uv版本: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ uv未安装或不在PATH中")
        print("💡 请安装uv: https://docs.astral.sh/uv/getting-started/installation/")
        return False


def check_python_version():
    """检查Python版本"""
    print("🐍 检查Python版本...")
    version = sys.version_info
    if version >= (3, 12):
        print(f"✅ Python版本: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python版本过低: {version.major}.{version.minor}.{version.micro}")
        print("💡 需要Python 3.12或更高版本")
        return False


def check_project_structure():
    """检查项目结构"""
    print("📁 检查项目结构...")
    
    required_files = [
        "pyproject.toml",
        "uv.lock",
        "src/multi_llm_comparator/main.py",
        "src/multi_llm_comparator/core/__init__.py",
        "src/multi_llm_comparator/services/__init__.py",
        "src/multi_llm_comparator/inferencers/__init__.py",
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ 缺少必要文件:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    else:
        print("✅ 项目结构完整")
        return True


def check_dependencies():
    """检查关键依赖是否可导入"""
    print("📦 检查关键依赖...")
    
    dependencies = [
        ("streamlit", "Streamlit"),
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("psutil", "psutil"),
        ("pydantic", "Pydantic"),
    ]
    
    missing_deps = []
    for module_name, display_name in dependencies:
        try:
            importlib.import_module(module_name)
            print(f"✅ {display_name}")
        except ImportError:
            print(f"❌ {display_name}")
            missing_deps.append(display_name)
    
    if missing_deps:
        print("💡 运行 'uv sync' 安装缺少的依赖")
        return False
    else:
        print("✅ 所有关键依赖可用")
        return True


def check_optional_dependencies():
    """检查可选依赖"""
    print("🔧 检查可选依赖...")
    
    optional_deps = [
        ("llama_cpp", "llama-cpp-python (GGUF支持)"),
    ]
    
    for module_name, display_name in optional_deps:
        try:
            importlib.import_module(module_name)
            print(f"✅ {display_name}")
        except ImportError:
            print(f"⚠️  {display_name} - 可选，GGUF模型需要")


def check_model_directories():
    """检查模型目录"""
    print("📂 检查模型目录...")
    
    model_dirs = ["models", "models/pytorch", "models/gguf"]
    
    for dir_path in model_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✅ {dir_path}")
        else:
            print(f"⚠️  {dir_path} - 将自动创建")
            try:
                path.mkdir(parents=True, exist_ok=True)
                print(f"✅ 已创建 {dir_path}")
            except Exception as e:
                print(f"❌ 创建 {dir_path} 失败: {e}")


def run_quick_test():
    """运行快速测试"""
    print("🧪 运行快速测试...")
    
    try:
        # 测试导入主模块
        sys.path.insert(0, str(Path("src")))
        from multi_llm_comparator.core.models import ModelType, ModelInfo
        from multi_llm_comparator.services.model_manager import ModelManager
        
        print("✅ 核心模块导入成功")
        
        # 测试基本功能
        manager = ModelManager()
        print("✅ 模型管理器创建成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 快速测试失败: {e}")
        return False


def main():
    """主函数"""
    print("🔍 多LLM比较器环境验证")
    print("=" * 50)
    
    checks = [
        ("uv可用性", check_uv),
        ("Python版本", check_python_version),
        ("项目结构", check_project_structure),
        ("关键依赖", check_dependencies),
        ("可选依赖", check_optional_dependencies),
        ("模型目录", check_model_directories),
        ("快速测试", run_quick_test),
    ]
    
    passed = 0
    total = len(checks)
    
    for name, check_func in checks:
        print(f"\n📋 {name}")
        if check_func():
            passed += 1
        print("-" * 30)
    
    print(f"\n📊 验证结果: {passed}/{total} 项通过")
    
    if passed == total:
        print("🎉 环境验证完全通过！")
        print("💡 可以运行以下命令启动应用:")
        print("   uv run python run_app.py")
        print("   或")
        print("   uv run streamlit run src/multi_llm_comparator/main.py")
    else:
        print("⚠️  部分检查未通过，请根据上面的提示进行修复")
        if passed >= total - 2:
            print("💡 大部分检查通过，应用可能仍可正常运行")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)