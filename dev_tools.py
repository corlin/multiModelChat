#!/usr/bin/env python3
"""
使用uv的开发工具脚本
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run_command(cmd, description):
    """运行命令并处理错误"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {description}完成")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description}失败: {e}")
        if e.stderr:
            print(f"错误输出: {e.stderr}")
        return False


def install_deps():
    """安装依赖"""
    print("📦 安装项目依赖...")
    return run_command(["uv", "sync"], "同步依赖")


def run_tests():
    """运行测试"""
    print("🧪 运行测试...")
    return run_command(["uv", "run", "pytest", "tests/", "-v"], "运行测试")


def run_linting():
    """运行代码检查"""
    print("🔍 运行代码检查...")
    success = True
    
    # 运行ruff检查
    if not run_command(["uv", "run", "ruff", "check", "src/"], "Ruff代码检查"):
        success = False
    
    # 运行black格式检查
    if not run_command(["uv", "run", "black", "--check", "src/"], "Black格式检查"):
        success = False
    
    return success


def format_code():
    """格式化代码"""
    print("🎨 格式化代码...")
    success = True
    
    # 使用ruff修复
    if not run_command(["uv", "run", "ruff", "check", "--fix", "src/"], "Ruff自动修复"):
        success = False
    
    # 使用black格式化
    if not run_command(["uv", "run", "black", "src/"], "Black代码格式化"):
        success = False
    
    return success


def run_type_check():
    """运行类型检查"""
    print("🔬 运行类型检查...")
    return run_command(["uv", "run", "mypy", "src/"], "MyPy类型检查")


def run_streaming_test():
    """运行流式输出测试"""
    print("🌊 运行流式输出测试...")
    return run_command(["uv", "run", "python", "test_streaming_output.py"], "流式输出测试")


def start_app():
    """启动应用"""
    print("🚀 启动应用...")
    app_path = Path(__file__).parent / "src" / "multi_llm_comparator" / "main.py"
    
    try:
        subprocess.run([
            "uv", "run", "streamlit", "run", str(app_path),
            "--server.address", "localhost",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n⏹️  应用已停止")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="多LLM比较器开发工具")
    parser.add_argument("command", choices=[
        "install", "test", "lint", "format", "typecheck", "streaming-test", "run", "all"
    ], help="要执行的命令")
    
    args = parser.parse_args()
    
    print("🛠️  多LLM比较器开发工具")
    print("=" * 50)
    
    if args.command == "install":
        install_deps()
    elif args.command == "test":
        run_tests()
    elif args.command == "lint":
        run_linting()
    elif args.command == "format":
        format_code()
    elif args.command == "typecheck":
        run_type_check()
    elif args.command == "streaming-test":
        run_streaming_test()
    elif args.command == "run":
        start_app()
    elif args.command == "all":
        print("🔄 运行完整开发流程...")
        success = True
        
        if not install_deps():
            success = False
        if not format_code():
            success = False
        if not run_linting():
            success = False
        if not run_type_check():
            success = False
        if not run_tests():
            success = False
        if not run_streaming_test():
            success = False
        
        if success:
            print("\n🎉 所有检查通过！")
        else:
            print("\n⚠️  部分检查失败，请查看上面的错误信息")
            sys.exit(1)


if __name__ == "__main__":
    main()