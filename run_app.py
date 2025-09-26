#!/usr/bin/env python3
"""
使用uv运行多LLM比较器应用的启动脚本
"""

import subprocess
import sys
from pathlib import Path


def main():
    """启动应用"""
    print("🚀 启动多LLM模型比较器...")
    print("📦 使用uv运行应用")
    
    # 检查uv是否可用
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
        print("✅ uv已安装")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ 错误: uv未安装或不在PATH中")
        print("💡 请先安装uv: https://docs.astral.sh/uv/getting-started/installation/")
        sys.exit(1)
    
    # 获取主应用文件路径
    app_path = Path(__file__).parent / "src" / "multi_llm_comparator" / "main.py"
    
    if not app_path.exists():
        print(f"❌ 错误: 找不到应用文件 {app_path}")
        sys.exit(1)
    
    print(f"📂 应用路径: {app_path}")
    print("🌐 启动Streamlit应用...")
    print("💡 应用将在浏览器中自动打开")
    print("⏹️  按 Ctrl+C 停止应用")
    print("-" * 50)
    
    try:
        # 使用uv运行streamlit
        subprocess.run([
            "uv", "run", "streamlit", "run", str(app_path),
            "--server.address", "localhost",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ], check=True)
    except KeyboardInterrupt:
        print("\n⏹️  应用已停止")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 启动失败: {e}")
        print("💡 请检查依赖是否正确安装")
        sys.exit(1)


if __name__ == "__main__":
    main()