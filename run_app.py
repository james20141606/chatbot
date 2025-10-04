#!/usr/bin/env python3
"""
ChatBOT.EDU - 启动脚本
运行这个脚本来启动ChatBOT.EDU应用
"""

import subprocess
import sys
import os

def main():
    """启动Streamlit应用"""
    print("🎓 启动 ChatBOT.EDU...")
    print("=" * 50)
    
    # 检查是否安装了streamlit
    try:
        import streamlit
        print(f"✅ Streamlit 已安装 (版本: {streamlit.__version__})")
    except ImportError:
        print("❌ Streamlit 未安装，请运行: pip install streamlit")
        sys.exit(1)
    
    # 检查环境变量
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  警告: 未设置 OPENAI_API_KEY 环境变量")
        print("   请在 .streamlit/secrets.toml 中设置或设置环境变量")
        print("   示例: export OPENAI_API_KEY='your-api-key-here'")
    
    print("\n🚀 启动应用...")
    print("📱 应用将在浏览器中自动打开")
    print("🔗 如果未自动打开，请访问: http://localhost:8501")
    print("⏹️  按 Ctrl+C 停止应用")
    print("=" * 50)
    
    try:
        # 启动streamlit应用
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "chatbot_edu.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 应用已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
