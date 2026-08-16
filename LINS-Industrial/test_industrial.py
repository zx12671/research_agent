"""
test_industrial.py: 快速测试 LINS-Industrial 项目配置
"""
import os
import sys

print("="*60)
print("LINS-Industrial 配置测试")
print("="*60)

# 1. 检查目录结构
print("\n[1] 检查目录结构...")
required_dirs = ['config', 'data', 'eval_scripts', 'results', 'utils']
for d in required_dirs:
    if os.path.exists(d):
        print(f"  ✅ {d}/")
    else:
        print(f"  ❌ {d}/ 缺失")

# 2. 检查 LINS-main 路径
print("\n[2] 检查 LINS-main 路径...")
lins_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'LINS-main'))
if os.path.exists(lins_path):
    print(f"  ✅ {lins_path}")
else:
    print(f"  ❌ 未找到: {lins_path}")

# 3. 测试导入
print("\n[3] 测试导入 LINS 核心模块...")
sys.path.insert(0, lins_path)
try:
    from model.model_LINS import LINS
    print("  ✅ 导入成功")
except ImportError as e:
    print(f"  ❌ 导入失败: {e}")

# 4. 检查配置文件
print("\n[4] 检查配置文件...")
config_file = os.path.join('config', 'industrial_config.yaml')
if os.path.exists(config_file):
    print(f"  ✅ {config_file}")
else:
    print(f"  ⚠️ {config_file} 未找到 (可选)")

# 5. 检查 API Key
print("\n[5] 检查 API Key...")
api_key = os.environ.get('DEEPSEEK_API_KEY')
if api_key:
    print(f"  ✅ DEEPSEEK_API_KEY 已设置 (长度: {len(api_key)})")
else:
    print("  ⚠️ DEEPSEEK_API_KEY 未设置，请设置环境变量")

print("\n" + "="*60)
print("测试完成！")
