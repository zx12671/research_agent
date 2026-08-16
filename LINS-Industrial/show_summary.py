"""show_summary.py: 显示最新的 IndustryBench 评估结果"""
import json, os

result_dir = os.path.join(os.path.dirname(__file__), 'results')
files = [f for f in os.listdir(result_dir) if f.endswith('.json')]
if not files:
    print("⚠️  未找到评估结果文件")
    exit(0)

latest = sorted(files)[-1]
path = os.path.join(result_dir, latest)

with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f'📄 文件: {latest}')
print()

# 兼容新格式 (summary.total_samples) 和旧格式 (total_samples)
if 'summary' in data:
    # 论文级评估格式
    s = data['summary']
    print('🎯 IndustryBench 论文级评估结果')
    print('=' * 50)
    print(f'  样本总数:    {s["total_samples"]}')
    print(f'  Raw Mean:    {s["raw_mean"]:.4f} / 3.0')
    print(f'  Adj Mean:    {s["adjusted_mean"]:.4f} (SV调整后)')
    print(f'  SV Rate:     {s["sv_stats"]["sv_rate"]:.4f}')
    print(f'  Δ Delta:     {s["sv_stats"]["delta"]:.4f}')
    print(f'  Raw分布:     {s["raw_distribution"]}')
    
    if 'by_difficulty' in s:
        print(f'\n📊 按难度:')
        for n, st in sorted(s['by_difficulty'].items(), key=lambda x: -x[1]['count']):
            print(f'  {n:<10} n={st["count"]:4d} Raw={st["raw_mean"]:.4f} Adj={st["adjusted_mean"]:.4f}')
    
    if 'by_capability' in s:
        print(f'\n📊 按能力:')
        for n, st in sorted(s['by_capability'].items(), key=lambda x: -x[1]['count']):
            print(f'  {n:<15} n={st["count"]:4d} Raw={st["raw_mean"]:.4f} Adj={st["adjusted_mean"]:.4f}')
else:
    # 旧格式（100条测试）
    print(f'总样本: {data.get("total_samples", "?")}')
    print(f'成功: {data.get("success_count", "?")}')
    print(f'模式: {data.get("mode", "?")}')
    print(f'数据库: {data.get("database", "?")}')

print(f'\n📋 配置: {json.dumps(data.get("config", {}), ensure_ascii=False)}')
print(f'⏱️  时序: {json.dumps(data.get("timing", {}))}')
