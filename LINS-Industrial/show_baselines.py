"""展示 IndustryBench 评估基线结果"""
import json, os

results_dir = './results'

files = sorted([f for f in os.listdir(results_dir) if f.endswith('.json') and 'eval' in f],
               key=lambda x: os.path.getmtime(os.path.join(results_dir, x)))

count = 0
for f in files:
    fp = os.path.join(results_dir, f)
    with open(fp, 'r', encoding='utf-8') as fh:
        data = json.load(fh)
    
    if 'summary' not in data:
        # 旧版格式，跳过
        print(f"⚠️ 跳过旧格式: {f}")
        continue
    
    count += 1
    s = data['summary']
    c = data['config']
    mode = c['mode']
    n = s['total_samples']
    raw_mu = s['raw_mean']
    adj_mu = s['adjusted_mean']
    sv_rate = s['sv_stats']['sv_rate']
    raw_dist = s.get('raw_distribution', {})
    adj_dist = s.get('adjusted_distribution', {})
    
    # 将数值键转为 float 排序
    def to_num(k):
        try: return float(k)
        except: return k
    
    rd = {k: v for k, v in raw_dist.items()}
    ad = {k: v for k, v in adj_dist.items()}
    
    print(f"📊 {mode:15s} | n={n:4d} | Raw μ={raw_mu:.4f} | Adj μ={adj_mu:.4f} | SV={sv_rate:.4f}")
    print(f"    Raw分布: {rd}")
    print(f"    Adj分布: {ad}")
    
    # 显示文件名和时间戳
    ts = f.split('_')[-1].replace('.json','')
    print(f"    📁 {f} | {ts}")
    print()

print(f"共 {count} 个评估结果文件")
