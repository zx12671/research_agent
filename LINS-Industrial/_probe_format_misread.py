# -*- coding: utf-8 -*-
"""
PROBE: 格式信息误读/稀释量测 (重写 v2)
============================================================
背景(来自 format×capability 交叉召回矩阵 与 y_format 量测):
  KED/混合检索对部分 format y(零散版本串 / RAG 文档文本) 召回失败,
  疑点为"格式身份(标准号/版本号)在原文中不可独立检索 → 语义检索拉不回源 chunk"。

本 probe 用真实 DeepSeek-chat 做【RAG 格式改写】, 对照【纯原文检索】,
量化: 改写后的相关性检索(hit@1/5/10)是否显著高于纯原文检索。

样本: knowledge_corpus/chunks 中 document_id 形如标准号/含编号的 RAG 文档 chunk。
判据: 若"改写 hit 提升 + 原文 miss", 则证明当前失败主因 = 格式身份无法从原文
      语义检索; 检索前反解析/改写是必要修复点。
"""
import os, sys, json, time, re
from openai import OpenAI
import httpx

os.environ.setdefault('DEEPSEEK_API_KEY', '<DEEPSEEK_API_KEY_FROM_ENV>')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from retrieval.retriever import OpenDomainRetriever

OUT = os.path.join(ROOT, 'results', 'probe_format_misread')
os.makedirs(OUT, exist_ok=True)
CHUNKS_JSONL = os.path.join(ROOT, 'knowledge_corpus', 'chunks', 'industrybench_chunks.jsonl')


def load_doc_candidates(limit=40):
    """从 chunks 中挑 document_id 含编号(标准号/版本号)的文档, 各取一个代表 chunk。"""
    doc2rep = {}            # document_id -> {content_head, chunk_id}
    doc_cnt = {}
    if not os.path.exists(CHUNKS_JSONL):
        print('未找到 chunks:', CHUNKS_JSONL)
        return []
    with open(CHUNKS_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            cid = d.get('chunk_id') or d.get('doc_id') or d.get('source')
            did = d.get('document_id') or d.get('doc_id') or d.get('source') or str(cid)
            content = d.get('content', '')
            if not did:
                continue
            doc_cnt[did] = doc_cnt.get(did, 0) + 1
            if did not in doc2rep and content:
                doc2rep[did] = {'chunk_id': cid, 'doc_id': did,
                                'content': content[:180]}
    # 只保留编号型 doc_id (标准号/版本号形态)
    cands = []
    for did, rep in doc2rep.items():
        s = str(did)
        if re.search(r'\d', s):            # 含数字 -> 版本/编号
            cands.append(rep)
        if len(cands) >= limit:
            break
    return cands


# ------------------------- DeepSeek -------------------------
def _client():
    http = httpx.Client(timeout=httpx.Timeout(90.0, connect=10.0))
    return OpenAI(api_key=os.environ['DEEPSEEK_API_KEY'],
                  base_url="https://api.deepseek.com", http_client=http), "deepseek-chat"

def rewrite_query(doc_id, head):
    """由 doc_id + 内容头反解析出唯一召回该源 chunk 的最佳检索查询。"""
    c, m = _client()
    prompt = (f"文档标识(doc_id): {doc_id}\n文档正文开头: {head}\n\n"
              f"请输出唯一能召回该源文档/源片段的最佳检索查询(≤25字, 保留关键编号、"
              f"标准代号与中文主题词, 不要解释)。")
    try:
        r = c.chat.completions.create(
            model=m,
            messages=[{"role": "system",
                       "content": "你是工业标准(RAG)文档检索专家, 把给定文档标题/正文转换为能唯一召回该源文档的检索查询。"},
                      {"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=80)
        return (r.choices[0].message.content or '').strip()[:60]
    except Exception as e:
        return ''
    finally:
        c.close()


def hit_in(res, doc_id):
    """判断检索结果的 chunk/document 中是否命中给定 doc_id。"""
    di = str(doc_id)
    for ch in res.chunks:
        cands = [str(ch.chunk_id), str(ch.document_id), str(ch.source), str(ch.content)[:120]]
        for c in cands:
            if c and (di in c or c in di):
                return True
    return False


def main():
    print('PROBE: 格式改写相对原文检索增益 (真实 DeepSeek + 混合检索)')
    print('=' * 74)
    retr = OpenDomainRetriever()
    try:
        retr.load_from_manifest()
    except Exception as e:
        print('retriever 加载失败:', e)
        return

    cands = load_doc_candidates(limit=24)
    print(f'编号型文档候选: {len(cands)}')
    rows = []
    for rep in cands:
        did = rep['doc_id']
        head = rep['content']
        # 原文检索: 用内容头第一句(无身份强调)
        raw_q = head
        rw_q = rewrite_query(did, head) or did
        if not rw_q:
            rw_q = did
        r_raw = retr.hybrid_retrieve(raw_q, k=10, use_ked=True)
        r_rw = retr.hybrid_retrieve(rw_q, k=10, use_ked=True)
        # 再兜底直接用 doc_id 反解析串检索
        r_docid = retr.hybrid_retrieve(did, k=10, use_ked=True)

        def topk(rr, k):
            ids = [str(c.document_id) for c in rr.chunks[:k] if c.document_id]
            blob = ' '.join([str(c.document_id)+'||'+str(c.chunk_id) for c in rr.chunks[:k]])
            return any(did in b or b in did for b in ids) or (did in blob)

        rows.append({
            'doc_id': did, 'content_head': head,
            'rewrite': rw_q,
            'raw_hit@1': topk(r_raw, 1), 'rw_hit@1': topk(r_rw, 1),
            'docid_hit@10': topk(r_docid, 10),
            'raw_num': len(r_raw.chunks), 'rw_num': len(r_rw.chunks),
        })
        tag = 'RAW-MISS/RW-HIT' if (topk(r_rw, 10) and not topk(r_raw, 10)) else \
              ('BOTH-HIT' if (topk(r_rw, 10) and topk(r_raw, 10)) else 'neither')
        print(f"[{tag:14}] {str(did)[:24]:24} raw@10={'Y' if topk(r_raw,10) else '-':3} "
              f"rw@10={'Y' if topk(r_rw,10) else '-':3} q='{rw_q[:38]}'")
        time.sleep(0.2)

    n = len(rows)
    raw_hit = sum(1 for r in rows if (r['rw_hit@1'] or (r['raw_hit@1'] and False)))  # placeholder
    raw_hit = sum(1 for r in rows if r['raw_hit@1'])
    rw_hit = sum(1 for r in rows if r['rw_hit@1'])
    pure_gain = sum(1 for r in rows if r['rw_hit@1'] and not r['raw_hit@1'])
    summary = {
        'docs_tested': n,
        'raw_hit@1': raw_hit,
        'rewrite_hit@1': rw_hit,
        'pure_gain_rw_over_raw@1': pure_gain,
        'sample': rows,
    }
    with open(os.path.join(OUT, 'format_misread_results.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print('=' * 74)
    print(f"原文字面查询 hit@1: {raw_hit}/{n} | 改写后 hit@1: {rw_hit}/{n} | 纯新增(仅改写命中): {pure_gain}")
    print('详细结果 ->', os.path.join(OUT, 'format_misread_results.json'))


if __name__ == '__main__':
    main()
