"""调用 MiniMax VL API 对项目可视化图做视觉识别"""
import base64, json, os, time
from pathlib import Path
import requests

ENV = Path.home()/'.hermes'/'.env'
if ENV.exists():
    for line in ENV.read_text().splitlines():
        if '=' in line and not line.strip().startswith('#'):
            k,v=line.split('=',1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

API_KEY = os.environ.get('MINIMAX_CN_API_KEY') or os.environ.get('MINIMAX_API_KEY')

BASE_URL = 'https://api.minimaxi.com/v1/chat/completions'
IMG_DIRS = [
    Path('/Users/yuwan/code/news-clustering/output_exp_i'),
    Path('/Users/yuwan/code/news-clustering/output_exp_i2'),
]
images = [
    ('Leiden社区二维分布图', IMG_DIRS[0]/'community_map.png'),
    ('Leiden社区规模图', IMG_DIRS[0]/'community_sizes.png'),
    ('Leiden resolution扫描图', IMG_DIRS[0]/'resolution_scan.png'),
    ('Bayesian GMM二维聚类图', IMG_DIRS[1]/'cluster_map.png'),
    ('Bayesian GMM聚类规模图', IMG_DIRS[1]/'cluster_sizes.png'),
    ('Bayesian GMM置信度分布图', IMG_DIRS[1]/'confidence_dist.png'),
    ('Bayesian GMM prior扫描图', IMG_DIRS[1]/'prior_scan.png'),
]

out = {}
for title, path in images:
    if not path.exists():
        out[title] = {'path': str(path), 'error': 'file not found'}
        continue
    b64 = base64.b64encode(path.read_bytes()).decode()
    prompt = '''你是一个数据可视化审阅员。请仔细观察这张图。
输出中文，要求：
1. 识别图中主要图形类型（散点图/柱状图/折线图等）、坐标轴含义、图例、标题；
2. 读出图中最明显的趋势、分布或异常点；
3. 说明它对"新闻标题无监督聚类的项目"有什么信息量；
4. 如果字太小看不清，直接说看不清，不要猜。'''
    payload = {
        'model': 'MiniMax-VL',
        'messages': [{
            'role': 'user',
            'content': [
                {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{b64}'}},
                {'type': 'text', 'text': prompt}
            ]
        }],
        'max_tokens': 1200,
        'temperature': 0.0,
    }
    try:
        r = requests.post(BASE_URL, headers={'Authorization': f'Bearer {API_KEY}'}, json=payload, timeout=120)
        j = r.json()
        if r.status_code >= 400:
            out[title] = {'path': str(path), 'status': r.status_code, 'response': str(j)[:1200]}
        else:
            text = j.get('choices', [{}])[0].get('message', {}).get('content', '')
            out[title] = {'path': str(path), 'analysis': text.strip(), 'model': 'MiniMax-VL'}
    except Exception as e:
        out[title] = {'path': str(path), 'error': repr(e)}
    print(f"[{'OK' if 'analysis' in out[title] else 'FAIL'}] {title}")
    time.sleep(0.5)

Path('/Users/yuwan/code/news-clustering/output_minimax_vl_analysis.json').write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
print('\nDone. Results saved.')
