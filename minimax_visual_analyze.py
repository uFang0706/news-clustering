import base64, json, os, sys, time
from pathlib import Path
import requests

ENV = Path.home()/'.hermes'/'.env'
if ENV.exists():
    for line in ENV.read_text().splitlines():
        if '=' in line and not line.strip().startswith('#'):
            k,v=line.split('=',1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

API_KEY = os.environ.get('MINIMAX_CN_API_KEY') or os.environ.get('MINIMAX_API_KEY')
if not API_KEY:
    raise SystemExit('MiniMax API key not found')

BASE_URL = 'https://api.minimaxi.com/anthropic/v1/messages'
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
    data = base64.b64encode(path.read_bytes()).decode()
    prompt = f'''请作为数据可视化审阅员，只根据这张图本身进行视觉识别和解读，不要补充图中看不到的信息。
图名：{title}
输出中文，要求：
1. 识别图中主要图形类型、坐标轴/图例/标题（如可见）；
2. 读出图中最明显的趋势、分布或异常；
3. 说明它对“新闻标题无监督分话题/聚类项目”的含义；
4. 如果图中字太小或看不清，明确说看不清，不要猜。'''
    payload = {
        'model': 'MiniMax-M2.7',
        'messages': [{
            'role': 'user',
            'content': [
                {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': data}},
                {'type': 'text', 'text': prompt}
            ]
        }],
        'max_tokens': 1200,
        'temperature': 0.0,
    }
    try:
        r = requests.post(BASE_URL, headers={'Authorization': f'Bearer {API_KEY}', 'Content-Type':'application/json'}, json=payload, timeout=90)
        if r.status_code >= 400:
            out[title] = {'path': str(path), 'status': r.status_code, 'text': r.text[:1000]}
        else:
            j = r.json()
            text = ''
            for c in j.get('content', []):
                if c.get('type') == 'text':
                    text += c.get('text','')
            out[title] = {'path': str(path), 'analysis': text.strip(), 'model': 'MiniMax-M2.7'}
    except Exception as e:
        out[title] = {'path': str(path), 'error': repr(e)}
    time.sleep(0.5)

Path('/Users/yuwan/code/news-clustering/output_minimax_visual_analysis.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
print('wrote /Users/yuwan/code/news-clustering/output_minimax_visual_analysis.json')
for k,v in out.items():
    print('\n##', k)
    print(v.get('analysis') or v.get('error') or v.get('text'))
