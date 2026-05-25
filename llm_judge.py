from __future__ import annotations
"""
LLM 裁判 — 对规则不确定的社区做语义命名 + 评级

v2: 压缩 prompt (3+2 样本), 自动重试 (thinking 时升 max_tokens)

用法:
    judge = LLMJudge(cache_path="output_catalog/llm_cache.json")
    result = judge.judge(node_id="v0_c0", size=6823, keywords="...",
                         top_samples=[...], boundary_samples=[...])
    # result = {"name_zh": "...", "grade": "A|B|C|D", "action": "...", ...}
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import requests


# ==================== API 配置 ====================

API_BASE_URL = "https://api.minimaxi.com/anthropic/v1/messages"
API_MODEL = "MiniMax-M2.7"

def _load_api_key() -> Optional[str]:
    api_key = os.environ.get("MINIMAX_CN_API_KEY") or os.environ.get("MINIMAX_API_KEY")
    if api_key:
        return api_key
    env_path = Path.home() / ".hermes" / ".env"
    if env_path.exists():
        for line in env_path.read_text().split("\n"):
            if line.startswith("MINIMAX_CN_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


# ==================== Prompt 模板 (v2: 压缩) ====================

PROMPT_TEMPLATE = """
社区有 {size} 条新闻标题
关键词: {keywords}
代表: {samples_top} | 边界: {samples_boundary}

输出JSON: {{"name_zh":"话题名(4-8字)","grade":"A","action":"keep"}}

grade 选项:
  A = 高度一致（所有标题围绕一个明确主题）
  B = 大致一致（主题基本明确，有少量杂音）
  C = 太杂（主题混杂，需要拆分）
  D = 栏目/节目（不是新闻话题，是广播/电视栏目）

action 选项:
  keep = 保留现状
  rename = 需要改名
  split = 需要拆分（仅 grade=C 时使用）
  reject = 拒绝（仅 grade=D 时使用）
""".strip()


# ==================== LLM 裁判类 ====================

class LLMJudge:
    """每社区一次 API 调用，返回编目结果（支持 retry）"""

    RETRY_TOKENS = 8192  # 重试时使用的 max_tokens

    def __init__(self, cache_path: str | Path = "output_catalog/llm_cache.json"):
        self.cache_path = Path(cache_path)
        self.cache: dict[str, dict] = self._load_cache()
        self.api_key = _load_api_key()
        self._has_key = self.api_key is not None

    # ---- 缓存 ----

    def _load_cache(self) -> dict[str, dict]:
        if self.cache_path.exists():
            with open(self.cache_path) as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(self.cache, f, indent=2, ensure_ascii=False)

    # ---- 核心调用 ----

    def judge(
        self,
        node_id: str,
        size: int,
        keywords: str,
        top_samples: list[str],
        boundary_samples: list[str],
        force: bool = False,
    ) -> dict:
        """
        对一个社区执行 LLM 编目评估。

        Args:
            node_id: 节点 ID (如 "v0_c0")
            size: 社区标题数
            keywords: TF-IDF 关键词
            top_samples: centroid 最近的 10 条标题
            boundary_samples: centroid 最远的 5 条标题
            force: 强制重新调用（忽略缓存）

        Returns:
            {"name_zh": "...", "grade": "A|B|C|D", "action": "...", "noise_count": N, "reason": "..."}
            或 {"error": "..."} 异常情况
        """
        # 检查缓存
        if not force and node_id in self.cache:
            cached = self.cache[node_id]
            # 跳过之前失败的缓存
            if "error" not in cached or force:
                return cached

        if not self._has_key:
            return self._fallback(keywords)

        # 构造 prompt — 压缩版 (3 代表 + 2 边界)
        samples_top_str = " | ".join(s[:80] for s in top_samples[:3])
        samples_boundary_str = " | ".join(s[:80] for s in boundary_samples[:2])

        prompt = PROMPT_TEMPLATE.format(
            size=size,
            keywords=keywords,
            samples_top=samples_top_str,
            samples_boundary=samples_boundary_str,
        )

        return self._call_api(node_id, prompt)

    def _call_api(self, node_id: str, prompt: str, retry: bool = True) -> dict:
        """执行 API 调用，支持 thinking-only 自动重试"""
        max_tokens = 2048
        attempt = 0

        while attempt < 2:
            attempt += 1
            try:
                resp = requests.post(
                    API_BASE_URL,
                    json={
                        "model": API_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": 0.0,
                    },
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=120,
                )
                data = resp.json()

                # 检查 API 错误
                base_resp = data.get("base_resp", {})
                if base_resp.get("status_code", 0) != 0:
                    raise ValueError(
                        f"API error: {base_resp.get('status_code')} "
                        f"{base_resp.get('status_msg', '')}"
                    )

                content_blocks = data.get("content", [])
                block_types = [b.get("type") for b in content_blocks]

                # 找 text 块
                raw_text = ""
                for block in content_blocks:
                    if block.get("type") == "text" and "text" in block:
                        raw_text = block["text"].strip()
                        break

                if not raw_text:
                    # thinking-only: 重试
                    if retry and "thinking" in block_types and max_tokens < self.RETRY_TOKENS:
                        print(f"  {node_id}: thinking-only (types={block_types}), retry with max_tokens={self.RETRY_TOKENS}")
                        max_tokens = self.RETRY_TOKENS
                        continue
                    raise KeyError(
                        f"no text block. types: {block_types}"
                    )

                # 解析 JSON
                result = self._parse_json(raw_text)

                # 缓存
                self.cache[node_id] = result
                self._save_cache()

                time.sleep(0.3)
                return result

            except Exception as e:
                error_msg = f"API call failed: {e}"
                # 如果是 thinking-only 且还有重试机会
                if "no text block" in str(e) and retry and attempt < 2:
                    print(f"  {node_id}: {error_msg[:60]}, retrying with more tokens...")
                    max_tokens = self.RETRY_TOKENS
                    continue
                error = {"error": error_msg[:120], "name_zh": "", "grade": None}
                self.cache[node_id] = error
                self._save_cache()
                return error

    # ---- 备用：没有 API key 时的退化方案 ----

    def _fallback(self, keywords: str) -> dict:
        top_kw = keywords.split(",")[0].strip() if keywords else "unknown"
        return {
            "name_zh": top_kw[:20],
            "grade": "B",
            "action": "keep",
            "noise_count": 0,
            "reason": "[fallback] no LLM API key",
        }

    # ---- JSON 解析 ----

    @staticmethod
    def _parse_json(text: str) -> dict:
        """从 LLM 回复中提取 JSON（处理 markdown 代码块包裹）"""
        # 去掉 markdown 代码块
        text = re.sub(r"^```(?:json)?\s*", "", text.strip())
        text = re.sub(r"\s*```$", "", text.strip())

        # 尝试解析
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            # 尝试找 JSON 部分
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group())
                except json.JSONDecodeError:
                    result = {"name_zh": "", "grade": None, "error": "JSON parse failed"}
            else:
                result = {"name_zh": "", "grade": None, "error": "No JSON found"}

        # 标准化字段 + 校验
        result.setdefault("name_zh", "")
        result.setdefault("grade", None)
        result.setdefault("action", "keep")
        result.setdefault("noise_count", 0)
        result.setdefault("reason", "")

        # 校验：grade 必须是单个字母，防 LLM 输出 "A/B/C/D" 等垃圾
        raw_grade = result.get("grade", None)
        if isinstance(raw_grade, str) and len(raw_grade) == 1 and raw_grade in "ABCD":
            result["grade"] = raw_grade
        else:
            result["grade"] = "B"  # fallback: 接受
            result["_grade_raw"] = raw_grade

        return result

    # ---- 批量重跑失败的节点 ----

    def clear_failed_cache(self, node_ids: list[str] | None = None):
        """清除失败缓存的条目"""
        to_remove = []
        for nid, cached in self.cache.items():
            if "error" in cached:
                if node_ids is None or nid in node_ids:
                    to_remove.append(nid)
        for nid in to_remove:
            del self.cache[nid]
        self._save_cache()
        print(f"  [缓存] 清除 {len(to_remove)} 条失败记录")
