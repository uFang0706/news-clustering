from __future__ import annotations
"""
规则打分器 — 确定性、零成本的社区初筛

在调 LLM 之前，先用规则给每个社区打一个初步等级。
规则能确定的（A、D-meta、D-fragment）直接归档，
只有 B/C 候选才交给 LLM 裁判。

规则列表：
  1. 栏目黑名单   → D-meta
  2. 关键词域熵   → C 候选（跨 3+ 主题域）
  3. 规模过小     → D-fragment（< 30 条）
  4. 规模过大     → C 候选（> 7,500 条或总量 15%）
  5. centroid 方差 → C 候选（社区内平均距离 > 全量中位数 × 1.5）
"""

import re
from typing import Any


# ==================== 黑名单 ====================

# 栏目/节目/内部标签 — 不是新闻话题
BLACKLIST_PATTERNS = [
    r"grandstand",
    r"the\s*drum",
    r"abc\s*weather",
    r"extended\s*interview",
    r"country\s*hour",
    r"rural\s*news",
    r"national\s*rural",
    r"abc\s*news",
    r"news\s*exchange",
    r"capital\s*hill",
    r"breakfast\s*show",
    r"morning\s*program",
    r"afternoon\s*show",
    r"pm\s*news",
    r"world\s*today",
    r"correspondents\s*report",
    r"abc\s*sport",
    r"abc\s*entertainment",
]


def match_blacklist(keywords_text: str) -> bool:
    """检查关键词中是否匹配黑名单"""
    text = keywords_text.lower()
    for pat in BLACKLIST_PATTERNS:
        if re.search(pat, text):
            return True
    return False


# ==================== 关键词域熵 ====================

# 每个域的关键词种子
DOMAIN_KEYWORDS = {
    "sport": ["sport", "afl", "nrl", "goal", "match", "coach", "cup", "win",
              "tigers", "blues", "raiders", "game", "final", "league",
              "player", "team", "stadium"],
    "politics": ["election", "vote", "parliament", "polls", "senate",
                 "labor", "liberal", "government", "pm", "minister",
                 "party", "political", "candidate"],
    "crime": ["crime", "police", "charged", "court", "murder", "arrest",
              "jail", "prison", "sentence", "convicted", "stabbing",
              "assault", "drug"],
    "disaster": ["fire", "bushfire", "flood", "earthquake", "hurricane",
                 "cyclone", "blaze", "firefighter", "evacuate", "damage",
                 "emergency", "rescue"],
    "health": ["health", "hospital", "patient", "doctor", "medical",
               "disease", "covid", "vaccine", "mental", "aged care",
               "medicare"],
    "economy": ["market", "budget", "economy", "price", "cost", "funding",
                "tax", "debt", "trade", "business", "shares", "stock",
                "profit", "mortgage"],
    "international": ["war", "troops", "gaza", "israel", "iraq", "iran",
                      "china", "russia", "ukraine", "refugee", "diplomat",
                      "sanction", "treaty"],
    "weather": ["weather", "rain", "storm", "cyclone", "temperature",
                "forecast", "humidity", "drought", "heatwave"],
    "education": ["school", "university", "student", "teacher", "education",
                  "college", "curriculum", "exam", "tuition"],
    "environment": ["climate", "environment", "emission", "renewable",
                    "solar", "wind", "carbon", "coal", "mining",
                    "conservation", "species"],
    "transport": ["road", "rail", "train", "bus", "highway", "traffic",
                  "tunnel", "bridge", "airport", "flight", "transport"],
    "entertainment": ["film", "movie", "music", "art", "festival", "concert",
                      "celebrity", "award", "television"],
}


def count_domains(keywords_text: str) -> int:
    """统计关键词覆盖了几个主题域"""
    text = keywords_text.lower()
    matched_domains = set()
    for domain, seeds in DOMAIN_KEYWORDS.items():
        for seed in seeds:
            if seed in text:
                matched_domains.add(domain)
                break
    return len(matched_domains)


# ==================== 规则评分器 ====================


class RuleScorer:
    """
    对单个社区应用全部 5 条规则。

    返回:
        grade: "A" | "D-meta" | "D-fragment" | "C" | None (不确定 → 走 LLM)
        signals: dict[str, str] 每个触发的规则
        score: 1-5 (触发的规则数，越高越确定)
    """

    def __init__(self, total_headlines: int, max_topic_pct: float = 0.15,
                 median_all_dist: float | None = None):
        self.total = total_headlines
        self.max_topic_pct = max_topic_pct
        self.min_size = 30
        self.median_all_dist = median_all_dist

    def score(self, community: dict[str, Any]) -> tuple[str | None, dict[str, str], int]:
        """
        community: {
            "id": str,
            "size": int,
            "keywords": str (TF-IDF 关键词逗号分隔),
            "mean_pairwise_dist": float (社区内平均 pairwise 距离)
        }
        """
        signals: dict[str, str] = {}
        triggered = 0
        forced = None  # 直接确定的等级

        cid = community.get("id", "?")
        size = community.get("size", 0)
        keywords = community.get("keywords", "")
        mpd = community.get("mean_pairwise_dist", None)

        # --- 规则 1: 规模过小 ---
        if size < self.min_size:
            signals[f"R1_{cid}"] = f"size={size} < {self.min_size}"
            triggered += 1
            forced = "D-fragment"

        # --- 规则 2: 规模过大 ---
        max_size = int(self.total * self.max_topic_pct)
        if size > max_size:
            signals[f"R2_{cid}"] = f"size={size} > {max_size} ({self.max_topic_pct*100:.0f}%)"
            triggered += 1
            forced = "C"

        # --- 规则 3: 黑名单 ---
        if match_blacklist(keywords):
            signals[f"R3_{cid}"] = f"blacklist match in keywords"
            triggered += 1
            forced = "D-meta"

        # --- 规则 4: 关键词域熵 ---
        nd = count_domains(keywords)
        if nd >= 4:
            signals[f"R4_{cid}"] = f"domains={nd}, suggests mixed bag"
            triggered += 1
            if forced is None:
                forced = "C"
        elif nd == 0 and len(keywords) > 0:
            # 没有匹配到任何域 → 可能是不常见的主题或碎片
            # 不直接定级，留给 LLM
            signals[f"R4_{cid}"] = f"domains=0, unusual keywords"
            triggered += 1

        # --- 规则 5: centroid 方差 ---
        if mpd is not None and self.median_all_dist is not None:
            if mpd > self.median_all_dist * 1.5:
                # 社区内平均距离远大于全量中位数 → 社区太散
                signals[f"R5_{cid}"] = f"mpd={mpd:.4f} > {self.median_all_dist*1.5:.4f}"
                triggered += 1
                if forced is None:
                    forced = "C"

        # --- 决策 ---
        if forced is not None:
            return forced, signals, triggered

        # 规则 1-4 都没触发明确的 C/D → 留给 LLM
        if triggered == 0:
            return None, signals, 0

        # 触发了低权重规则（如域熵=0）→ 不确定，交给 LLM
        return None, signals, triggered

    def batch_score(
        self,
        communities: list[dict[str, Any]],
        median_all_dist: float | None = None,
    ) -> list[tuple[str, int, dict[str, str]]]:
        """
        批量评分 + 规则 5 支持。

        返回: [(community_id, grade, signals), ...]
        grade = "A" | "D-meta" | "D-fragment" | "C" | None
        """
        results = []
        for comm in communities:
            # 先用规则 1-4
            grade, signals, _ = self.score(comm)

            # 规则 5: centroid 方差
            mpd = comm.get("mean_pairwise_dist", None)
            if mpd is not None and median_all_dist is not None:
                if mpd > median_all_dist * 1.5:
                    if grade is None or grade == "A":
                        cid = comm.get("id", "?")
                        signals[f"R5_{cid}"] = f"mpd={mpd:.4f} > {median_all_dist*1.5:.4f}"
                        grade = "C"
                    elif grade in ("D-meta", "D-fragment"):
                        # D 级优先于 C
                        pass

            results.append((comm.get("id"), grade, signals))

        return results
