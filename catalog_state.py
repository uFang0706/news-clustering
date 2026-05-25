from __future__ import annotations
"""
状态管理器 — 读/写 catalog_state.json

负责：
  - 初始化 state 文件
  - 读写节点（社区）数据
  - 停止条件检查
  - 状态机进度记录
"""

import json
from pathlib import Path
from typing import Any, Optional


STATE_FILENAME = "catalog_state.json"

# 输出目录名，与聚类引擎一致
OUTPUT_DIR = Path("/Users/yuwan/code/news-clustering/output_catalog")


class CatalogState:
    """编目状态封装"""

    def __init__(self, data_dir: str | Path | None = None):
        self.data_dir = Path(data_dir) if data_dir else OUTPUT_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.data_dir / STATE_FILENAME
        self.state: dict[str, Any] = {"round": 0, "max_rounds": 3}
        self._dirty = False

    # ---- 加载/保存 ----

    def load(self) -> dict[str, Any]:
        if self.state_path.exists():
            with open(self.state_path) as f:
                self.state = json.load(f)
        return self.state

    def save(self):
        with open(self.state_path, "w") as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)
        self._dirty = False

    def _mark_dirty(self):
        self._dirty = True

    # ---- 初始 ----

    def init_new(
        self,
        total_headlines: int,
        max_rounds: int = 3,
        round: int = 0,
    ) -> dict[str, Any]:
        self.state = {
            "round": round,
            "max_rounds": max_rounds,
            "total_headlines": total_headlines,
            "nodes": [],
            "stopping": {
                "max_topic_ok": None,
                "min_size_ok": None,
                "cd_ratio_ok": None,
                "meta_clean": None,
                "all_named": None,
            },
        }
        self._mark_dirty()
        self.save()
        return self.state

    # ---- 节点操作 ----

    def add_node(self, node: dict[str, Any]) -> str:
        """添加节点，返回 node_id"""
        node.setdefault("status", "pending")
        node.setdefault("children", [])
        self.state["nodes"].append(node)
        self._mark_dirty()
        return node["id"]

    def get_node(self, node_id: str) -> Optional[dict[str, Any]]:
        for n in self.state["nodes"]:
            if n["id"] == node_id:
                return n
        return None

    def update_node(self, node_id: str, updates: dict[str, Any]):
        for n in self.state["nodes"]:
            if n["id"] == node_id:
                n.update(updates)
                self._mark_dirty()
                return

    def get_pending_nodes(self) -> list[dict[str, Any]]:
        """当前轮次中未处理的节点"""
        return [n for n in self.state["nodes"]
                if n["status"] == "pending"]

    def get_nodes_for_round(self, round: int) -> list[dict[str, Any]]:
        return [n for n in self.state["nodes"]
                if n["id"].startswith(f"v{round}_")]

    def get_leaf_nodes(self) -> list[dict[str, Any]]:
        return [n for n in self.state["nodes"]
                if len(n.get("children", [])) == 0
                and n["status"] != "binned"]

    def get_accepted(self) -> list[dict[str, Any]]:
        return [n for n in self.state["nodes"]
                if n["status"] == "accepted"]

    def get_binned(self) -> list[dict[str, Any]]:
        return [n for n in self.state["nodes"]
                if n["status"] == "binned"]

    # ---- 轮次 ----

    def current_round(self) -> int:
        return self.state.get("round", 0)

    def increment_round(self) -> int:
        self.state["round"] += 1
        self._mark_dirty()
        return self.state["round"]

    def is_iterating(self) -> bool:
        return self.state["round"] <= self.state["max_rounds"]

    # ---- 停止条件 ----

    def check_stopping(self, max_pct: float = 0.15, max_cd_pct: float = 0.10) -> bool:
        """
        检查 5 项停止条件。
        更新 stopping 字段。
        全部通过返回 True。
        """
        total = self.state["total_headlines"]
        s = self.state["stopping"]

        leaves = self.get_leaf_nodes()
        accepted = self.get_accepted()
        binned = self.get_binned()

        # 1. 所有话题有名字（accepted 节点必须都有 final_grade）
        named = [n for n in accepted if n.get("final_grade") in ("A", "B")]
        s["all_named"] = len(named) == len(accepted) if accepted else False

        # 2. 最大话题 < 总量 × max_pct
        max_size = max((n["size"] for n in accepted if n["status"] == "accepted"), default=0)
        s["max_topic_ok"] = max_size < total * max_pct

        # 3. 最小话题 >= 15（浅层 v0/v1 节点，包含人工确认的合法小话题）
        shallow = [n for n in accepted if n["depth"] < 2]
        shallow_min = min((n["size"] for n in shallow), default=0) if shallow else 0
        deep_under = [n for n in accepted if n["depth"] >= 2 and n["size"] < 10]
        s["min_size_ok"] = shallow_min >= 15
        s["min_size_shallow_min"] = shallow_min
        s["deep_under_10"] = len(deep_under)

        # 4. C+D 标题合计 < 总量 × max_cd_pct
        binned_size = sum(n["size"] for n in binned)
        s["cd_ratio_ok"] = binned_size < total * max_cd_pct

        # 5. 无未处理的 meta 候选
        pending_meta = [n for n in self.state["nodes"]
                        if n["status"] == "pending"
                        and n.get("rule_grade") == "D-meta"]
        s["meta_clean"] = len(pending_meta) == 0

        # 6. 无 pending 节点（split 后的子节点已评估完毕）
        pending_count = len([n for n in self.state["nodes"] if n["status"] == "pending"])
        s["no_pending"] = pending_count == 0

        all_pass = all(
            s.get(k) for k in
            ["all_named", "max_topic_ok", "min_size_ok", "cd_ratio_ok", "meta_clean", "no_pending"]
        )

        self._mark_dirty()
        return all_pass
