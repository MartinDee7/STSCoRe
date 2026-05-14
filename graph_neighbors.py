# -*- coding: utf-8 -*-
"""可选路网邻接：从边表 CSV 加载，供 prompt 中「邻居传感器」一行使用。"""
from __future__ import annotations

import os
from typing import Dict, List, Optional


def load_neighbor_map_from_csv(path: Optional[str]) -> Dict[int, List[int]]:
    """
    边表：每行两个整数 src dst（无向）。可含表头，非数字行跳过。
    返回 node_id -> 邻居 id 列表（升序、去重）。
    """
    if not path or not str(path).strip():
        return {}
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        return {}
    adj: Dict[int, set] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().startswith("src") or line.lower().startswith("from"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                a, b = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
    return {k: sorted(v) for k, v in adj.items()}


def neighbor_prompt_line(node_id: int, neighbor_map: Dict[int, List[int]], max_neighbors: int = 12) -> str:
    if not neighbor_map:
        return ""
    neigh = neighbor_map.get(int(node_id), [])
    if not neigh:
        return ""
    shown = neigh[: max(1, int(max_neighbors))]
    return f"Graph neighbors (sensor column ids): {', '.join(str(x) for x in shown)}.\n"
