#!/usr/bin/env python3
"""Feature Graph Enforcement Utilities.

Builds a deterministic DAG representation of feature definitions and provides:
- Graph hashing (stable across ordering) used in reproducibility & registry.
- Validation of active feature store definitions against expected hash stored with model.
- Dependency expansion to ensure transitive dependencies are included.

Design Notes:
The feature graph hash intentionally excludes non-functional metadata (created_at, tags, author)
to avoid noisy hash invalidations. Only fields affecting semantics:
  name, version, dependencies (recursively), transformation_logic (if present).

If a dependency is missing or its definition changed, validation fails and inference will be blocked
unless override=True is explicitly passed (for controlled emergency scenarios).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Any, Tuple
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GraphNode:
    name: str
    version: str
    dependencies: Tuple[str, ...]
    logic: str | None

    def canonical(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "dependencies": sorted(self.dependencies),
            "logic": self.logic or None,
        }


class FeatureGraphBuilder:
    def __init__(self, feature_definitions: Dict[str, Any]):
        """feature_definitions: mapping of feature_id -> FeatureDefinition"""
        self.feature_definitions = feature_definitions

    def build_subgraph(self, root_feature_names: List[str]) -> Dict[str, GraphNode]:
        visited: Dict[str, GraphNode] = {}

        def visit(name: str):
            # locate definition by name (since keys are feature_id name:version)
            fdef = None
            for fd in self.feature_definitions.values():
                if fd.name == name:
                    fdef = fd
                    break
            if not fdef:
                raise ValueError(f"Feature definition not found: {name}")
            node_key = f"{fdef.name}:{fdef.version}"
            if node_key in visited:
                return
            # visit dependencies first
            for dep in fdef.dependencies:
                # dependencies may reference raw source columns like market_data.close -> skip those without definitions
                if dep.startswith("market_data.") or "." in dep and dep.split(".")[0] == "market_data":
                    continue
                try:
                    visit(dep)
                except ValueError as e:
                    raise ValueError(f"Missing dependency '{dep}' for feature '{name}': {e}") from e
            visited[node_key] = GraphNode(
                name=fdef.name,
                version=fdef.version,
                dependencies=tuple(sorted([d for d in fdef.dependencies if not d.startswith("market_data.")])),
                logic=fdef.transformation_logic,
            )

        for rf in root_feature_names:
            visit(rf)
        return visited

    def hash_subgraph(self, nodes: Dict[str, GraphNode]) -> str:
        canonical_list = [n.canonical() for n in nodes.values()]
        stable_json = json.dumps(sorted(canonical_list, key=lambda x: x["name"]), sort_keys=True)
        return hashlib.sha256(stable_json.encode()).hexdigest()[:16]

    def compute_hash_for_features(self, feature_names: List[str]) -> str:
        sub = self.build_subgraph(feature_names)
        return self.hash_subgraph(sub)


async def validate_feature_graph(feature_store, expected_hash: str, feature_names: List[str]) -> bool:
    """Validate the current feature definitions graph matches expected hash.

    Returns True on match else False. Logs discrepancies.
    """
    builder = FeatureGraphBuilder(feature_store.feature_definitions)
    try:
        current_hash = builder.compute_hash_for_features(feature_names)
    except ValueError as e:
        logger.error("Feature graph validation failed: %s", e)
        return False
    if current_hash != expected_hash:
        logger.error("Feature graph hash mismatch expected=%s current=%s", expected_hash, current_hash)
        return False
    return True

__all__ = [
    "FeatureGraphBuilder",
    "validate_feature_graph",
    "GraphNode",
]
