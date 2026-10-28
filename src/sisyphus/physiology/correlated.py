"""Jointly sample registered enzyme and transporter abundances."""

from __future__ import annotations

import dataclasses

import numpy as np

from sisyphus.core import Distribution
from sisyphus.graph.body import BodyGraph
from sisyphus.physiology import correlation_registry
from sisyphus.resources import get_resource_config

# ---------------------------------------------------------------------------
# Correlated abundance resampler
# ---------------------------------------------------------------------------


def resample_correlated_abundances(
    graph: BodyGraph, rng: np.random.Generator
) -> BodyGraph:
    """Replace every grouped Distribution in the graph with a sampled draw.

    For each node, Distributions (in enzymes and transporters) that share a
    correlation_group are collected and sampled jointly from their registered
    multivariate-lognormal spec. Sampled values replace the original means;
    the new Distributions have cv=0 and correlation_group=None, representing
    "one realized individual" at this MC iteration.

    Distributions with correlation_group=None are left unchanged.
    """
    g = BodyGraph()
    g.edges = list(graph.edges)
    g.global_params = dict(graph.global_params)

    for name, node in graph.nodes.items():
        # Collect items by group for this node
        groups: dict[str, list[tuple[str, str, Distribution]]] = {}
        for tag, d in node.enzymes.items():
            if d.correlation_group:
                groups.setdefault(d.correlation_group, []).append(("enzyme", tag, d))
        for tag, d in node.transporters.items():
            if d.correlation_group:
                groups.setdefault(d.correlation_group, []).append(("transporter", tag, d))

        new_enzymes = dict(node.enzymes)
        new_transporters = dict(node.transporters)

        for group_name, entries in groups.items():
            spec = correlation_registry.get(group_name)
            if spec is None and group_name == "liver_achour2021":
                correlation_registry.load_from_json(
                    get_resource_config().data("physiology", "achour2021_correlation.json")
                )
                spec = correlation_registry.get(group_name)
            if spec is None:
                raise KeyError(
                    f"Node {name!r} references correlation_group "
                    f"{group_name!r} but no such group is registered. "
                    f"Did you call load_from_json() on the appropriate JSON?"
                )
            # Index by tag so we can reorder to spec.members
            by_tag = {tag: (kind, d) for (kind, tag, d) in entries}
            missing = set(spec.members) - set(by_tag.keys())
            if missing:
                raise KeyError(
                    f"Node {name!r} group {group_name!r} is missing members "
                    f"{sorted(missing)} required by registered spec."
                )
            means = np.array([by_tag[m][1].mean for m in spec.members])
            cvs = np.array([by_tag[m][1].cv for m in spec.members])
            sampled = correlation_registry.sample_correlated(
                means, cvs, spec.log_corr_matrix, rng
            )
            for i, member_tag in enumerate(spec.members):
                kind, _old = by_tag[member_tag]
                new_d = Distribution(
                    mean=float(sampled[i]), cv=0.0, correlation_group=None
                )
                if kind == "enzyme":
                    new_enzymes[member_tag] = new_d
                else:
                    new_transporters[member_tag] = new_d

        g.nodes[name] = dataclasses.replace(
            node, enzymes=new_enzymes, transporters=new_transporters
        )

    return g


