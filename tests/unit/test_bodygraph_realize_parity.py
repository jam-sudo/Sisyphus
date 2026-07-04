"""Parity guard: BodyGraph.sample() and .realize_means() must realize the SAME
Distribution-typed fields.

Both methods walk the graph's Distribution-valued fields and collapse each to a
concrete ``Distribution(value, cv=0.0)`` — ``sample()`` by drawing, ``realize_means()``
by taking the mean. The set of fields each handles is hard-coded in two parallel
isinstance chains. If a new Distribution field (or a whole new edge type) is wired
into only one of them, that field's uncertainty is silently dropped on the other
path with no other test failing.

This test auto-discovers every Distribution field on Node and on each concrete Edge
subclass and asserts BOTH methods realize all of them (output cv == 0). Adding a
Distribution field to only one method makes the other leave it at cv > 0 → fail.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from sisyphus.core import Distribution
from sisyphus.graph.body import BodyGraph
from sisyphus.graph.types import Edge, Node


def _scalar_distribution_fields(cls) -> list[str]:
    """Names of constructor fields on ``cls`` annotated as a scalar Distribution."""
    return [
        f.name
        for f in dataclasses.fields(cls)
        if f.init and str(f.type) == "Distribution"
    ]


def _concrete_edge_subclasses() -> list[type]:
    """Every concrete Edge subclass (recursively), excluding the base Edge."""
    seen: dict[type, None] = {}
    stack = list(Edge.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls not in seen:
            seen[cls] = None
            stack.extend(cls.__subclasses__())
    return list(seen)


def _all_realized(edge) -> bool:
    """True iff every scalar Distribution field on ``edge`` has cv == 0.0."""
    return all(
        getattr(edge, name).cv == 0.0
        for name in _scalar_distribution_fields(type(edge))
    )


def _edge_with_uncertainty(cls):
    """Construct an edge of ``cls`` with every Distribution field at cv > 0."""
    kwargs = {"source": "a", "target": "b"}
    for name in _scalar_distribution_fields(cls):
        kwargs[name] = Distribution(mean=2.0, cv=0.5)
    return cls(**kwargs)


def test_every_edge_type_is_realized_by_both_methods():
    """For each concrete edge type carrying a Distribution, both sample() and
    realize_means() must collapse all its Distribution fields to cv == 0."""
    edge_types = [c for c in _concrete_edge_subclasses() if _scalar_distribution_fields(c)]
    assert edge_types, "expected at least one Distribution-bearing edge subclass"

    rng = np.random.default_rng(0)
    for cls in edge_types:
        g = BodyGraph()
        g.edges = [_edge_with_uncertainty(cls)]

        sampled = g.sample(rng).edges[0]
        realized = g.realize_means().edges[0]

        assert _all_realized(sampled), (
            f"{cls.__name__}: sample() left a Distribution field unrealized "
            f"(new field not wired into sample())"
        )
        assert _all_realized(realized), (
            f"{cls.__name__}: realize_means() left a Distribution field unrealized "
            f"(new field not wired into realize_means())"
        )


def test_node_distribution_fields_realized_by_both_methods():
    """volume + every enzyme/transporter Distribution must be realized by both."""
    node = Node(
        name="n",
        node_type="organ",
        volume=Distribution(mean=1.5, cv=0.5),
        enzymes={"CYP3A4": Distribution(mean=100.0, cv=0.5)},
        transporters={"OATP1B1": Distribution(mean=50.0, cv=0.5)},
    )
    g = BodyGraph()
    g.nodes["n"] = node

    for realized_graph in (g.sample(np.random.default_rng(0)), g.realize_means()):
        n = realized_graph.nodes["n"]
        assert n.volume.cv == 0.0
        assert all(d.cv == 0.0 for d in n.enzymes.values())
        assert all(d.cv == 0.0 for d in n.transporters.values())


def test_global_params_realized_by_both_methods():
    g = BodyGraph()
    g.global_params = {"cardiac_output": Distribution(mean=390.0, cv=0.3)}
    for realized in (g.sample(np.random.default_rng(0)), g.realize_means()):
        assert realized.global_params["cardiac_output"].cv == 0.0


def test_sample_draws_but_realize_uses_mean():
    """Sanity that the two methods differ as intended: realize_means is exact,
    sample is stochastic (so the guard above is testing two real code paths)."""
    g = BodyGraph()
    g.global_params = {"x": Distribution(mean=10.0, cv=0.8)}
    assert g.realize_means().global_params["x"].mean == 10.0
    drawn = g.sample(np.random.default_rng(1)).global_params["x"].mean
    assert drawn != 10.0  # a cv=0.8 draw is ~never exactly the mean
