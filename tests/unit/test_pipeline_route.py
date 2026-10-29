"""Route enum compatibility."""

from sisyphus.pipeline.predict import Route, predict


def test_enum_route_matches_plain_string():
    assert Route.ORAL == "oral"
    assert Route.IV == "iv"
    result = predict("Cn1cnc2n(C)c(=O)n(C)c(=O)c12", 100.0, Route.ORAL)
    assert type(result.route) is str
