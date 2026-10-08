from nexora.research import heuristic_queries, parse_queries
from nexora.research.markets import format_markets, parse_manifold, parse_polymarket

MANIFOLD_ITEMS = [
    {
        "id": "udAUupudlE",
        "closeTime": 1793211900000,
        "question": "Will the Fed cut rates by at least 25 bps at the Oct. 28, 2026 FOMC meeting?",
        "url": "https://manifold.markets/CalibratedGhosts/will-the-fed-cut-rates",
        "probability": 0.31,
        "outcomeType": "BINARY",
        "volume": 1045.15,
        "isResolved": False,
        "uniqueBettorCount": 8,
    },
    {"question": "Which month?", "outcomeType": "MULTIPLE_CHOICE", "isResolved": False},
    {"question": "Old", "outcomeType": "BINARY", "isResolved": True, "probability": 1.0},
]

POLYMARKET_PAYLOAD = {
    "events": [
        {
            "slug": "fed-decision-in-october",
            "title": "Fed decision in October?",
            "markets": [
                {
                    "question": "Fed decreases rates by 25 bps after October 2026 meeting?",
                    "outcomes": '["Yes", "No"]',
                    "outcomePrices": '["0.27", "0.73"]',
                    "volume": "1250000.5",
                    "endDate": "2026-10-28T00:00:00Z",
                    "closed": False,
                    "active": True,
                },
                {
                    "question": "Closed market",
                    "outcomes": '["Yes", "No"]',
                    "outcomePrices": '["0", "1"]',
                    "closed": True,
                    "active": True,
                },
            ],
        }
    ]
}


def test_parse_manifold_filters():
    markets = parse_manifold(MANIFOLD_ITEMS)
    assert len(markets) == 1
    assert markets[0].probability == 0.31
    assert markets[0].close == "2026-10-28"


def test_parse_polymarket_json_strings():
    markets = parse_polymarket(POLYMARKET_PAYLOAD)
    assert len(markets) == 1
    assert markets[0].probability == 0.27
    assert markets[0].volume == 1250000.5
    assert markets[0].url.endswith("fed-decision-in-october")


def test_format_markets_orders_by_volume():
    text = format_markets(parse_manifold(MANIFOLD_ITEMS) + parse_polymarket(POLYMARKET_PAYLOAD))
    lines = text.splitlines()
    assert lines[0].startswith("- [Polymarket")
    assert "27%" in lines[0]


def test_query_parsing_with_fallbacks():
    q = "Will the Fed cut rates at the 2026-10-28 FOMC meeting?"
    news, markets = parse_queries('{"news_query": "Fed October rate decision", "market_queries": ["Fed cut October"]}', q)
    assert news == "Fed October rate decision" and markets == ["Fed cut October"]
    news2, markets2 = parse_queries("not json", q)
    assert news2 == q and "2026-10-28" not in markets2[0]
    assert heuristic_queries(q)[1][0].startswith("Fed cut rates")
