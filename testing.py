import pandas as pd
from strategy.strategy import create_leg, price_strategy, calculate_time_to_maturity, get_implied_volatility,set_implied_volatility
from pricing.implied_vol import implied_volatility

data = pd.DataFrame({
    "Date": [
        "10/16/2025",
        "10/15/2025",
        "10/14/2025",
        "10/13/2025",
        "10/10/2025",
        "10/9/2025",
        "10/8/2025",
        "10/7/2025"
    ],
    "Last Price": [
        61.31,
        63.17,
        63.91,
        65.81,
        66.2,
        68.74,
        70.06,
        69.13
    ]
})


leg = create_leg(
    asset_class="Metals",
    S=61.31,
    K=60,
    current_date="09/16/2026",
    expiration_date="12/16/2026",
    r=0.04,
    q=0,
    sigma=0.25,
    option_type="call",
    position="long"
)



leg = create_leg(
    asset_class="Metals",
    S=100,
    K=100,
    current_date="09/16/2026",
    expiration_date="09/16/2027",
    r=0.05,
    q=0,
    sigma=0.25,
    option_type="call",
    position="long"
)

set_implied_volatility(
    leg,
    market_price=10.4506
)

print(leg["sigma"])