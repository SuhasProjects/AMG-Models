from pricing.bsm import calculate_bsm, calculate_garman_kohlhagen
from monte_carlo.monte_carlo import load_price_data, calculate_returns, fit_garch, forecast_garch_volatility, simulate_paths, price_option_monte_carlo
from datetime import datetime
from pricing.implied_vol import implied_volatility


def calculate_time_to_maturity(current_date, expiration_date):
    current_date = datetime.strptime(current_date, "%Y-%m-%d")
    expiration_date = datetime.strptime(expiration_date, "%Y-%m-%d")

    if expiration_date <= current_date:
        raise ValueError("Expiration date must be after current date.")

    days = (expiration_date - current_date).days

    return days / 365.0

def validate_leg(
    asset_class,
    S,
    K,
    T,
    r,
    q,
    sigma,
    option_type,
    position,
    quantity=1,
    rf=None
):
    if S <= 0:
        raise ValueError("Underlying price must be greater than 0.")

    if K <= 0:
        raise ValueError("Strike price must be greater than 0.")

    if T <= 0:
        raise ValueError("Time to maturity must be greater than 0.")

    if sigma <= 0:
        raise ValueError("Volatility must be greater than 0.")

    if quantity <= 0:
        raise ValueError("Quantity must be greater than 0.")

    if option_type.lower() not in ["call", "put"]:
        raise ValueError("option_type must be 'call' or 'put'.")

    if position.lower() not in ["long", "short"]:
        raise ValueError("position must be 'long' or 'short'.")

    if asset_class.lower() == "fx" and rf is None:
        raise ValueError("Foreign risk-free rate is required for FX.")

    return True

def create_leg(
    asset_class,
    S,
    K,
    current_date,
    expiration_date,
    r,
    q,
    sigma,
    option_type,
    position,
    quantity=1,
    rf=None,
    actual_price=None,
):

    T = calculate_time_to_maturity(
            current_date,
            expiration_date
        )
    validate_leg(
    asset_class=asset_class,
    S=S,
    K=K,
    T=T,
    r=r,
    q=q,
    sigma=sigma,
    option_type=option_type,
    position=position,
    quantity=quantity,
    rf=rf
)
    """
    Create an option leg with its pricing inputs.

    rf is only required for FX options.
    """

    if position.lower() not in ["long", "short"]:
        raise ValueError("position must be 'long' or 'short'")

    if option_type.lower() not in ["call", "put"]:
        raise ValueError("option_type must be 'call' or 'put'")

    if quantity <= 0:
        raise ValueError("quantity must be greater than 0")

    return {
        "asset_class": asset_class,
        "S": S,
        "K": K,
        "T": T,
        "r": r,
        "q": q,
        "sigma": sigma,
        "option_type": option_type.lower(),
        "position": position.lower(),
        "quantity": quantity,
        "rf": rf,
        "actual_price": actual_price,
    }




def aggregate_legs(legs):
    totals = {
        "price": 0.0,
        "mc_price": 0.0,
        "delta": 0.0,
        "gamma": 0.0,
        "vega": 0.0,
        "theta": 0.0,
        "rho": 0.0, 
        "phi": 0.0
    }

    for leg in legs:
        if leg["position"].lower() == "long":
            sign = 1
        elif leg["position"].lower() == "short":
            sign = -1
        else:
            raise ValueError("position must be 'long' or 'short'")

        quantity = leg.get("quantity", 1)

        for metric in totals:
            totals[metric] += sign * quantity * leg.get(metric, 0.0)

    return totals

def price_leg(leg):
    """
    Calculate the analytical price and Greeks for one option leg.
    """
    asset = str(leg.get("asset_class", "")).lower()

    if asset == "fx":
        if leg["rf"] is None:
            raise ValueError("Foreign risk-free rate is required for FX.")
        
        result = calculate_garman_kohlhagen(
            S=leg["S"],
            K=leg["K"],
            T=leg["T"],
            rd=leg["r"],
            rf=leg["rf"],
            sigma=leg["sigma"],
            option_type=leg["option_type"]
        )

    elif asset == "futures":
        # Black 76: Set dividend yield (q) equal to risk-free rate (r)
        result = calculate_bsm(
            S=leg["S"],
            K=leg["K"],
            T=leg["T"],
            r=leg["r"],
            q=leg["r"],  
            sigma=leg["sigma"],
            option_type=leg["option_type"]
        )
        result["model"] = "Black 76 (Futures)"

    else:
        # Standard Spot Equity / ETF
        result = calculate_bsm(
            S=leg["S"],
            K=leg["K"],
            T=leg["T"],
            r=leg["r"],
            q=leg["q"],
            sigma=leg["sigma"],
            option_type=leg["option_type"]
        )

    leg.update(result)
    return leg

def price_strategy(
    historical_data,
    legs,
    n_paths=10000,
    trading_days=252,
    seed=None
):
    # Process historical data once
    cleaned_data = load_price_data(historical_data)
    returns = calculate_returns(cleaned_data)

    # Fit GARCH once for the underlying
    garch_result = fit_garch(returns)

    priced_legs = []

    for leg in legs:
        # Forecast volatility over this option's remaining maturity
        horizon = max(1, int(round(leg["T"] * trading_days)))

        garch_volatility = forecast_garch_volatility(
            garch_result,
            horizon=horizon,
            trading_days=trading_days
        )

        # Analytical price + Greeks
        priced_leg = price_leg(leg)

        # FX uses the foreign risk-free rate as q
        # Ensure Monte Carlo uses the correct continuous yield for the asset class
        asset = str(leg["asset_class"]).lower()
        if asset == "fx":
            mc_q = leg["rf"]
        elif asset == "futures":
            mc_q = leg["r"]
        else:
            mc_q = leg["q"]

        # Monte Carlo
        final_prices = simulate_paths(
            S=leg["S"],
            T=leg["T"],
            r=leg["r"],
            q=mc_q,
            garch_result=garch_result,
            n_paths=n_paths,
            trading_days=trading_days,
            seed=seed
        )

        mc_price = price_option_monte_carlo(
            final_prices=final_prices,
            K=leg["K"],
            T=leg["T"],
            r=leg["r"],
            option_type=leg["option_type"]
        )

        priced_leg["mc_price"] = mc_price
        priced_leg["garch_volatility"] = garch_volatility
        priced_leg["garch_horizon"] = horizon

        priced_legs.append(priced_leg)

    # Aggregate analytical prices + Greeks
    totals = aggregate_legs(priced_legs)

    return {
        "legs": priced_legs,
        "totals": totals
    }

def get_implied_volatility(
    market_price,
    S,
    K,
    T,
    r,
    q,
    option_type
):
    return implied_volatility(
        market_price=market_price,
        S=S,
        K=K,
        T=T,
        r=r,
        q=q,
        option_type=option_type
    )

def set_implied_volatility(
    leg,
    market_price
):
    leg["market_price"] = market_price

    leg["sigma"] = get_implied_volatility(
        market_price=market_price,
        S=leg["S"],
        K=leg["K"],
        T=leg["T"],
        r=leg["r"],
        q=leg["q"],
        option_type=leg["option_type"]
    )

    return leg