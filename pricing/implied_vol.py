import numpy as np

from pricing.bsm import option_price, vega


def implied_volatility(
    market_price,
    S,
    K,
    T,
    r,
    q,
    option_type,
    initial_guess=0.20,
    tolerance=1e-8,
    max_iterations=100
):
    """
    Calculate implied volatility using Newton-Raphson,
    with a bisection fallback.

    Returns volatility as a decimal.
    Example: 0.25 = 25% implied volatility.
    """

    option_type = option_type.lower()

    # --------------------------------------------------------
    # No-arbitrage price bounds
    # --------------------------------------------------------

    if option_type == "call":

        lower_bound = max(
            0,
            S * np.exp(-q * T) - K * np.exp(-r * T)
        )

        upper_bound = S * np.exp(-q * T)

    elif option_type == "put":

        lower_bound = max(
            0,
            K * np.exp(-r * T) - S * np.exp(-q * T)
        )

        upper_bound = K * np.exp(-r * T)

    else:
        raise ValueError("option_type must be 'call' or 'put'")

    if not lower_bound <= market_price <= upper_bound:
        raise ValueError(
            f"Market price {market_price} violates "
            f"no-arbitrage bounds [{lower_bound}, {upper_bound}]"
        )

    if np.isclose(market_price, lower_bound, atol=1e-12):
        return 1e-8

    # --------------------------------------------------------
    # Newton-Raphson
    # --------------------------------------------------------

    sigma = initial_guess

    for _ in range(max_iterations):

        price = option_price(
            S, K, T, r, q, sigma, option_type
        )

        price_error = price - market_price

        if abs(price_error) < tolerance:
            return sigma

        current_vega = vega(
            S, K, T, r, q, sigma
        )

        # Newton-Raphson becomes unstable when vega is tiny
        if current_vega < 1e-10:
            break

        sigma -= price_error / current_vega

        # Volatility cannot be negative
        if sigma <= 0 or sigma > 10:
            break

    # --------------------------------------------------------
    # Bisection fallback
    # --------------------------------------------------------

    low = 1e-8
    high = 5.0

    low_price = option_price(
        S, K, T, r, q, low, option_type
    )

    high_price = option_price(
        S, K, T, r, q, high, option_type
    )

    if not (low_price <= market_price <= high_price):
        raise ValueError(
            "Could not bracket implied volatility."
        )

    for _ in range(max_iterations * 2):

        sigma = (low + high) / 2

        price = option_price(
            S, K, T, r, q, sigma, option_type
        )

        if abs(price - market_price) < tolerance:
            return sigma

        if price > market_price:
            high = sigma
        else:
            low = sigma

    raise ValueError(
        "Implied volatility solver failed to converge."
    )