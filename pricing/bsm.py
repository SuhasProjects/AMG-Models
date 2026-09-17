import numpy as np
from scipy.stats import norm
from statsmodels.tsa.stattools import adfuller

# ============================================================
# Black-Scholes-Merton Core
# ============================================================

def d1(S, K, T, r, q, sigma):
    """
    Calculate d1 used in the Black-Scholes-Merton model.

    Parameters
    ----------
    S : float
        Current underlying price
    K : float
        Strike price
    T : float
        Time to expiration in years
    r : float
        Continuously compounded risk-free rate
    q : float
        Continuously compounded dividend yield
    sigma : float
        Volatility
    """

    return (
        np.log(S / K)
        + (r - q + 0.5 * sigma**2) * T
    ) / (sigma * np.sqrt(T))


def d2(S, K, T, r, q, sigma):
    """
    Calculate d2 used in the Black-Scholes-Merton model.
    """

    return d1(S, K, T, r, q, sigma) - sigma * np.sqrt(T)


# ============================================================
# Option Prices
# ============================================================

def call_price(S, K, T, r, q, sigma):
    """
    Black-Scholes-Merton price of a European call.
    """

    D1 = d1(S, K, T, r, q, sigma)
    D2 = d2(S, K, T, r, q, sigma)

    return (
        S * np.exp(-q * T) * norm.cdf(D1)
        - K * np.exp(-r * T) * norm.cdf(D2)
    )


def put_price(S, K, T, r, q, sigma):
    """
    Black-Scholes-Merton price of a European put.
    """

    D1 = d1(S, K, T, r, q, sigma)
    D2 = d2(S, K, T, r, q, sigma)

    return (
        K * np.exp(-r * T) * norm.cdf(-D2)
        - S * np.exp(-q * T) * norm.cdf(-D1)
    )


def option_price(S, K, T, r, q, sigma, option_type):
    """
    General BSM pricing function.

    option_type:
        "call" or "put"
    """

    option_type = option_type.lower()

    if option_type == "call":
        return call_price(S, K, T, r, q, sigma)

    elif option_type == "put":
        return put_price(S, K, T, r, q, sigma)

    else:
        raise ValueError("option_type must be 'call' or 'put'")


# ============================================================
# Greeks
# ============================================================

def delta(S, K, T, r, q, sigma, option_type):
    """
    Delta: sensitivity of option price to a $1 change in S.
    """

    D1 = d1(S, K, T, r, q, sigma)

    option_type = option_type.lower()

    if option_type == "call":
        return np.exp(-q * T) * norm.cdf(D1)

    elif option_type == "put":
        return np.exp(-q * T) * (norm.cdf(D1) - 1)

    else:
        raise ValueError("option_type must be 'call' or 'put'")


def gamma(S, K, T, r, q, sigma):
    """
    Gamma: sensitivity of delta to a $1 change in S.

    Gamma is identical for calls and puts.
    """

    D1 = d1(S, K, T, r, q, sigma)

    return (
        np.exp(-q * T)
        * norm.pdf(D1)
        / (S * sigma * np.sqrt(T))
    )


def vega(S, K, T, r, q, sigma):
    """
    Vega: sensitivity of option price to a 1.00 change
    in volatility.

    Example:
        vega = 0.25

    means a 1.00 increase in sigma changes price by
    approximately $0.25.

    For a 1 percentage-point volatility change,
    divide by 100.
    """

    D1 = d1(S, K, T, r, q, sigma)

    return (
        S
        * np.exp(-q * T)
        * norm.pdf(D1)
        * np.sqrt(T)
    )


def theta(S, K, T, r, q, sigma, option_type):
    """
    Theta: change in option price per year as time passes.

    This returns the standard BSM theta convention:
    the amount the option loses/gains per year from
    the passage of time.
    """

    D1 = d1(S, K, T, r, q, sigma)
    D2 = d2(S, K, T, r, q, sigma)

    option_type = option_type.lower()

    first_term = (
        -(
            S
            * np.exp(-q * T)
            * norm.pdf(D1)
            * sigma
        )
        / (2 * np.sqrt(T))
    )

    if option_type == "call":

        return (
            first_term
            - r * K * np.exp(-r * T) * norm.cdf(D2)
            + q * S * np.exp(-q * T) * norm.cdf(D1)
        )

    elif option_type == "put":

        return (
            first_term
            + r * K * np.exp(-r * T) * norm.cdf(-D2)
            - q * S * np.exp(-q * T) * norm.cdf(-D1)
        )

    else:
        raise ValueError("option_type must be 'call' or 'put'")


def rho(S, K, T, r, q, sigma, option_type):
    """
    Rho: sensitivity of option price to a 1.00 change
    in the risk-free rate.
    """

    D2 = d2(S, K, T, r, q, sigma)

    option_type = option_type.lower()

    if option_type == "call":
        return (
            K
            * T
            * np.exp(-r * T)
            * norm.cdf(D2)
        )

    elif option_type == "put":
        return (
            -K
            * T
            * np.exp(-r * T)
            * norm.cdf(-D2)
        )

    else:
        raise ValueError("option_type must be 'call' or 'put'")


# ============================================================
# Complete BSM Result
# ============================================================

def calculate_bsm(S, K, T, r, q, sigma, option_type):
    """
    Calculate the complete BSM result for one European option.

    Returns
    -------
    dict
        Price and all five Greeks.
    """

    return {
        "model": "Black-Scholes-Merton",
        "option_type": option_type.lower(),

        "price": option_price(
            S, K, T, r, q, sigma, option_type
        ),

        "delta": delta(
            S, K, T, r, q, sigma, option_type
        ),

        "gamma": gamma(
            S, K, T, r, q, sigma
        ),

        "vega": vega(
            S, K, T, r, q, sigma
        ),

        "theta": theta(
            S, K, T, r, q, sigma, option_type
        ),

        "rho": rho(
            S, K, T, r, q, sigma, option_type
        )
    }


#FX Specific BSM

def garman_kohlhagen_d1(S, K, T, rd, rf, sigma):
    return (
        np.log(S / K)
        + (rd - rf + 0.5 * sigma**2) * T
    ) / (sigma * np.sqrt(T))


def garman_kohlhagen_d2(S, K, T, rd, rf, sigma):
    return (
        garman_kohlhagen_d1(S, K, T, rd, rf, sigma)
        - sigma * np.sqrt(T)
    )


def garman_kohlhagen_price(S, K, T, rd, rf, sigma, option_type):
    d1 = garman_kohlhagen_d1(S, K, T, rd, rf, sigma)
    d2 = garman_kohlhagen_d2(S, K, T, rd, rf, sigma)

    option_type = option_type.lower()

    if option_type == "call":
        return (
            S * np.exp(-rf * T) * norm.cdf(d1)
            - K * np.exp(-rd * T) * norm.cdf(d2)
        )

    elif option_type == "put":
        return (
            K * np.exp(-rd * T) * norm.cdf(-d2)
            - S * np.exp(-rf * T) * norm.cdf(-d1)
        )

    else:
        raise ValueError("option_type must be 'call' or 'put'")


def garman_kohlhagen_delta(S, K, T, rd, rf, sigma, option_type):
    d1 = garman_kohlhagen_d1(S, K, T, rd, rf, sigma)

    if option_type.lower() == "call":
        return np.exp(-rf * T) * norm.cdf(d1)
    elif option_type.lower() == "put":
        return np.exp(-rf * T) * (norm.cdf(d1) - 1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def garman_kohlhagen_gamma(S, K, T, rd, rf, sigma):
    d1 = garman_kohlhagen_d1(S, K, T, rd, rf, sigma)

    return (
        np.exp(-rf * T)
        * norm.pdf(d1)
        / (S * sigma * np.sqrt(T))
    )


def garman_kohlhagen_vega(S, K, T, rd, rf, sigma):
    d1 = garman_kohlhagen_d1(S, K, T, rd, rf, sigma)

    return (
        S
        * np.exp(-rf * T)
        * norm.pdf(d1)
        * np.sqrt(T)
    )


def garman_kohlhagen_theta(S, K, T, rd, rf, sigma, option_type):
    d1 = garman_kohlhagen_d1(S, K, T, rd, rf, sigma)
    d2 = garman_kohlhagen_d2(S, K, T, rd, rf, sigma)

    first_term = (
        -S
        * np.exp(-rf * T)
        * norm.pdf(d1)
        * sigma
        / (2 * np.sqrt(T))
    )

    option_type = option_type.lower()

    if option_type == "call":
        return (
            first_term
            + rf * S * np.exp(-rf * T) * norm.cdf(d1)
            - rd * K * np.exp(-rd * T) * norm.cdf(d2)
        )

    elif option_type == "put":
        return (
            first_term
            - rf * S * np.exp(-rf * T) * norm.cdf(-d1)
            + rd * K * np.exp(-rd * T) * norm.cdf(-d2)
        )

    else:
        raise ValueError("option_type must be 'call' or 'put'")


def garman_kohlhagen_rho(
    S, K, T, rd, rf, sigma, option_type
):
    d2 = garman_kohlhagen_d2(S, K, T, rd, rf, sigma)

    if option_type.lower() == "call":
        return K * T * np.exp(-rd * T) * norm.cdf(d2)
    elif option_type.lower() == "put":
        return -K * T * np.exp(-rd * T) * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

def garman_kohlhagen_phi(S, K, T, rd, rf, sigma, option_type):
    d1 = garman_kohlhagen_d1(S, K, T, rd, rf, sigma)

    if option_type.lower() == "call":
        return -S * T * np.exp(-rf * T) * norm.cdf(d1)

    elif option_type.lower() == "put":
        return S * T * np.exp(-rf * T) * norm.cdf(-d1)

    else:
        raise ValueError("option_type must be 'call' or 'put'")

def calculate_garman_kohlhagen(
    S,
    K,
    T,
    rd,
    rf,
    sigma,
    option_type
):
    return {
        "model": "Garman-Kohlhagen",
        "option_type": option_type.lower(),
        "price": garman_kohlhagen_price(
            S, K, T, rd, rf, sigma, option_type
        ),
        "delta": garman_kohlhagen_delta(
            S, K, T, rd, rf, sigma, option_type
        ),
        "gamma": garman_kohlhagen_gamma(
            S, K, T, rd, rf, sigma
        ),
        "vega": garman_kohlhagen_vega(
            S, K, T, rd, rf, sigma
        ),
        "theta": garman_kohlhagen_theta(
            S, K, T, rd, rf, sigma, option_type
        ),
        "rho": garman_kohlhagen_rho(
            S, K, T, rd, rf, sigma, option_type
        ),
        "phi": garman_kohlhagen_phi(
                    S, K, T, rd, rf, sigma, option_type
                )
    }