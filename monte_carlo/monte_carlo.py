import pandas as pd
import numpy as np

from arch import arch_model


# ============================================================
# Historical Price Data
# ============================================================

def load_price_data(data):
    """
    Clean historical price data pasted from Excel.

    Expected columns:
        Date
        Last Price

    Parameters
    ----------
    data : pandas.DataFrame
        Historical price data.

    Returns
    -------
    pandas.DataFrame
        Cleaned data sorted chronologically.
    """

    df = data.copy()

    # Standardize column names
    df.columns = [str(col).strip() for col in df.columns]

    # Make sure required columns exist
    if "Date" not in df.columns or "Last Price" not in df.columns:
        raise ValueError(
            "Data must contain 'Date' and 'Last Price' columns."
        )

    # Convert dates and prices
    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df["Last Price"] = pd.to_numeric(
        df["Last Price"],
        errors="coerce"
    )

    # Remove rows with missing dates or prices
    df = df.dropna(
        subset=["Date", "Last Price"]
    )

    # Remove invalid prices
    df = df[df["Last Price"] > 0]

    # Remove duplicate dates
    df = df.drop_duplicates(
        subset="Date",
        keep="last"
    )

    # Sort oldest -> newest
    df = df.sort_values("Date")

    # Reset index
    df = df.reset_index(drop=True)

    if len(df) < 2:
        raise ValueError(
            "At least two valid price observations are required."
        )

    return df


# ============================================================
# Historical Returns
# ============================================================

def calculate_returns(df):
    """
    Calculate daily logarithmic returns.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned price data containing 'Last Price'.

    Returns
    -------
    pandas.Series
        Daily log returns.
    """

    prices = df["Last Price"]

    returns = np.log(
        prices / prices.shift(1)
    )

    # Remove first NaN
    returns = returns.dropna()

    return returns


# ============================================================
# GARCH(1,1) Volatility
# ============================================================

def fit_garch(returns):
    """
    Fit a GARCH(1,1) model to historical daily log returns.

    The ARCH package is given returns in percentage units.

    Parameters
    ----------
    returns : pandas.Series
        Daily logarithmic returns in decimal form.

    Returns
    -------
    ARCHModelResult
        Fitted GARCH model.
    """

    if len(returns) < 30:
        raise ValueError(
            "At least 30 return observations are recommended for GARCH estimation."
        )

    # ARCH works in percentage-return units.
    returns_percent = returns * 100

    model = arch_model(
        returns_percent,
        mean="Constant",
        vol="GARCH",
        p=1,
        q=1,
        dist="normal"
    )

    result = model.fit(
        disp="off"
    )

    return result


# ============================================================
# GARCH Volatility Forecast
# ============================================================

def forecast_garch_volatility(
    model_result,
    horizon=1,
    trading_days=252
):
    """
    Forecast GARCH volatility over a specified horizon.

    Parameters
    ----------
    model_result : ARCHModelResult
        Fitted GARCH model.

    horizon : int
        Number of trading days to forecast.

    trading_days : int
        Number of trading days used for annualization.

    Returns
    -------
    float
        Annualized forecast volatility as a decimal.

    Notes
    -----
    For horizon=1, this represents the next-day conditional
    volatility annualized.

    For a larger horizon, the returned value is based on the
    average forecast variance over the requested horizon.
    """

    if horizon < 1:
        raise ValueError(
            "Forecast horizon must be at least 1 trading day."
        )

    if trading_days < 1:
        raise ValueError(
            "trading_days must be at least 1."
        )

    forecast = model_result.forecast(
        horizon=horizon
    )

    # Variance is returned in percentage-squared units.
    forecast_variance = forecast.variance.iloc[-1].to_numpy()

    # Convert percentage-squared variance to decimal daily variance.
    forecast_variance = forecast_variance / 10000.0

    # Average variance across the requested horizon.
    average_daily_variance = np.mean(
        forecast_variance
    )

    # Annualize the average variance.
    annualized_volatility = np.sqrt(
        average_daily_variance * trading_days
    )

    return float(annualized_volatility)


# ============================================================
# Dynamic GARCH Monte Carlo Simulation
# ============================================================

def simulate_paths(
    S,
    T,
    r,
    q,
    garch_result,
    n_paths=10000,
    trading_days=252,
    seed=None
):
    """
    Simulate risk-neutral underlying price paths using a
    dynamic GARCH(1,1) conditional variance process.

    The GARCH model is estimated from historical daily returns.
    Therefore, the variance and volatility generated by the
    GARCH recursion are DAILY quantities.

    Each simulation step represents one trading day.

    Antithetic variates are used to reduce Monte Carlo error.

    Parameters
    ----------
    S : float
        Current underlying price.

    T : float
        Time to expiration in years.

    r : float
        Continuously compounded annualized risk-free rate.

    q : float
        Continuously compounded annualized dividend yield.

        For FX, q should be the foreign risk-free rate.

    garch_result : ARCHModelResult
        Fitted GARCH model.

    n_paths : int
        Number of independent antithetic path pairs.

    trading_days : int
        Number of trading days per year.

    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    numpy.ndarray
        Simulated underlying prices at expiration.

    Notes
    -----
    Because GARCH variance is already daily variance, the
    stochastic terms are:

        -0.5 * variance
        volatility * Z

    rather than:

        -0.5 * variance * dt
        volatility * sqrt(dt) * Z

    The annualized risk-free/dividend drift DOES use dt:

        (r - q) * dt
    """

    # --------------------------------------------------------
    # Validate inputs
    # --------------------------------------------------------

    if S <= 0:
        raise ValueError(
            "Underlying price S must be positive."
        )

    if T <= 0:
        raise ValueError(
            "Time to maturity T must be positive."
        )

    if n_paths < 1:
        raise ValueError(
            "n_paths must be at least 1."
        )

    if trading_days < 1:
        raise ValueError(
            "trading_days must be at least 1."
        )

    # --------------------------------------------------------
    # Extract GARCH parameters
    # --------------------------------------------------------

    omega_percent = float(
        garch_result.params["omega"]
    )

    alpha = float(
        garch_result.params["alpha[1]"]
    )

    beta = float(
        garch_result.params["beta[1]"]
    )

    # The ARCH package works with percentage returns.
    #
    # omega therefore has percentage-squared units.
    #
    # Convert:
    #
    #     percentage^2 -> decimal^2
    #
    omega = omega_percent / 10000.0

    # alpha and beta are dimensionless and therefore
    # require no scaling.

    # --------------------------------------------------------
    # Initial conditional variance
    # --------------------------------------------------------

    # conditional_volatility is in percentage units.
    #
    # Convert:
    #
    #     percentage volatility -> decimal volatility
    #
    initial_volatility = (
        float(
            garch_result.conditional_volatility.iloc[-1]
        ) / 100.0
    )

    initial_variance = (
        initial_volatility ** 2
    )

    if initial_variance <= 0:
        raise ValueError(
            "GARCH produced a non-positive initial variance."
        )

    # --------------------------------------------------------
    # Simulation setup
    # --------------------------------------------------------

    # Number of trading days until expiration.
    n_steps = max(
        1,
        int(np.ceil(T * trading_days))
    )

    # Each simulation step represents one trading day.
    dt = T / n_steps

    rng = np.random.default_rng(
        seed
    )

    # Generate all shocks once instead of allocating inside the loop.
    z_half_matrix = rng.standard_normal((n_steps, n_paths))

    # --- ADD THESE MISSING LINES ---
    total_paths = 2 * n_paths

    prices = np.full(
        total_paths,
        float(S),
        dtype=float
    )

    variances = np.full(
        total_paths,
        initial_variance,
        dtype=float
    )

    for step in range(n_steps):
        z_half = z_half_matrix[step]

        z = np.concatenate(
            [
                z_half,
                -z_half
            ]
        )

        # Current DAILY conditional volatility.
        volatility = np.sqrt(
            np.maximum(
                variances,
                1e-16
            )
        )

        # ----------------------------------------------------
        # Risk-neutral price evolution
        # ----------------------------------------------------
        #
        # Since variances and volatility are DAILY quantities:
        #
        #     log(S[t+1]/S[t])
        #
        #       = (r-q)dt
        #         - 0.5*sigma_t^2
        #         + sigma_t*Z
        #
        # Only the annualized rate term receives dt.
        # ----------------------------------------------------

        prices *= np.exp(
            (r - q) * dt
            - 0.5 * variances
            + volatility * z
        )

        # ----------------------------------------------------
        # GARCH variance update
        # ----------------------------------------------------
        #
        # epsilon_t = sigma_t * Z_t
        #
        # sigma_(t+1)^2 =
        #     omega
        #     + alpha * epsilon_t^2
        #     + beta * sigma_t^2
        #
        # All of these quantities are daily.
        # ----------------------------------------------------

        epsilon = (
            volatility * z
        )

        variances = (
            omega
            + alpha * epsilon ** 2
            + beta * variances
        )

        # Numerical safeguard.
        variances = np.maximum(
            variances,
            1e-16
        )

    return prices


# ============================================================
# Monte Carlo Option Pricing
# ============================================================

def price_option_monte_carlo(
    final_prices,
    K,
    T,
    r,
    option_type
):
    """
    Calculate the Monte Carlo price of a European option.

    Parameters
    ----------
    final_prices : numpy.ndarray
        Simulated underlying prices at expiration.

    K : float
        Strike price.

    T : float
        Time to expiration in years.

    r : float
        Continuously compounded annualized risk-free rate.

    option_type : str
        "call" or "put".

    Returns
    -------
    float
        Monte Carlo option price.
    """

    option_type = option_type.lower()

    if K <= 0:
        raise ValueError(
            "Strike K must be positive."
        )

    if T <= 0:
        raise ValueError(
            "Time to maturity T must be positive."
        )

    # --------------------------------------------------------
    # Calculate expiration payoff
    # --------------------------------------------------------

    if option_type == "call":

        payoffs = np.maximum(
            final_prices - K,
            0
        )

    elif option_type == "put":

        payoffs = np.maximum(
            K - final_prices,
            0
        )

    else:
        raise ValueError(
            "option_type must be 'call' or 'put'."
        )

    # --------------------------------------------------------
    # Discount expected payoff
    # --------------------------------------------------------

    option_price = (
        np.exp(-r * T)
        * np.mean(payoffs)
    )

    return float(option_price)

