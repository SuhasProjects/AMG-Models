from pathlib import Path
import math

import openpyxl
from openpyxl import load_workbook


MAX_LEGS = 4
PAYOFF_START_ROW = 2
PAYOFF_END_ROW = 51
PAYOFF_POINTS = PAYOFF_END_ROW - PAYOFF_START_ROW + 1


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _smart_round(val):
    """
    Rounds a number to either 2 decimal places or 3 significant figures,
    whichever is more precise. Safely ignores strings (like 'Unlimited').
    """
    if isinstance(val, str):
        return val
    if val is None:
        return None
    try:
        f_val = float(val)
    except (TypeError, ValueError):
        return val
        
    if f_val == 0.0:
        return 0.0
        
    # Find the magnitude (power of 10) of the number
    digits = int(math.floor(math.log10(abs(f_val))))
    
    # 3 sig figs means rounding to (2 - digits) decimal places
    sig_places = 2 - digits
    
    # Use max() so it rounds to at least 2 decimal places, but goes deeper if needed for 3 sig figs
    places = max(2, sig_places)
    
    return round(f_val, places)


def _position_sign(position):
    return 1 if str(position).lower() == "long" else -1


def _leg_full_name(leg):
    position = str(leg.get("position", "")).strip().title()
    option_type = str(leg.get("option_type", "")).strip().title()
    return f"{position} {option_type}".strip()


def _leg_short_name(leg):
    strike = _safe_float(leg.get("K"))
    position = "L" if str(leg.get("position", "")).lower() == "long" else "S"
    option_type = "C" if str(leg.get("option_type", "")).lower() == "call" else "P"
    return f"{strike:g} {position}{option_type}"


def _model_name(leg):
    asset = str(leg.get("asset_class", "")).lower()
    
    if asset == "fx":
        return "Garman-Kohlhagen"
    elif asset == "futures":
        return "Black 76"
    else:
        return "Black-Scholes-Merton"


def _option_payoff(spot, strike, option_type):
    if str(option_type).lower() == "call":
        return max(spot - strike, 0.0)
    return max(strike - spot, 0.0)


def _leg_pnl(spot, leg):
    payoff = _option_payoff(spot, _safe_float(leg.get("K")), leg.get("option_type"))
    
    # If actual price was left at 0, fall back to theoretical price so the chart isn't skewed
    premium = _safe_float(leg.get("actual_price"))
    if premium <= 1e-8:
        premium = _safe_float(leg.get("price"))
        
    quantity = _safe_float(leg.get("quantity"), 1.0)
    if str(leg.get("position", "")).lower() == "long":
        return quantity * (payoff - premium)
    return quantity * (premium - payoff)


def _nice_step(raw_step):
    """Round a raw chart step to a readable 1/2/5 x 10^n increment."""
    if raw_step <= 0:
        return 1.0

    exponent = math.floor(math.log10(raw_step))
    fraction = raw_step / (10 ** exponent)

    if fraction <= 1:
        nice = 1
    elif fraction <= 2:
        nice = 2
    elif fraction <= 5:
        nice = 5
    else:
        nice = 10

    return nice * (10 ** exponent)


def calculate_chart_range(spot_price, legs, points=PAYOFF_POINTS):
    """
    Determine a sensible payoff-chart starting price and step size.

    The chart always contains the current spot and all strikes, with a
    modest buffer around the relevant range. Python only supplies these
    values; the Excel chart itself is left untouched.
    """
    spot = _safe_float(spot_price)

    strikes = [
        _safe_float(leg.get("K"))
        for leg in legs
        if _safe_float(leg.get("K")) > 0
    ]

    relevant = [spot] + strikes
    low = min(relevant)
    high = max(relevant)

    # Give the chart room around the relevant strikes/spot.
    span = max(high - low, spot * 0.10, 1.0)
    chart_low = max(0.0, low - 0.20 * span)
    chart_high = high + 0.20 * span

    # 50 points over the range, rounded to a readable increment.
    step = _nice_step((chart_high - chart_low) / max(points - 1, 1))

    # Snap the beginning to the step size.
    chart_start = math.floor(chart_low / step) * step

    # Keep the step fixed and generate exactly PAYOFF_POINTS rows.
    return chart_start, step


def generate_payoff_data(spot_price, legs):
    """
    Returns:
        chart_start, chart_step, rows

    Each row contains the underlying price, individual-leg P&Ls, and
    total strategy P&L. The Excel template is responsible for charting.
    """
    chart_start, chart_step = calculate_chart_range(spot_price, legs)

    rows = []

    for i in range(PAYOFF_POINTS):
        underlying = chart_start + i * chart_step
        leg_values = [_leg_pnl(underlying, leg) for leg in legs]
        total = sum(leg_values)

        rows.append({
            "underlying": underlying,
            "legs": leg_values,
            "total": total,
        })

    return chart_start, chart_step, rows


def calculate_trade_details(legs):
    """
    Estimate max profit/loss and risk-reward for the strategy.

    The calculation checks S=0, all strikes, and a sufficiently high
    underlying price. Unlimited outcomes are reported explicitly.
    """
    strikes = sorted({
        _safe_float(leg.get("K"))
        for leg in legs
        if _safe_float(leg.get("K")) > 0
    })

    test_spots = [0.0] + strikes

    if strikes:
        max_strike = max(strikes)
        test_spots.extend([
            max_strike * 2.0,
            max_strike * 5.0,
            max_strike * 10.0,
        ])

    pnl_values = [sum(_leg_pnl(s, leg) for leg in legs) for s in test_spots]

    max_profit = max(pnl_values) if pnl_values else 0.0
    max_loss = min(pnl_values) if pnl_values else 0.0

    # Determine asymptotic slope as S becomes very large.
    high_spot = max(
        [1.0] + [10.0 * _safe_float(leg.get("K")) for leg in legs]
    )
    slope_test = high_spot * 2.0
    pnl_high_1 = sum(_leg_pnl(high_spot, leg) for leg in legs)
    pnl_high_2 = sum(_leg_pnl(slope_test, leg) for leg in legs)

    slope = pnl_high_2 - pnl_high_1

    if slope > 1e-8:
        max_profit_display = "Unlimited"
    else:
        max_profit_display = max_profit

    if slope < -1e-8:
        max_loss_display = "Unlimited"
    else:
        max_loss_display = max_loss

    if (
        isinstance(max_profit_display, (int, float))
        and isinstance(max_loss_display, (int, float))
        and max_loss_display < 0
    ):
        risk_reward = max_profit_display / abs(max_loss_display)
    else:
        risk_reward = "N/A"

    return {
        "max_profit": max_profit_display,
        "max_loss": max_loss_display,
        "risk_reward": risk_reward,
    }


def populate_input_sheet(ws, spot_price, legs, sector="", strategy_name=""):
    """Populate the Input sheet while preserving the workbook formatting."""
    ws["B7"] = _smart_round(spot_price)
    ws["B8"] = len(legs)
    ws["B9"] = sector
    ws["B10"] = strategy_name

    leg_columns = ["B", "E", "H", "K"]

    for col in leg_columns:
        for row in range(13, 24):
            ws[f"{col}{row}"] = None

    for i, leg in enumerate(legs[:MAX_LEGS]):
        col = leg_columns[i]

        ws[f"{col}13"] = leg.get("asset_class")
        ws[f"{col}14"] = _smart_round(leg.get("K"))
        ws[f"{col}15"] = str(leg.get("option_type", "")).title()
        ws[f"{col}16"] = str(leg.get("position", "")).title()
        ws[f"{col}17"] = leg.get("current_date")
        ws[f"{col}18"] = leg.get("expiration_date")

        ws[f"{col}19"] = _smart_round(leg.get("T"))
        ws[f"{col}20"] = _smart_round(leg.get("sigma"))
        ws[f"{col}21"] = _smart_round(leg.get("r"))
        ws[f"{col}22"] = _smart_round(leg.get("rf"))
        ws[f"{col}23"] = _smart_round(leg.get("quantity"))


def _clear_output_leg_columns(ws):
    """Clear previous leg output so reports with fewer legs never retain stale data."""
    for col in range(3, 7):  # C:F
        for row in range(7, 14):
            ws.cell(row=row, column=col).value = None

        for row in range(16, 22):  # Increased bound to clear row 21 properly
            ws.cell(row=row, column=col).value = None


def populate_output_sheet(ws, strategy_result):
    """
    Populate Greeks, pricing, and Monte Carlo slide data.

    Column B is Total. Columns C:F are the actual option legs.
    Leg values are signed and quantity-adjusted so they reconcile to Total.
    """
    legs = strategy_result.get("legs", [])
    totals = strategy_result.get("totals", {})

    _clear_output_leg_columns(ws)

    # --------------------------------------------------------------
    # GREEKS
    # --------------------------------------------------------------
    ws["B7"] = "Total"

    greek_rows = {
        "delta": 8,
        "gamma": 9,
        "vega": 10,
        "theta": 11,
        "rho": 12,
        "phi": 13,
    }

    # Helper function to scale raw greeks to AMG institutional conventions
    def _scale_greek(metric, raw_val):
        metric_lower = metric.lower()
        if metric_lower == "vega" or metric_lower == "rho":
            return raw_val / 100.0  # Per 1 percentage point move (1%)
        elif metric_lower == "theta":
            return raw_val / 365.0  # Daily time decay
        return raw_val

    # Populate Total column (Column B) with scaled totals
    for metric, row in greek_rows.items():
        raw_total = totals.get(metric, 0.0)
        ws.cell(row=row, column=2).value = _smart_round(_scale_greek(metric, raw_total))

    # Populate individual leg columns (Columns C:F) with scaled leg greeks
    for i, leg in enumerate(legs[:MAX_LEGS], start=3):
        short_name = _leg_short_name(leg)

        # Compact trade label, e.g. "70 SC"
        ws.cell(row=7, column=i).value = short_name

        sign = _position_sign(leg.get("position"))

        for metric, row in greek_rows.items():
            raw_value = _safe_float(leg.get(metric))
            scaled_value = _scale_greek(metric, raw_value)
            ws.cell(row=row, column=i).value = _smart_round(sign * scaled_value)


    # --------------------------------------------------------------
    # PRICING
    # --------------------------------------------------------------
    ws["B16"] = "Total"
    ws["B18"] = _smart_round(totals.get("price", 0.0))
    ws["B19"] = _smart_round(totals.get("mc_price", 0.0))
    ws["B20"] = _smart_round(strategy_result.get("garch_volatility"))
    
    # Calculate total actual/Bloomberg price across legs (position & quantity adjusted)
    total_actual_price = 0.0
    for leg in legs:
        sign = _position_sign(leg.get("position"))
        qty = _safe_float(leg.get("quantity"), 1.0)
        leg_actual = _safe_float(leg.get("actual_price"))
        if leg_actual <= 1e-8:
            leg_actual = _safe_float(leg.get("price"))
        total_actual_price += sign * qty * leg_actual
    ws["B21"] = _smart_round(total_actual_price)

    for i, leg in enumerate(legs[:MAX_LEGS], start=3):
        short_name = _leg_short_name(leg)

        # Use only the compact trade label, e.g. "65 LC"
        ws.cell(row=16, column=i).value = short_name

        # Model name remains in the model row.
        ws.cell(row=17, column=i).value = _model_name(leg)

        # Individual prices stay raw/per-unit.
        ws.cell(row=18, column=i).value = _smart_round(_safe_float(leg.get("price")))
        ws.cell(row=19, column=i).value = _smart_round(_safe_float(leg.get("mc_price")))

        ws.cell(row=20, column=i).value = _smart_round(
            leg.get("garch_volatility", strategy_result.get("garch_volatility"))
        )

        # Individual Actual / Bloomberg Price
        actual_val = _safe_float(leg.get("actual_price"))
        if actual_val <= 1e-8:
            actual_val = _safe_float(leg.get("price"))
        ws.cell(row=21, column=i).value = _smart_round(actual_val)

    # Make the total section explicit about the analytical model only
    # when all legs use the same model. Otherwise state that multiple
    # analytical models were used.
    models = {_model_name(leg) for leg in legs}
    if len(models) == 1:
        ws["B17"] = next(iter(models))
    else:
        ws["B17"] = "Multiple analytical models"

    # --------------------------------------------------------------
    # TRADE DETAILS
    # --------------------------------------------------------------
    details = calculate_trade_details(legs)
    ws["B24"] = _smart_round(details["max_profit"])   # Shifted down one cell
    ws["B25"] = _smart_round(details["max_loss"])     # Shifted down one cell
    ws["B26"] = _smart_round(details["risk_reward"])  # Shifted down one cell

    # --------------------------------------------------------------
    # MONTE CARLO FOR SLIDES
    # --------------------------------------------------------------
    # Preserve A29/A30 as the section's labels and put only the
    # human-readable leg names across the row (Shifted down one cell).
    for col in range(2, 7):
        ws.cell(row=29, column=col).value = None
        ws.cell(row=30, column=col).value = None

    for i, leg in enumerate(legs[:MAX_LEGS], start=2):
        ws.cell(row=29, column=i).value = _leg_full_name(leg)
        ws.cell(row=30, column=i).value = _smart_round(_safe_float(leg.get("mc_price")))


def populate_payoff_data(ws, spot_price, legs):
    """
    Populate the payoff data table only.

    O = underlying-price grid
    P:S = individual option P&Ls
    T = total strategy P&L

    No Excel chart is created, deleted, reformatted, or otherwise touched.
    """
    chart_start, chart_step, rows = generate_payoff_data(spot_price, legs)

    # Clear the data area first.
    for row in range(PAYOFF_START_ROW, PAYOFF_END_ROW + 1):
        for col in range(15, 21):  # O:T
            ws.cell(row=row, column=col).value = None

    # Headers are useful for the payoff data itself, but the chart is
    # deliberately left to the template/editor.
    for i, leg in enumerate(legs[:MAX_LEGS], start=16):  # P:S
        ws.cell(row=1, column=i).value = _leg_short_name(leg)

    for col in range(16 + len(legs), 20):
        ws.cell(row=1, column=col).value = None

    ws["O1"] = "Underlying Price"
    ws["T1"] = "All"

    for row_idx, row_data in enumerate(rows, start=PAYOFF_START_ROW):
        ws.cell(row=row_idx, column=15).value = _smart_round(row_data["underlying"])

        for leg_idx, value in enumerate(row_data["legs"], start=16):
            ws.cell(row=row_idx, column=leg_idx).value = _smart_round(value)

        ws.cell(row=row_idx, column=20).value = _smart_round(row_data["total"])

    return chart_start, chart_step


def create_excel_report(
    strategy_result,
    spot_price,
    template_path,
    output_path,
    sector="",
    strategy_name="",
):
    """
    Create an Excel report from the provided template.

    The template owns all chart formatting. Python only populates the
    payoff data used by the chart.
    """
    template_path = Path(template_path)
    output_path = Path(output_path)

    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    wb = load_workbook(template_path)

    if "Input" not in wb.sheetnames:
        raise ValueError("Template must contain an 'Input' sheet.")

    if "Output" not in wb.sheetnames:
        raise ValueError("Template must contain an 'Output' sheet.")

    input_ws = wb["Input"]
    output_ws = wb["Output"]

    legs = strategy_result.get("legs", [])

    if len(legs) > MAX_LEGS:
        raise ValueError(f"Excel template supports at most {MAX_LEGS} option legs.")

    populate_input_sheet(
        input_ws,
        spot_price=spot_price,
        legs=legs,
        sector=sector,
        strategy_name=strategy_name,
    )

    populate_output_sheet(
        output_ws,
        strategy_result=strategy_result,
    )

    chart_start, chart_step = populate_payoff_data(
        output_ws,
        spot_price=spot_price,
        legs=legs,
    )

    # Store the calculated chart start/step in the template's existing
    # chart-control cells. (Shifted one cell right: J->K).
    output_ws["K7"] = _smart_round(chart_start)
    output_ws["K8"] = _smart_round(chart_step)

    wb.save(output_path)

    return output_path


def create_report_from_strategy_result(
    strategy_result,
    spot_price,
    template_path,
    output_path,
    sector="",
    strategy_name="",
):
    """Compatibility wrapper used by app.py."""
    return create_excel_report(
        strategy_result=strategy_result,
        spot_price=spot_price,
        template_path=template_path,
        output_path=output_path,
        sector=sector,
        strategy_name=strategy_name,
    )


if __name__ == "__main__":
    print("AMG Excel report generator loaded successfully.")