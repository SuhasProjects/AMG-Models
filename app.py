from flask import Flask, render_template, request, session, send_file, jsonify
import pandas as pd
from io import StringIO
from pathlib import Path
import os
import copy

from monte_carlo.monte_carlo import load_price_data
from reports.excel_report import (
    create_report_from_strategy_result, 
    generate_payoff_data, 
    calculate_trade_details, 
    _leg_pnl, 
    calculate_chart_range
)

from strategy.strategy import (
    create_leg,
    get_implied_volatility,
    price_leg,
    price_strategy,
    aggregate_legs,
    calculate_time_to_maturity
)

app = Flask(__name__)
# Secure environment variable mapping for PaaS deployment
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "amg-options-tool-dev")

BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = BASE_DIR / "templates" / "AMG_Excel.xlsx"
REPORT_DIR = BASE_DIR / "reports"
REPORT_PATH = REPORT_DIR / "AMG_Options_Report.xlsx"

@app.route("/download-report")
def download_report():
    if not REPORT_PATH.exists():
        return "No report has been generated yet.", 404

    return send_file(
        REPORT_PATH,
        as_attachment=True,
        download_name="AMG_Options_Report.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# -------------------------------------------------------------------
# MAIN DASHBOARD / REPORT GENERATOR
# -------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    cleaned_data = None
    error = None
    leg = None
    
    legs = session.get("legs", [])
    totals = session.get("totals", None)

    if request.method == "POST":
        try:
            action = request.form.get("action")

            if action == "clear":
                session["legs"] = []
                session["totals"] = None
                session["report_ready"] = False
                session.modified = True
                legs = []
                totals = None

            elif action == "add_leg":
                historical_text = _required_form_value("historical_data")
                data = pd.read_csv(StringIO(historical_text), sep=None, engine="python")
                cleaned_data = load_price_data(data)
                S = float(cleaned_data["Last Price"].iloc[-1])

                asset_class = _required_form_value("asset_class")
                K = float(_required_form_value("K"))
                current_date = _required_form_value("current_date")
                expiration_date = _required_form_value("expiration_date")
                r = float(_required_form_value("r"))
                q = float(request.form.get("q", "0") or 0)
                option_type = _required_form_value("option_type")
                position = _required_form_value("position")
                
                # UPDATED: Contract Size & Multiplier Math
                contracts = int(request.form.get("contracts", "1"))
                multiplier = float(request.form.get("multiplier", "100"))
                quantity = contracts * multiplier 
                
                actual_price = float(request.form.get("actual_price", "0"))

                if actual_price < 0:
                    raise ValueError("Bloomberg option price cannot be negative.")

                rf = None
                if asset_class.lower() == "fx":
                    rf = float(_required_form_value("foreign_risk_free_rate"))

                T = calculate_time_to_maturity(current_date, expiration_date)

                volatility_method = _required_form_value("volatility_method")

                # Route the correct dividend yield (q) based on the asset class before solving IV
                iv_q = q
                if asset_class.lower() == "fx":
                    iv_q = rf
                elif asset_class.lower() == "futures":
                    iv_q = r

                if volatility_method == "manual":
                    sigma = float(_required_form_value("volatility"))
                elif volatility_method == "implied":
                    sigma = get_implied_volatility(
                        market_price=actual_price,
                        S=S, K=K, T=T, r=r, q=iv_q,
                        option_type=option_type,
                    )
                else:
                    raise ValueError("Invalid volatility input method.")

                leg = create_leg(
                    asset_class=asset_class, S=S, K=K,
                    current_date=current_date, expiration_date=expiration_date,
                    r=r, q=q, sigma=sigma, option_type=option_type,
                    position=position, quantity=quantity, rf=rf,
                    actual_price=actual_price,
                )

                leg = price_leg(leg)
                leg["actual_price"] = actual_price
                leg["risk_free_rate"] = r
                leg["volatility_method"] = volatility_method
                leg["current_date"] = current_date
                leg["expiration_date"] = expiration_date
                
                # Store display fields
                leg["contracts"] = contracts
                leg["multiplier"] = multiplier
                
                leg["mc_price"] = None
                leg["garch_volatility"] = None

                legs.append(leg)
                session["legs"] = legs
                session["totals"] = None
                session["report_ready"] = False
                session.modified = True
                totals = None

            elif action == "done":
                if not legs:
                    raise ValueError("Add at least one option before calculating.")

                historical_text = _required_form_value("historical_data")
                historical_data = pd.read_csv(StringIO(historical_text), sep=None, engine="python")
                cleaned_data = load_price_data(historical_data)

                strategy_result = price_strategy(
                    historical_data=historical_data,
                    legs=legs,
                    n_paths=10000,
                    trading_days=252,
                    seed=42,
                )

                legs = strategy_result["legs"]
                totals = strategy_result["totals"]
                
                session["legs"] = legs
                session["totals"] = totals
                session["spot_price"] = float(cleaned_data["Last Price"].iloc[-1])
                session["report_ready"] = False
                session.modified = True

            elif action == "generate":
                if not legs or not totals:
                    raise ValueError("Calculate the strategy first.")

                historical_text = _required_form_value("historical_data")
                historical_data = pd.read_csv(StringIO(historical_text), sep=None, engine="python")
                cleaned_data = load_price_data(historical_data)

                strategy_result = {"legs": legs, "totals": totals}
                REPORT_DIR.mkdir(exist_ok=True)
                spot_price = session.get("spot_price", legs[0]["S"])

                create_report_from_strategy_result(
                    strategy_result=strategy_result,
                    spot_price=spot_price,
                    template_path=TEMPLATE_PATH,
                    output_path=REPORT_PATH,
                )

                session["report_ready"] = True
                session.modified = True

        except Exception as exc:
            error = str(exc)

    return render_template(
        "index.html",
        cleaned_data=cleaned_data,
        error=error,
        leg=leg,
        legs=legs,
        totals=totals,
        historical_data=request.form.get("historical_data", ""),
        report_ready=session.get("report_ready", False),
    )


# -------------------------------------------------------------------
# VISUALIZATION PAGES (No historical data required)
# -------------------------------------------------------------------

@app.route('/pnl')
def pnl_page():
    return render_template('pnl.html')
@app.route('/greeks')
def greeks_page():
    return render_template('greeks.html')
@app.route('/methodology')
def methodology_page():
    return render_template('methodology.html')

def _safe_float(value, default=0.0):
    """Safely cast empty strings or partial inputs to a float without crashing."""
    if value is None or str(value).strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default

def _parse_sandbox_leg_dict(data, global_spot, global_tte=30):
    """Helper to parse a leg from JSON without requiring historical data."""
    asset_class = data.get("asset_class", "equity")
    K = _safe_float(data.get("K"), 100.0)
    r = _safe_float(data.get("r"), 0.05)
    q = _safe_float(data.get("q"), 0.0)
    option_type = data.get("option_type", "call")
    position = data.get("position", "long")
    
    contracts = int(_safe_float(data.get("contracts"), 1))
    multiplier = _safe_float(data.get("multiplier"), 100.0)
    quantity = contracts * multiplier
    
    actual_price = _safe_float(data.get("actual_price"), 0.0)
    
    T = max(1, _safe_float(global_tte, 30)) / 365.0 
    rf = _safe_float(data.get("foreign_risk_free_rate"), 0.0) if asset_class == "fx" else None

    volatility_method = data.get("volatility_method", "manual")
    iv_q = rf if asset_class == "fx" else (r if asset_class == "futures" else q)
    
    if volatility_method == "manual":
        sigma = _safe_float(data.get("volatility"), 0.2)
    else:
        try:
            # If the user is mid-typing, IV might fail to converge, so we use a try/except
            sigma = get_implied_volatility(actual_price, global_spot, K, T, r, iv_q, option_type)
        except:
            sigma = 0.2 

    leg = create_leg(
        asset_class=asset_class, S=global_spot, K=K, current_date="2025-01-01", 
        expiration_date="2025-01-31", r=r, q=q, sigma=sigma, 
        option_type=option_type, position=position, quantity=quantity, 
        rf=rf, actual_price=actual_price
    )
    leg["T"] = T # Override with exact TTE
    return price_leg(leg)


@app.route('/api/pnl', methods=['POST'])
def api_pnl():
    req = request.json
    spot = float(req.get("spot_price", 100))
    tte_days = float(req.get("tte_days", 30))
    raw_legs = req.get("legs", [])
    
    parsed_legs = [_parse_sandbox_leg_dict(l, spot, tte_days) for l in raw_legs]
    
    if not parsed_legs:
        return jsonify({"error": "No legs provided."}), 400

    chart_start, chart_step, rows = generate_payoff_data(spot, parsed_legs)
    details = calculate_trade_details(parsed_legs)
    
    chart_data = {
        "underlying": [r["underlying"] for r in rows],
        "total": [r["total"] for r in rows]
    }
    return jsonify({"chart": chart_data, "details": details})

@app.route('/api/greeks', methods=['POST'])
def api_greeks():
    req = request.json
    spot = float(req.get("spot_price", 100))
    tte_days = float(req.get("tte_days", 30))
    raw_legs = req.get("legs", [])
    
    parsed_legs = [_parse_sandbox_leg_dict(l, spot, tte_days) for l in raw_legs]

    if not parsed_legs:
        return jsonify({"error": "No legs provided."}), 400
    
    chart_start, chart_step = calculate_chart_range(spot, parsed_legs, points=100)
    
    results = {
        'chart': {
            'underlying': [], 'payoff_exp': [], 'pnl_tte': [], 
            'delta': [], 'gamma': [], 'vega': [], 'theta': [], 'rho': []
        },
        'aggregates': {},
        'legs': []
    }
    
    # --- Calculate Current State (Top Bar Aggregates & Table Details) ---
    current_priced_legs = []
    initial_cost = 0
    for l in parsed_legs:
        temp_l = copy.deepcopy(l)
        temp_l['S'] = spot
        temp_l['T'] = max(1, tte_days) / 365.0
        try:
            p_leg = price_leg(temp_l)
            current_priced_legs.append(p_leg)
            # Track cost basis: Price * Qty * Sign
            qty = p_leg.get("quantity", 1)
            sign = 1 if p_leg.get("position") == "long" else -1
            initial_cost += p_leg.get("actual_price", 0) * qty * sign
        except:
            pass
            
    current_totals = aggregate_legs(current_priced_legs)
    
    results['aggregates'] = {
        'delta': current_totals.get('delta', 0),
        'gamma': current_totals.get('gamma', 0),
        'vega': current_totals.get('vega', 0) / 100.0,
        'theta': current_totals.get('theta', 0) / 365.0,
        'rho': current_totals.get('rho', 0) / 100.0,
        'theor': current_totals.get('price', 0),
        'cost': initial_cost
    }
    
    for leg in current_priced_legs:
        results['legs'].append({
            'iv': leg.get('sigma', 0) * 100, # Display as percentage
            'theor': leg.get('price', 0),
            'delta': leg.get('delta', 0),
            'gamma': leg.get('gamma', 0),
            'vega': leg.get('vega', 0) / 100.0,
            'theta': leg.get('theta', 0) / 365.0,
            'rho': leg.get('rho', 0) / 100.0
        })
        
    # --- Calculate Chart Arrays (The 100-point curve) ---
    for i in range(100):
        current_spot = chart_start + i * chart_step
        results['chart']['underlying'].append(current_spot)
        
        priced_legs = []
        theo_value_tte = 0
        for leg in parsed_legs:
            temp_leg = copy.deepcopy(leg)
            temp_leg['S'] = current_spot
            temp_leg['T'] = max(1, tte_days) / 365.0 
            try:
                p_leg = price_leg(temp_leg)
                priced_legs.append(p_leg)
                qty = p_leg.get("quantity", 1)
                sign = 1 if p_leg.get("position") == "long" else -1
                theo_value_tte += p_leg.get("price", 0) * qty * sign
            except:
                pass
        
        totals = aggregate_legs(priced_legs)
        payoff_exp = sum(_leg_pnl(current_spot, l) for l in priced_legs)
        
        results['chart']['payoff_exp'].append(payoff_exp)
        results['chart']['pnl_tte'].append(theo_value_tte - initial_cost)
        results['chart']['delta'].append(totals.get('delta', 0))
        results['chart']['gamma'].append(totals.get('gamma', 0))
        results['chart']['vega'].append(totals.get('vega', 0) / 100.0) 
        results['chart']['theta'].append(totals.get('theta', 0) / 365.0) 
        results['chart']['rho'].append(totals.get('rho', 0) / 100.0) 
            
    return jsonify(results)


def _required_form_value(name):
    value = request.form.get(name)
    if value is None or not value.strip():
        raise ValueError(f"Missing form field: {name}")
    return value.strip()

if __name__ == "__main__":
    app.run(debug=True)