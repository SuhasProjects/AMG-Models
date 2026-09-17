from flask import Flask, render_template, request, session, send_file
import pandas as pd
from io import StringIO
from pathlib import Path
import os

from monte_carlo.monte_carlo import load_price_data
from reports.excel_report import create_report_from_strategy_result

from strategy.strategy import (
    create_leg,
    get_implied_volatility,
    price_leg,
    price_strategy,
    aggregate_legs
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
                quantity = int(request.form.get("quantity", "1"))
                actual_price = float(request.form.get("actual_price", "0"))

                if actual_price < 0:
                    raise ValueError("Bloomberg option price cannot be negative.")

                rf = None
                if asset_class.lower() == "fx":
                    rf = float(_required_form_value("foreign_risk_free_rate"))

                from strategy.strategy import calculate_time_to_maturity
                T = calculate_time_to_maturity(current_date, expiration_date)

                volatility_method = _required_form_value("volatility_method")

                # FIX 1: Route the correct dividend yield (q) based on the asset class before solving IV
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

def _required_form_value(name):
    value = request.form.get(name)
    if value is None or not value.strip():
        raise ValueError(f"Missing form field: {name}")
    return value.strip()

if __name__ == "__main__":
    app.run()