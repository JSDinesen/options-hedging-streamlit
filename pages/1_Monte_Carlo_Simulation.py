import numpy as np
import altair as alt
import streamlit as st
import pandas as pd

from MC import MonteCarlo

DAYS_PER_YEAR = 252
DT = 1 / DAYS_PER_YEAR

st.set_page_config(layout="wide")
st.title("Monte Carlo: Fixed Parameter Distributions")

with st.expander("Description", expanded=False):
    st.write(
        """
        ## Monte Carlo Simulation: Fixed-Parameter Distributions

        This page runs a **Monte Carlo simulation** of the delta-hedging strategy under **fixed parameters**.
        Instead of looking at a single stock path, it simulates **many independent paths** and studies the
        distribution of outcomes.

        ### What this page shows
        - Thousands of simulated stock paths generated under the same assumptions.
        - A delta-hedged option strategy applied independently to each path.
        - The resulting **distributions** of PnL, transaction costs, turnover, and hedge stress.
        - Summary risk metrics such as VaR and Expected Shortfall.

        ### Underlying stock dynamics
        - Each path follows a **Geometric Brownian Motion** with drift μ and true volatility σ_true.
        - All paths are independent but reproducible through the base RNG seed.

        ### Interpreting the distributions
        - **Terminal PnL**: Shows the full outcome distribution of the hedged strategy.
          The vertical line at zero separates profitable from loss-making paths.
        - **Transaction Costs**: Indicates how expensive the hedging strategy is across scenarios.
        - **Turnover**: Measures total trading activity.
        - **Max |Delta|**: Captures the worst hedge stress experienced along each path.

        ### Risk metrics
        - **Mean PnL** summarizes average performance.
        - **VaR (5% / 95%)** highlights tail outcomes.
        - **Expected Shortfall (ES5)** measures average losses in the worst 5% of cases.
        - These metrics are essential for understanding downside risk, not just averages.

        ### Why this page matters
        The Monte Carlo view moves beyond intuition from single paths.
        It shows how robust (or fragile) the hedging strategy is across many market realizations,
        making it possible to compare hedging frequency, volatility assumptions, and cost structures
        on a statistically meaningful basis.
        """
    )

# ----------------------
# Session state init
# ----------------------
if "mc_results" not in st.session_state:
    st.session_state.mc_results = None
if "mc_params" not in st.session_state:
    st.session_state.mc_params = None

mc = MonteCarlo()


# ----------------------
# Helper: summarize array (generic)
# ----------------------
def summarize(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "p5": float(np.quantile(x, 0.05)),
        "p50": float(np.quantile(x, 0.50)),
        "p95": float(np.quantile(x, 0.95)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


# ----------------------
# Helper: histogram chart (bin slider)
# ----------------------
def plot_hist(x, title, bins=40):
    s = pd.Series(x, name="value").dropna()
    df = s.to_frame()

    # Convert "bins" into a fixed bin width ("step") so it reacts to the slider
    data_min = float(s.min())
    data_max = float(s.max())
    # guard against constant arrays
    span = max(data_max - data_min, 1e-12)
    bin_step = span / float(bins)

    hist = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("value:Q", bin=alt.Bin(step=bin_step), title=title),
            y=alt.Y("count()", title="Frequency"),
            tooltip=[alt.Tooltip("count()", title="Count")]
        )
        .properties(height=300)
    )
    return hist

#----------------------
# Helper plotting vertical reference line at x=0
#-----------------------
def vline(x_value=0.0, color="red", stroke_width=2):
    return (
        alt.Chart(pd.DataFrame({"x": [x_value]}))
        .mark_rule(color=color, strokeWidth=stroke_width)
        .encode(x="x:Q")
    )


# ----------------------
# Sidebar inputs
# ----------------------
with st.sidebar:
    st.header("Monte Carlo Settings")
    n_paths = st.number_input("Number of paths", value=2000, min_value=10, step=10)
    base_seed = st.number_input("Base RNG seed", value=12345, min_value=0, step=1)

    st.header("Underlying Stock Parameters")
    S0 = st.number_input("Initial stock price", value=100.0, min_value=0.0)
    mu = st.number_input("Drift μ", value=0.05, min_value=-1.0)
    sigma_true = st.number_input("True volatility σ_true", value=0.2, min_value=0.0)

    st.header("Option Parameters")
    T = st.number_input("Time to maturity (years)", value=1.0, min_value=0.01)
    K = st.number_input("Strike K", value=100.0, min_value=0.0, step=0.1)
    sigma_model = st.number_input("Model volatility σ_model", value=0.2, min_value=0.0)
    r = st.number_input("Risk-free rate r", value=0.05, min_value=0.0)

    st.header("Hedging Parameters")
    num_hedges = st.number_input("Number of hedges", value=52, min_value=2, max_value=int(T*DAYS_PER_YEAR), step=1)
    n_options = st.number_input("Number of options held", value=100, min_value=1, step=1)

    st.header("Costs / Constraints")
    txn_bps = st.number_input("Transaction cost (bps)", value=0.0, min_value=0.0, step=1.0)
    integer_stocks = st.checkbox("Integer shares", value=False)

    run_mc = st.button("▶ Run Monte Carlo", type="primary")


# ----------------------
# Non-sim controls (Histogram bin slider)
# ----------------------
bins = st.slider("Histogram resolution (higher = finer bins)", min_value=5, max_value=150, value=30, step=1)

current_params = dict(
    n_paths=int(n_paths),
    base_seed=int(base_seed),
    S0=float(S0),
    mu=float(mu),
    sigma_true=float(sigma_true),
    T=float(T),
    K=float(K),
    sigma_model=float(sigma_model),
    r=float(r),
    num_hedges=int(num_hedges),
    n_options=int(n_options),
    transaction_bps=float(txn_bps),
    integer_stocks=bool(integer_stocks),
    DT=DT,
)

#Watning user if parameters have changed since last run
warn_slot = st.empty()
if (not run_mc) and (st.session_state.mc_params is not None) and (current_params != st.session_state.mc_params):
    warn_slot.warning("Parameters changed since the last Monte Carlo run. Click **Run Monte Carlo** to refresh results.")
else:
    warn_slot.empty()

if (st.session_state.mc_results is None) and (not run_mc):
    st.info("Set parameters in the sidebar and click **Run Monte Carlo**.")
    st.stop()


# ----------------------
# Run MC only when button clicked (using your class)
# ----------------------
if run_mc:
    #Added a spinner to show that program is running, since runtime can be long!
    with st.spinner("Running Monte Carlo..."):
        res = mc.run_monte_carlo(**current_params)

    st.session_state.mc_results = res
    st.session_state.mc_params = current_params
    st.success("Done.")
    st.divider()


# ----------------------
# Load previous results
# ----------------------
res = st.session_state.mc_results

pnl_terminal = res["Final PnL"]
tc_total = res["Total Transaction Costs"]
turnover = res["Turnover"]
max_abs_delta = res["Max Absolute Delta"]


# ----------------------
# Summary of MC results
# ----------------------
summary = mc.summarize_mc(res)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Terminal PnL (mean)", f"{summary['mean_pnl']:.2f}")
c2.metric("Terminal PnL (VaR5 / VaR95)", f"{summary['var5']:.2f} / {summary['var95']:.2f}")
c3.metric("Expected Shortfall (ES5)", f"{summary['es5']:.2f}")
#c4.metric("Avg Transaction Costs", f"{summary['mean_tc']:.2f}")
c4.metric("Avg Max |Delta|", f"{summary['mean_max_abs_delta']:.2f}")


# ----------------------
# Distributions
# ----------------------
st.subheader("Distributions")
d1, d2 = st.columns(2)

d1.write("Terminal PnL")
d1.altair_chart(plot_hist(pnl_terminal, "Total PnL", bins=bins) + vline(0.0), use_container_width=True)

d2.write("Transaction Costs")
d2.altair_chart(plot_hist(tc_total, "Transaction Costs", bins=bins), use_container_width=True)

st.subheader("More diagnostics")
e1, e2 = st.columns(2)

e1.write("Turnover (|shares traded|)")
e1.altair_chart(plot_hist(turnover, "Turnover", bins=bins), use_container_width=True)

e2.write("Max |Overall Delta|")
e2.altair_chart(plot_hist(max_abs_delta, "Max |Delta|", bins=bins), use_container_width=True)

with st.expander("Raw arrays"):
    st.write("PnL terminal:", pnl_terminal)
    st.write("TC total:", tc_total)
    st.write("Turnover:", turnover)
    st.write("Max |Delta|:", max_abs_delta)
