from simulator import Simulator
from brownian_motion import BrownianMotion

import streamlit as st
import numpy as np

DAYS_PER_YEAR = 252
DT = 1 / DAYS_PER_YEAR

# -----------------------------
# Session state init (once)
# -----------------------------
if "stock" not in st.session_state:
    st.session_state.stock = None
if "stock_seed" not in st.session_state:
    st.session_state.stock_seed = None

# Store BOTH outputs now
if "df_fulltime" not in st.session_state:
    st.session_state.df_fulltime = None
if "df_hedge" not in st.session_state:
    st.session_state.df_hedge = None

# Track previous-run totals (for deltas) - based on the currently selected df view
if "prev_pnl" not in st.session_state:
    st.session_state.prev_pnl = None
if "prev_tc" not in st.session_state:
    st.session_state.prev_tc = None
if "prev_shares" not in st.session_state:
    st.session_state.prev_shares = None

# Initialize previous delta stats once
if "prev_mean_abs_delta" not in st.session_state:
    st.session_state.prev_mean_abs_delta = None
if "prev_max_abs_delta" not in st.session_state:
    st.session_state.prev_max_abs_delta = None
if "prev_final_delta" not in st.session_state:
    st.session_state.prev_final_delta = None

st.set_page_config(layout="wide")
st.title("Dynamic Delta Hedging Simulator")
with st.expander("Description"):
    st.write(
        """
        ## Single-Path Dynamic Delta Hedging Simulator

        This page simulates **one single stock path** and applies **dynamic delta hedging** along that path.
        Instead of averaging over many outcomes, you see exactly **what happens step by step** for one realization
        of the market.
        
        Monte Carlo Simulation and Parameter Sweep Simulation can be found at the top of the sidebar

        ### What this page shows
        - A single **Geometric Brownian Motion stock path**, generated either randomly or from a fixed seed.
        - A dynamically rebalanced **delta hedge** applied to an option position.
        - The resulting **PnL, transaction costs, and hedge positions** over time.

        ### Underlying stock
        - The stock follows a **Geometric Brownian Motion** with drift μ and true volatility σ_true.
        - Using a fixed RNG seed allows you to reproduce the exact same path.

        ### Option pricing
        - The option is priced using the **Black–Scholes model**.
        - Pricing and delta calculations use the **model volatility σ_model**, not σ_true.
        - This separation allows you to study **model misspecification**: the market evolves with σ_true,
          while the hedge is constructed using σ_model.

        ### Hedging strategy
        - The hedge consists of trading the underlying stock to offset the option delta.
        - The position is rebalanced a fixed number of times over the option’s lifetime.
        - Between hedge times, delta drifts due to stock movements and time decay.

        ### How PnL is determined
        - The simulation starts with an **initial cash balance**, which grows at the risk-free rate r.
        - Cash flows from stock trades and transaction costs are continuously accounted for.
        - At maturity, the option payoff is realized and all positions are closed.
        - **Total PnL** is the difference between the final portfolio value and the value of the initial
          investment compounded at the risk-free rate.
        - This isolates the economic effect of hedging performance from passive cash growth.

        ### FULL TIME vs HEDGING TIME view
        **FULL TIME (ON)**  
        Portfolio values are marked-to-market at every time step.
        Useful for understanding continuous PnL evolution.

        **HEDGING TIMES (OFF)**  
        Only shows values at hedge dates.
        Highlights actual trading decisions and transaction costs.

        ### Key metrics
        - **Total PnL**: Final profit or loss of the hedged option position.
        - **Transaction costs**: Accumulated costs from trading the hedge.
        - **Shares traded**: Total trading volume required by the hedge.
        - **Overall delta**: Combined delta of the stock and option positions over time.
          Ideally this stays close to zero, indicating effective hedging.

        ### Why this page matters
        This single-path view builds intuition.
        You can directly see how delta hedging works, how costs accumulate, how delta drifts between hedge times,
        and how incorrect volatility assumptions translate into realized profit or loss — before moving to
        Monte Carlo averages.
        """
    )

st.divider()

# -----------------------------
# Sidebar inputs
# -----------------------------
with st.sidebar:
    st.header("Underlying Stock Parameters")
    S0 = st.number_input("Initial stock price", value=100.0, min_value=0.0)
    mu = st.number_input("Drift μ", value=0.05, min_value=-1.0)
    sigma_true = st.number_input("True volatility σ_true", value=0.2, min_value=0.0)
    seed = st.number_input("RNG seed", value=12345, min_value=0, step=1)
    gen_ran_stock = st.button("Generate Random Stock", type="secondary")
    gen_seed_stock = st.button("Generate Seeded Stock", type="secondary")

    st.header("Option Parameters")
    T = st.number_input("Time to maturity (years)", value=1.0, min_value=0.01)
    K = st.number_input("Strike K", value=100.0, min_value=0.0, step=0.1)
    sigma_model = st.number_input("Model volatility σ_model", value=0.2, min_value=0.0)

    st.header("Hedging Parameters")
    num_hedges = st.number_input("Number of hedges", value=52, min_value=1, step=1)
    n_options = st.number_input("Number of options held", value=100, min_value=1, step=1)

    st.header("General Parameters")
    cash0 = st.number_input("Initial cash balance", value=50_000, min_value=0, step=100)
    r = st.number_input("Risk-free rate r", value=0.05, min_value=0.0)
    txn_bps = st.number_input("Transaction cost (bps)", value=0.0, min_value=0.0, step=1.0)
    integer_stocks = st.checkbox("Integer shares", value=False)

    #toggle which dataframe drives plots/tables
    use_fulltime = st.toggle(
        "Plot FULL TIME (mark-to-market every step)",
        value=True,
        help="ON: full-time dataframe (evaluated every dt). OFF: hedging-time dataframe (only hedge dates).",
    )

    run = st.button("▶ Run simulation", type="primary")

# -----------------------------
# Stock generation (only on click)
# -----------------------------
bm = BrownianMotion(x_0=S0, t=T, delta_t=DT, mu=mu, sigma=sigma_true)

if gen_ran_stock:
    rng = np.random.default_rng()
    random_seed = int(rng.integers(0, 2**32 - 1, dtype=np.uint32))
    st.session_state.stock_seed = random_seed
    st.session_state.stock = bm.geometric_brownian_motion(seed=random_seed)
    # clear last results when you change the stock
    st.session_state.df_fulltime = None
    st.session_state.df_hedge = None

if gen_seed_stock:
    st.session_state.stock_seed = int(seed)
    st.session_state.stock = bm.geometric_brownian_motion(seed=int(seed))
    # clear last results when you change the stock
    st.session_state.df_fulltime = None
    st.session_state.df_hedge = None

# -----------------------------
# Plot stock (sticky)
# -----------------------------
st.subheader("Underlying Stock")
if st.session_state.stock is None:
    st.info("Generate a stock path in the sidebar.")
else:
    st.caption(f"Seed: {st.session_state.stock_seed}")
    st.line_chart(st.session_state.stock)

st.divider()

# Helper: pick currently selected df for display
def get_df_for_view():
    return st.session_state.df_fulltime if use_fulltime else st.session_state.df_hedge

# -----------------------------
# Run hedging sim (compute + store dfs)
# -----------------------------
if run:
    if st.session_state.stock is None:
        st.warning("Generate a stock path first.")
    else:
        # Snapshot previous totals BEFORE overwriting dfs (so deltas compare to last run)
        old_df_view = get_df_for_view()
        if old_df_view is not None:
            st.session_state.prev_pnl = float(old_df_view["Total Pnl"].iloc[-1])
            st.session_state.prev_tc = float(old_df_view["Transaction costs"].sum()) if "Transaction costs" in old_df_view else 0.0
            st.session_state.prev_shares = float(old_df_view["Hedge in this timestep"].abs().sum()) if "Hedge in this timestep" in old_df_view else 0.0

        total_days = int(round(DAYS_PER_YEAR * T))
        num_hedges_int = max(2, int(num_hedges))

        #Setting the hedging time indexes
        hedge_idx = np.linspace(0, total_days, num=num_hedges_int)
        hedge_idx = np.round(hedge_idx).astype(int)
        hedge_idx[0] = 0
        hedge_idx[-1] = total_days
        hedge_idx = np.unique(hedge_idx)

        sim = Simulator(
            time_unit="days",
            transaction_fee_bps=txn_bps,
            integer_stocks=integer_stocks,
        )

        df_fulltime, df_hedge = sim.run_path(
            num_options=int(n_options),
            stock_time_step=DT,
            num_hedges=int(len(hedge_idx)),
            sigma_model=sigma_model,
            risk_free_rate=r,
            option_strike_price=K,
            init_cash_balance=cash0,
            stock_prices=st.session_state.stock,
        )

        st.session_state.df_fulltime = df_fulltime
        st.session_state.df_hedge = df_hedge

df = get_df_for_view()

st.subheader("Results")

#Display result of previous run simulation
if df is not None:
    st.caption(f"Showing: {'FULL TIME' if use_fulltime else 'HEDGING TIMES'}  •  rows: {len(df)}")

#---------------------------------
# Helper function used to calculate percentage change from last run
#----------------------------------
def pct_change_str(curr, prev):
    if prev in (None, 0):
        return None
    return f"{(curr / prev - 1):.2%}"

#----------------------------
# plotting layout on the streamlit website
#----------------------------
if df is None:
    st.info("Click **Run simulation** to generate results.")
else:
    total_pnl = float(df["Total Pnl"].iloc[-1])
    total_tc = float(df["Transaction costs"].sum()) if "Transaction costs" in df else 0.0
    total_shares = float(df["Hedge in this timestep"].abs().sum()) if "Hedge in this timestep" in df else 0.0

    prev_pnl = st.session_state.prev_pnl
    prev_tc = st.session_state.prev_tc
    prev_shares = st.session_state.prev_shares

    #----------------------
    # First row of metrics / Plots
    #----------------------
    c1, c2, c3 = st.columns(3)

    #C1.1: Total PnL
    c1.metric("Total PnL", f"{total_pnl:,.2f}", delta=pct_change_str(total_pnl, prev_pnl))
    c1.line_chart(df[["Total Pnl"]])


    #--------------------------------
    # helper: keep only non-zero entries (i.e., hedge times) for bar charts
    #--------------------------------
    def sparse_bars(series, eps=0.0):
        s = series.copy()
        if eps > 0:
            return s[np.abs(s) > eps]
        return s[s != 0]


    #C1.2: Transaction costs
    c2.metric(
        "Total Transaction Costs",
        f"{total_tc:,.2f}",
        delta=pct_change_str(total_tc, prev_tc),
        delta_color="inverse",
    )
    if "Transaction costs" in df:
        tc_series = df["Transaction costs"]

        # show bars only at hedge times (non-zero costs)
        tc_bars = sparse_bars(tc_series)

        if tc_bars.empty:
            c2.info("No transaction costs incurred.")
        else:
            c2.bar_chart(tc_bars)
    else:
        c2.info("No transaction costs column found.")

    #C1.3: Shares traded
    c3.metric("Total Shares Traded", f"{total_shares:,.0f}", delta=pct_change_str(total_shares, prev_shares))
    if "Hedge in this timestep" in df:
        trade_series = df["Hedge in this timestep"]

        # show bars only at hedge times (non-zero trades)
        trade_bars = sparse_bars(trade_series)

        if trade_bars.empty:
            c3.info("No hedges executed.")
        else:
            c3.bar_chart(trade_bars)
    else:
        c3.info("No 'Hedge in this timestep' column found.")

    st.divider()

    # -----------------------------
    # Overall Delta Position
    # -----------------------------
    st.subheader("Overall Delta Position")

    overall_delta = (
        df["Shares held"].astype(float)
        + df["Options held"].astype(float) * df["Option Delta"].astype(float)
    )

    mean_abs_delta = overall_delta.abs().mean()
    max_abs_delta = overall_delta.abs().max()

    final_delta = overall_delta.iloc[-2] if len(overall_delta) >= 2 else overall_delta.iloc[-1]

    #------------------------
    # Second row of Metrics
    #------------------------
    c4, c5, c6 = st.columns(3)

    #C2.1: Mean overall delta
    c4.metric(
        "Mean |Overall Delta|",
        f"{mean_abs_delta:,.4f}",
        delta=pct_change_str(mean_abs_delta, st.session_state.prev_mean_abs_delta),
        delta_color="inverse",
    )

    #C2.2: Max Overall Delta
    c5.metric(
        "Max |Overall Delta|",
        f"{max_abs_delta:,.4f}",
        delta=pct_change_str(max_abs_delta, st.session_state.prev_max_abs_delta),
        delta_color="inverse",
    )

    #C3.2: Pre close overall delta
    c6.metric(
        "Pre-Close Overall Delta",
        f"{final_delta:,.4f}",
        delta=pct_change_str(final_delta, st.session_state.prev_final_delta),
    )

    # DELTA POSITION PLOT!
    st.line_chart(overall_delta.rename("Overall Delta"))

    with st.expander("Delta position data"):
        delta_table = df[["Option Delta", "Options held", "Shares held"]].copy()
        delta_table["Overall Delta"] = overall_delta
        st.dataframe(delta_table, use_container_width=True)

    # Update stored values AFTER rendering
    st.session_state.prev_mean_abs_delta = mean_abs_delta
    st.session_state.prev_max_abs_delta = max_abs_delta
    st.session_state.prev_final_delta = final_delta

    st.divider()
    st.subheader("Simulation table")
    with st.expander("Show full simulation data"):
        st.dataframe(df, use_container_width=True)
