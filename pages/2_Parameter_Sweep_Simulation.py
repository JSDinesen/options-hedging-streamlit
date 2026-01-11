import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import plotly.express as px
import plotly.graph_objects as go

from MC import MonteCarlo

DAYS_PER_YEAR = 252
DT = 1 / DAYS_PER_YEAR

st.set_page_config(layout="wide")
st.title("Monte Carlo: Parameter Sweep")

with st.expander("Description"):
    st.write(
        """
        ## Parameter Sweep: Monte Carlo Sensitivity Analysis

        This page runs a **Monte Carlo simulation** many times while changing **one parameter (1D sweep)** or
        **two parameters (2D sweep)** to show how hedging performance and risk change across settings.

        ### Possible parameter sweeps

        #### Hedging frequency (Number of hedges)
        Sweeps how often you rebalance the delta hedge over the option’s life.  
        More hedges usually means **lower hedging error (lower risk)**, but **more trading**.

        #### Volatility (True vs Model)
        Tests **volatility misspecification**:  
        how your results change when the volatility used in the model (**σ_model**) is different from the true volatility (**σ_true**).

        In the **Full 2D Sweep**, you vary **both σ_true and σ_model** and visualize the results as heatmaps.

        #### Transaction costs
        Sweeps the transaction fee (in basis points) and shows how costs and trading affect the strategy.  
        Higher costs don’t just reduce PnL directly — they also matter more when turnover is high.

        ### How the sweep controls work

        All sweeps are controlled by the **Sweep Parameters** at the bottom of the sidebar.

        #### Steps (1D sweeps)
        **Steps** determines how many values you test between the minimum and maximum.

        **Example (1D):**  
        If you sweep σ_model from 0.10 to 0.30 with Steps = 5, the simulator will test:
        0.10, 0.15, 0.20, 0.25, 0.30  
        (5 parameter points)

        #### Steps (2D volatility sweep)
        For the **Full 2D volatility sweep**, Steps is used for **both axes**.  
        This creates a grid of size **Steps × Steps**, so the total number of parameter combinations is:

        **Steps²**

        **Example (2D):**  
        If Steps = 5, you test 5 values for σ_true and 5 values for σ_model, giving:
        5 × 5 = **25** different (σ_true, σ_model) combinations.

        This is why the number of required simulations grows much faster for 2D sweeps.

        ### Model assumptions

        The same assumptions as on the **Single Path** and **Monte Carlo** pages apply here:
        identical GBM stock dynamics, option pricing model, hedging mechanics, and cost/constraint settings.
        Only the selected sweep parameter(s) are changed.
        """
    )

# ----------------------
# Session state init
# ----------------------
if "df_sweep" not in st.session_state:
    st.session_state.df_sweep = None
if "sweep_label" not in st.session_state:
    st.session_state.sweep_label = None

if "vol_sweep_mode" not in st.session_state:
    st.session_state.vol_sweep_mode = None


# ----------------------
# Sidebar inputs
# ----------------------

mc = MonteCarlo()

with st.sidebar:
    st.header("Fixed Parameters")
    st.subheader("Underlying Stock Parameters")
    S0 = st.number_input("Initial stock price", value=100.0, min_value=0.0)
    mu = st.number_input("Drift μ", value=0.05, min_value=-1.0)
    sigma_true = st.number_input("True volatility σ_true", value=0.2, min_value=0.0)


    st.subheader("Option Parameters")
    T = st.number_input("Time to maturity (years)", value=1.0, min_value=0.01)
    K = st.number_input("Strike K", value=100.0, min_value=0.0, step=0.1)
    sigma_model = st.number_input("Model volatility σ_model", value=0.2, min_value=0.0)
    r = st.number_input("Risk-free rate r", value=0.05, min_value=0.0)

    st.subheader("Hedging Parameters")
    num_hedges = st.number_input("Number of hedges", value=52, min_value=2, max_value=int(T*DAYS_PER_YEAR), step=1)
    n_options = st.number_input("Number of options held", value=100, min_value=1, step=1)

    st.subheader("Costs / Constraints")
    txn_bps = st.number_input("Transaction cost (bps)", value=0.0, min_value=0.0, step=1.0)
    integer_stocks = st.checkbox("Integer shares", value=False)

    #Using a radio to select what type of sweep to do
    st.header("Parameter to be swept")
    sweep_param = st.radio("Choose Sweep Parameter", ["Number of hedges", "True/Model volatility", "Transaction costs"])

    #Initialize vol_sweep to None to avoid errors when it is not defined
    vol_sweep = None
    #------------------------------
    # Display of sweep parameters
    #------------------------------
    if sweep_param == "Number of hedges":
        st.header("Sweep Parameters")
        min_hedges = st.number_input("Minimum number of hedges", value=2, min_value=2, step=1)
        max_hedges = st.number_input("Maximum number of hedges", value=int(T*DAYS_PER_YEAR), min_value=min_hedges+1, max_value=int(T*DAYS_PER_YEAR), step=1)
        steps = st.number_input("Steps", value = 10, min_value=2, step=1)

    if sweep_param == "True/Model volatility":
        st.header("Sweep Parameters")
        vol_sweep = st.radio("Choose Type of Volatility Sweep", ["Full 2D Sweep", "Fix True vol", "Fix Model vol"])
        #Changing sweep parameters based on type of vol sweep:
        if vol_sweep == "Full 2D Sweep":
            min_true_vol = st.number_input("Minimum true volatility", value=0.0, min_value=0.0, step=0.01)
            max_true_vol = st.number_input("Maximum true volatility", value=1.0, min_value=min_true_vol + 0.1, step=0.01)
            min_model_vol = st.number_input("Minimum model volatility", value=0.0, min_value=0.0, step=0.01)
            max_model_vol = st.number_input("Maximum model volatility", value=1.0, min_value=min_model_vol+0.1, step=0.01)

        elif vol_sweep == "Fix True vol":
            st.write("Fix True volatility in parameters above")
            min_model_vol = st.number_input("Minimum model volatility", value=0.0, min_value=0.0, step=0.01)
            max_model_vol = st.number_input("Maximum model volatility", value=1.0, min_value=min_model_vol + 0.1, step=0.01)

        elif vol_sweep == "Fix Model vol":
            st.write("Fix model volatility in parameters above")
            min_true_vol = st.number_input("Minimum true volatility", value=0.0, min_value=0.0, step=0.01)
            max_true_vol = st.number_input("Maximum true volatility", value=1.0, min_value=min_true_vol + 0.1, step=0.01)

        steps = st.number_input("Steps", value=10, min_value=2, step=1)

    if sweep_param == "Transaction costs":
        st.header("Sweep Parameters")
        min_tc = st.number_input("Minimum Transaction fee (bps)", value=0, min_value=0, step=1)
        max_tc = st.number_input("Maximum Transaction fee (bps)", value=100, min_value=min_tc+1, step=1)
        steps = st.number_input("Steps", value=10, min_value=2, step=1)

    n_paths = st.number_input("Number of Simulations per step", value=200, min_value=10, step=10)
    base_seed = st.number_input("Base RNG seed", value=12345, min_value=0, step=1)

    if sweep_param is not None:
        run_mc = st.button("▶ Run Parameter Sweep", type="primary")

    #Display number of single path simulations to be run
    if sweep_param == "True/Model volatility" and vol_sweep == "Full 2D Sweep":
        sim_required = steps**2 * n_paths
    else:
        sim_required = steps * n_paths

    st.write("SIMULATIONS REQUIRED:", sim_required)

    #Indicating "runtime" feasibility
    if sim_required < 10000:
        st.badge("FAST RUNTIME", color="green")
    elif 50000 > sim_required >= 10000:
        st.badge("LONG RUNTIME", color="orange")
    elif sim_required >= 50000:
        st.badge("!VERY LONG RUNTIME!", color="red")




# Initialise result list
sweep_rows = []

# ---------------------------------
#    RUN PARAMETER SWEEP (ONLY ON CLICK)
# ---------------------------------
if run_mc:

    #---------------------------------
    #    RUN NUMBER OF HEDGES SWEEP: COMPUTATIONS
    #---------------------------------
    if sweep_param == "Number of hedges":
        #Defining Monte Carlo Parameters grid
        hedge_grid = np.linspace(min_hedges, max_hedges, steps, endpoint=True)
        hedge_grid = np.unique(np.round(hedge_grid).astype(int))

        progress_bar = st.progress(0)

        for count, current_num_hedges in enumerate(hedge_grid):
            #Define the current sim parameters
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
                num_hedges=int(current_num_hedges),
                n_options=int(n_options),
                transaction_bps=float(txn_bps),
                integer_stocks=bool(integer_stocks),
                DT=DT,
            )

            #Run single parameter MC sim
            res = mc.run_monte_carlo(**current_params)

            #Add to result list
            pnl = res["Final PnL"]
            tc = res["Total Transaction Costs"]
            to = res["Turnover"]
            md = res["Max Absolute Delta"]

            sweep_rows.append({
                "hedges": int(current_num_hedges),

                "pnl_mean": float(np.mean(pnl)),
                "pnl_std": float(np.std(pnl, ddof=1)),
                "pnl_p5": float(np.quantile(pnl, 0.05)),
                "pnl_p50": float(np.quantile(pnl, 0.50)),
                "pnl_p95": float(np.quantile(pnl, 0.95)),

                "tc_mean": float(np.mean(tc)),
                "turnover_mean": float(np.mean(to)),
                "max_delta_p95": float(np.quantile(md, 0.95)),
            })

            progress_bar.progress((count + 1) / (len(hedge_grid)), text=f"Sweep {count + 1} of {len(hedge_grid)}")

        # Store results
        st.session_state.df_sweep = pd.DataFrame(sweep_rows)
        st.session_state.sweep_label = "Number of hedges"

    # ---------------------------------
    #    RUN VOLATILITY SWEEP: COMPUTATIONS
    # ---------------------------------
    if sweep_param == "True/Model volatility":
        #Check which type of volatility sweep is being run:
        if vol_sweep == "Full 2D Sweep":
            # Defining Monte Carlo Parameters grid
            true_grid = np.linspace(min_true_vol, max_true_vol, steps, endpoint=True)
            true_grid = np.unique(true_grid)

            model_grid = np.linspace(min_model_vol, max_model_vol, steps, endpoint=True)
            model_grid = np.unique(model_grid)

            progress_bar = st.progress(0)
            count = 0

            #Looping over all options of true vol and model vol
            #(Will cause long runtime...)
            for current_true_vol in true_grid:
                for current_model_vol in model_grid:
                    # Define the current sim parameters
                    current_params = dict(
                        n_paths=int(n_paths),
                        base_seed=int(base_seed),
                        S0=float(S0),
                        mu=float(mu),
                        sigma_true=float(current_true_vol),
                        T=float(T),
                        K=float(K),
                        sigma_model=float(current_model_vol),
                        r=float(r),
                        num_hedges=int(num_hedges),
                        n_options=int(n_options),
                        transaction_bps=float(txn_bps),
                        integer_stocks=bool(integer_stocks),
                        DT=DT,
                    )

                    # Run single parameter MC sim
                    res = mc.run_monte_carlo(**current_params)

                    # Add to result list
                    pnl = res["Final PnL"]
                    tc = res["Total Transaction Costs"]
                    to = res["Turnover"]
                    md = res["Max Absolute Delta"]

                    sweep_rows.append({
                        "hedges": int(num_hedges),

                        "pnl_mean": float(np.mean(pnl)),
                        "pnl_std": float(np.std(pnl, ddof=1)),
                        "pnl_p5": float(np.quantile(pnl, 0.05)),
                        "pnl_p50": float(np.quantile(pnl, 0.50)),
                        "pnl_p95": float(np.quantile(pnl, 0.95)),

                        "tc_mean": float(np.mean(tc)),
                        "turnover_mean": float(np.mean(to)),
                        "max_delta_p95": float(np.quantile(md, 0.95)),

                        "sigma_true": float(current_true_vol),
                        "sigma_model": float(current_model_vol),
                    })

                    #Updating progress bar
                    progress_bar.progress((count + 1) / (len(true_grid)**2), text=f"Sweep {count + 1} of {len(true_grid)**2}")
                    count += 1

        elif vol_sweep == "Fix True vol":
            # Defining Monte Carlo Parameters grid
            model_grid = np.linspace(min_model_vol, max_model_vol, steps, endpoint=True)
            model_grid = np.unique(model_grid)

            progress_bar = st.progress(0)

            for count, current_model_vol in enumerate(model_grid):
                # Define the current sim parameters
                current_params = dict(
                    n_paths=int(n_paths),
                    base_seed=int(base_seed),
                    S0=float(S0),
                    mu=float(mu),
                    sigma_true=float(sigma_true),
                    T=float(T),
                    K=float(K),
                    sigma_model=float(current_model_vol),
                    r=float(r),
                    num_hedges=int(num_hedges),
                    n_options=int(n_options),
                    transaction_bps=float(txn_bps),
                    integer_stocks=bool(integer_stocks),
                    DT=DT,
                )

                # Run single parameter MC sim
                res = mc.run_monte_carlo(**current_params)

                # Add to result list
                pnl = res["Final PnL"]
                tc = res["Total Transaction Costs"]
                to = res["Turnover"]
                md = res["Max Absolute Delta"]

                # Useful extra metrics for TC sweep
                eps = 1e-9
                cost_share = tc / (np.abs(pnl) + eps)

                sweep_rows.append({
                    "model_volatility": current_model_vol,

                    "pnl_mean": float(np.mean(pnl)),
                    "pnl_std": float(np.std(pnl, ddof=1)),
                    "pnl_p5": float(np.quantile(pnl, 0.05)),
                    "pnl_p50": float(np.quantile(pnl, 0.50)),
                    "pnl_p95": float(np.quantile(pnl, 0.95)),

                    "tc_mean": float(np.mean(tc)),
                    "tc_p95": float(np.quantile(tc, 0.95)),

                    "turnover_mean": float(np.mean(to)),
                    "turnover_p95": float(np.quantile(to, 0.95)),

                    "max_delta_p95": float(np.quantile(md, 0.95)),

                    "cost_share_mean": float(np.mean(cost_share)),
                    "cost_share_p95": float(np.quantile(cost_share, 0.95)),
                })

                progress_bar.progress((count + 1) / (len(model_grid)), text=f"Sweep {count + 1} of {len(model_grid)}")

        elif vol_sweep == "Fix Model vol":
            # Defining Monte Carlo Parameters grid
            true_grid = np.linspace(min_true_vol, max_true_vol, steps, endpoint=True)
            true_grid = np.unique(true_grid)

            progress_bar = st.progress(0)

            for count, current_true_vol in enumerate(true_grid):
                # Define the current sim parameters
                current_params = dict(
                    n_paths=int(n_paths),
                    base_seed=int(base_seed),
                    S0=float(S0),
                    mu=float(mu),
                    sigma_true=float(current_true_vol),
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

                # Run single parameter MC sim
                res = mc.run_monte_carlo(**current_params)

                # Add to result list
                pnl = res["Final PnL"]
                tc = res["Total Transaction Costs"]
                to = res["Turnover"]
                md = res["Max Absolute Delta"]

                # Useful extra metrics for TC sweep
                eps = 1e-9
                cost_share = tc / (np.abs(pnl) + eps)

                sweep_rows.append({
                    "true_volatility": current_true_vol,

                    "pnl_mean": float(np.mean(pnl)),
                    "pnl_std": float(np.std(pnl, ddof=1)),
                    "pnl_p5": float(np.quantile(pnl, 0.05)),
                    "pnl_p50": float(np.quantile(pnl, 0.50)),
                    "pnl_p95": float(np.quantile(pnl, 0.95)),

                    "tc_mean": float(np.mean(tc)),
                    "tc_p95": float(np.quantile(tc, 0.95)),

                    "turnover_mean": float(np.mean(to)),
                    "turnover_p95": float(np.quantile(to, 0.95)),

                    "max_delta_p95": float(np.quantile(md, 0.95)),

                    "cost_share_mean": float(np.mean(cost_share)),
                    "cost_share_p95": float(np.quantile(cost_share, 0.95)),
                })

                progress_bar.progress((count + 1) / (len(true_grid)), text=f"Sweep {count + 1} of {len(true_grid)}")

        # Store results
        st.session_state.df_sweep = pd.DataFrame(sweep_rows)
        st.session_state.sweep_label = "True/Model volatility"
        st.session_state.vol_sweep_mode = vol_sweep

    # ---------------------------------
    #    RUN TRANSACTION COSTS SWEEP: COMPUTATIONS
    # ---------------------------------
    if sweep_param == "Transaction costs":
        #Defining Monte Carlo Parameters grid
        tc_grid = np.linspace(min_tc, max_tc, steps, endpoint=True)
        tc_grid = np.unique(np.round(tc_grid).astype(int))

        progress_bar = st.progress(0)

        for count, current_tc in enumerate(tc_grid):
            #Define the current sim parameters
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
                transaction_bps=float(current_tc),
                integer_stocks=bool(integer_stocks),
                DT=DT,
            )

            #Run single parameter MC sim
            res = mc.run_monte_carlo(**current_params)

            #Add to result list
            pnl = res["Final PnL"]
            tc = res["Total Transaction Costs"]
            to = res["Turnover"]
            md = res["Max Absolute Delta"]

            # Useful extra metrics for TC sweep
            eps = 1e-9
            cost_share = tc / (np.abs(pnl) + eps)

            sweep_rows.append({
                "transaction_bps": int(current_tc),

                "pnl_mean": float(np.mean(pnl)),
                "pnl_std": float(np.std(pnl, ddof=1)),
                "pnl_p5": float(np.quantile(pnl, 0.05)),
                "pnl_p50": float(np.quantile(pnl, 0.50)),
                "pnl_p95": float(np.quantile(pnl, 0.95)),

                "tc_mean": float(np.mean(tc)),
                "tc_p95": float(np.quantile(tc, 0.95)),

                "turnover_mean": float(np.mean(to)),
                "turnover_p95": float(np.quantile(to, 0.95)),

                "max_delta_p95": float(np.quantile(md, 0.95)),

                "cost_share_mean": float(np.mean(cost_share)),
                "cost_share_p95": float(np.quantile(cost_share, 0.95)),
            })

            progress_bar.progress((count + 1) / (len(tc_grid)), text=f"Sweep {count + 1} of {len(tc_grid)}")

        # Store results
        st.session_state.df_sweep = pd.DataFrame(sweep_rows)
        st.session_state.sweep_label = "Transaction costs"


df_sweep = st.session_state.df_sweep

if df_sweep is None:
    st.info("Click **Run Parameter Sweep** to generate results.")
else:
    st.caption(f"Showing results for: {st.session_state.sweep_label}")

    # ============================================================
    # NUMBER OF HEDGES SWEEP: PLOTTING
    # ============================================================
    if st.session_state.sweep_label == "Number of hedges":
        c1, c2 = st.columns(2)

        # ------------------------------------------------------------
        # Column 1.1 — Risk reduction vs hedge frequency
        # ------------------------------------------------------------
        c1.caption("PnL Standard Deviation vs Number of Hedges")
        c1.line_chart(df_sweep.set_index("hedges")["pnl_std"])

        with c1.expander("Description", expanded=False):
            st.write(
                """
                Shows how portfolio risk (PnL standard deviation) decreases as the hedge frequency increases.
                This illustrates the classical variance–reduction effect of more frequent delta hedging.
                """
            )

        # ------------------------------------------------------------
        # Column 1.2 — Distribution shape across hedge frequencies
        # ------------------------------------------------------------
        c2.caption("PnL 5th and 95th Quantile vs Number of Hedges")

        base = alt.Chart(df_sweep).encode(x="hedges:Q")

        band = base.mark_area(opacity=0.25).encode(
            y="pnl_p5:Q",
            y2="pnl_p95:Q"
        )

        line = base.mark_line().encode(
            y="pnl_mean:Q"
        )

        c2.altair_chart(band + line, use_container_width=True)

        with c2.expander("Description", expanded=False):
            st.write(
                """
                Visualizes the central 90% of outcomes (5th–95th percentile) together with mean PnL.
                Narrowing bands indicate reduced tail risk from more frequent hedging.
                """
            )

        d1, d2, d3 = st.columns(3)

        # ------------------------------------------------------------
        # Column 2.1 — Trading intensity cost proxy
        # ------------------------------------------------------------
        d1.caption("Turnover vs Hedge Frequency")
        d1.line_chart(df_sweep.set_index("hedges")["turnover_mean"])

        with d1.expander("Description", expanded=False):
            st.write(
                """
                Measures how trading activity increases with hedge frequency.
                Turnover serves as a proxy for execution costs and operational burden.
                """
            )

        # ------------------------------------------------------------
        # Column 2.2 — Risk–cost trade-off frontier
        # ------------------------------------------------------------
        d2.caption("Risk–Cost Frontier")

        chart = alt.Chart(df_sweep).mark_circle(size=80, opacity=0.6).encode(
            x=alt.X("turnover_mean:Q", title="Mean turnover"),
            y=alt.Y("pnl_std:Q", title="PnL standard deviation"),
            tooltip=["hedges", "pnl_std", "turnover_mean"]
        )

        d2.altair_chart(chart, use_container_width=True)

        with d2.expander("Description", expanded=False):
            st.write(
                """
                Illustrates the trade-off between risk reduction and trading intensity.
                Efficient hedging regimes lie near the lower-left boundary of the frontier.
                """
            )

        # ------------------------------------------------------------
        # Column 2.3 — Tail-risk compression
        # ------------------------------------------------------------
        d3.caption("Tail Risk vs Hedge Frequency")

        df_sweep = df_sweep.copy()
        df_sweep["pnl_range90"] = df_sweep["pnl_p95"] - df_sweep["pnl_p5"]
        d3.line_chart(df_sweep.set_index("hedges")["pnl_range90"])

        with d3.expander("Description", expanded=False):
            st.write(
                """
                Shows how the width of the PnL distribution tails shrinks as hedge frequency increases.
                Useful for evaluating downside protection rather than average performance.
                """
            )

    # ============================================================
    # VOLATILITY SWEEP: PLOTTING
    # ============================================================
    if st.session_state.sweep_label == "True/Model volatility":
        vol_mode = st.session_state.get("vol_sweep_mode", None)
        if vol_mode == "Full 2D Sweep":
            #Plot 2D Sweep features
            df = st.session_state.df_sweep.copy()

            df["vol_error"] = df["sigma_model"] - df["sigma_true"]
            df["vol_ratio"] = df["sigma_model"] / df["sigma_true"]


            # ----------------------------
            # Helper: plotly heatmap (diverging OR sequential)
            # ----------------------------
            def plotly_market_heatmap(
                    df: pd.DataFrame,
                    value_col: str,
                    title: str,
                    x_col: str = "sigma_model",
                    y_col: str = "sigma_true",
                    clip_quantile: float = 0.90,
                    height: int = 520,
                    diverging: bool = True,  # True: red-black-green centered at 0
                    sequential_scale=None,  # for non-negative metrics
            ):
                mat = df.pivot(index=y_col, columns=x_col, values=value_col).sort_index().sort_index(axis=1)
                z = mat.to_numpy(dtype=float)

                finite = z[np.isfinite(z)]
                if finite.size == 0:
                    raise ValueError(f"No finite values to plot for {value_col}.")

                if diverging:
                    # symmetric around 0
                    zmax_abs = float(np.quantile(np.abs(finite), clip_quantile))
                    zmax_abs = max(zmax_abs, 1e-12)
                    zmin, zmax = -zmax_abs, +zmax_abs
                    colorscale = [
                        [0.0, "#ff2a2a"],  # vivid red
                        [0.5, "#000000"],  # black at 0
                        [1.0, "#00ff66"],  # vivid green
                    ]
                else:
                    # sequential for >=0 metrics
                    zmax = float(np.quantile(finite, clip_quantile))
                    zmax = max(zmax, 1e-12)
                    zmin = float(np.nanmin(finite))
                    if not np.isfinite(zmin):
                        zmin = 0.0
                    colorscale = sequential_scale or [
                        [0.0, "#000000"],  # black low
                        [1.0, "#00ff66"],  # green high (fits your theme)
                    ]

                fig = px.imshow(
                    mat,
                    origin="lower",
                    aspect="auto",
                    color_continuous_scale=colorscale,
                    zmin=zmin,
                    zmax=zmax,
                )

                fig.update_layout(
                    title=title,
                    template="plotly_dark",
                    height=height,
                    margin=dict(l=60, r=20, t=60, b=60),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    coloraxis_colorbar=dict(
                        title=dict(text=value_col, font=dict(color="white")),
                        tickcolor="white",
                        tickfont=dict(color="white"),
                    ),
                )

                fig.update_xaxes(title="Model volatility σ_model", showgrid=False, zeroline=False)
                fig.update_yaxes(title="True volatility σ_true", showgrid=False, zeroline=False)

                return fig

            #--------------------------
            # Helper function adding diagonal where sigma_model = sigma_true
            #--------------------------
            def add_diag(fig, df, x_col="sigma_model", y_col="sigma_true"):
                vmin = float(min(df[y_col].min(), df[x_col].min()))
                vmax = float(max(df[y_col].max(), df[x_col].max()))
                fig.add_trace(
                    go.Scatter(
                        x=[vmin, vmax],
                        y=[vmin, vmax],
                        mode="lines",
                        line=dict(color="white", width=2, dash="dash"),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
                return fig


            #Defining blue scale for non PnL metrics
            BLUE_SCALE = [
                [0.0, "#000000"],  # black (low)
                [0.3, "#0b3c5d"],  # deep blue
                [0.6, "#1f77b4"],  # classic plotly blue
                [1.0, "#7fdbff"],  # light cyan
            ]

            # ----------------------------
            # Derived metrics
            # ----------------------------
            eps = 1e-12
            df = df.copy()
            df["sharpe_like"] = df["pnl_mean"] / (df["pnl_std"] + eps)

            # -------------------------
            # Row 1 — Performance
            # -------------------------
            c1, c2 = st.columns(2)

            # 1) Mean PnL (diverging)
            fig1 = plotly_market_heatmap(
                df, "pnl_mean", "Mean Final PnL",
                clip_quantile=0.90, height=520, diverging=True
            )
            add_diag(fig1, df)
            c1.plotly_chart(fig1, use_container_width=True)

            with c1.expander("Description", expanded=False):
                st.write(
                    """
                    This shows the **average final hedging PnL** for each combination of (σ_true, σ_model).
                    The dashed diagonal is where the model volatility equals the true volatility.
                    Off the diagonal it can be seen how **volatility misspecification** creates systematic gains/losses.
                    """
                )

            # 2) Sharpe-like (diverging)
            fig2 = plotly_market_heatmap(
                df, "sharpe_like", "Sharpe-like: Mean / Std",
                clip_quantile=0.90, height=520, diverging=True
            )
            add_diag(fig2, df)
            c2.plotly_chart(fig2, use_container_width=True)

            with c2.expander("Description", expanded=False):
                st.write(
                    """
                    This is **mean PnL divided by PnL standard deviation**, so it’s a quick “risk-adjusted” score.
                    Higher means you get **more PnL per unit of risk**.
                    If it blows up in some regions, it’s usually because the std is very small there.
                    """
                )

            # -------------------------
            # Row 2 — Risk
            # -------------------------
            c3, c4 = st.columns(2)

            # 3) PnL Standard Deviation (sequential)
            fig3 = plotly_market_heatmap(
                df, "pnl_std", "PnL Standard Deviation",
                diverging=False, sequential_scale=BLUE_SCALE,
                clip_quantile=0.90
            )
            add_diag(fig3, df)
            c3.plotly_chart(fig3, use_container_width=True)

            with c3.expander("Description", expanded=False):
                st.write(
                    """
                    This shows how **spread out** the final PnL outcomes are (risk / uncertainty).
                    Bigger values mean the hedging result is **less stable** across Monte Carlo paths.
                    Use this to see where the strategy becomes risky when your vol assumption is wrong.
                    """
                )

            # 4) PnL 5% Quantile (Downside) (diverging)
            fig4 = plotly_market_heatmap(
                df, "pnl_p5", "PnL 5% Quantile (Downside)",
                clip_quantile=0.90, height=520, diverging=True
            )
            add_diag(fig4, df)
            c4.plotly_chart(fig4, use_container_width=True)

            with c4.expander("Description", expanded=False):
                st.write(
                    """
                    This is the **bad-case PnL**: the 5th percentile.
                    Meaning: in the worst ~5% of paths, your final PnL is around this value.
                    This is a clean way to see **downside risk** from volatility misspecification.
                    """
                )

            # -------------------------
            # Row 3 — Operational Cost
            # -------------------------
            # 5) Mean Turnover (sequential) — full width
            fig5 = plotly_market_heatmap(
                df, "turnover_mean", "Mean Turnover",
                clip_quantile=0.90, height=520,
                diverging=False, sequential_scale=BLUE_SCALE
            )
            add_diag(fig5, df)
            st.plotly_chart(fig5, use_container_width=True)

            with st.expander("Description", expanded=False):
                st.write(
                    """
                    This shows how much you **trade on average** (turnover) for each (σ_true, σ_model).
                    Higher turnover means you are rebalancing more aggressively, which usually means:
                    more **transaction costs**.
                    """
                )

        else:
            #Plot 1D Sweep features
            df_sweep = st.session_state.df_sweep.copy()

            # Detect x-axis column depending on which 1D sweep has been selected:
            if "model_volatility" in df_sweep.columns:
                x_col = "model_volatility"
                x_label = "Model volatility σ_model"
                sweep_title = "1D Vol Sweep (Fix σ_true, sweep σ_model)"
            elif "true_volatility" in df_sweep.columns:
                x_col = "true_volatility"
                x_label = "True volatility σ_true"
                sweep_title = "1D Vol Sweep (Fix σ_model, sweep σ_true)"
            elif "sigma_model" in df_sweep.columns and "sigma_true" not in df_sweep.columns:
                x_col = "sigma_model"
                x_label = "Model volatility σ_model"
                sweep_title = "1D Vol Sweep"
            elif "sigma_true" in df_sweep.columns and "sigma_model" not in df_sweep.columns:
                x_col = "sigma_true"
                x_label = "True volatility σ_true"
                sweep_title = "1D Vol Sweep"
            else:
                st.error("Could not detect the swept volatility column for 1D plotting.")
                st.stop()

            # Sort for clean lines
            df_sweep = df_sweep.sort_values(x_col).reset_index(drop=True)

            # Derived metrics
            df_sweep["pnl_range90"] = df_sweep["pnl_p95"] - df_sweep["pnl_p5"]

            # Sharpe-like (Mean / Std) — mask tiny std to avoid blow-ups
            eps = 1e-12
            df_sweep["sharpe_like"] = df_sweep["pnl_mean"] / (df_sweep["pnl_std"])

            #std_floor = df_sweep["pnl_std"].quantile(0.05)  # tune 0.01–0.10
            #df_sweep.loc[df_sweep["pnl_std"] < std_floor, "sharpe_like"] = np.nan
            df_sweep["sharpe_like"] = df_sweep["sharpe_like"].replace([np.inf, -np.inf], np.nan)

            st.subheader(sweep_title)

            c1, c2 = st.columns(2)

            # ------------------------------------------------------------
            # Column 1.1 — Mean PnL vs volatility parameter
            # ------------------------------------------------------------
            c1.caption(f"Mean Final PnL vs {x_label}")
            c1.line_chart(df_sweep.set_index(x_col)["pnl_mean"])

            with c1.expander("Description", expanded=False):
                st.write(
                    f"""
                    Shows how the average hedging PnL changes as you vary the volatility assumption.
                    This is your **model misspecification bias** curve: systematic profit/loss caused by using
                    the wrong volatility in pricing/hedging.
                    """
                )

            # ------------------------------------------------------------
            # Column 1.2 — Sharpe-like efficiency vs volatility parameter
            # ------------------------------------------------------------
            c2.caption(f"Sharpe-like (Mean / Std) vs {x_label}")
            c2.line_chart(df_sweep.set_index(x_col)["sharpe_like"])

            with c2.expander("Description", expanded=False):
                st.write(
                    f"""
                    Summarizes **reward per unit risk** as the volatility assumption changes.
                    Values are masked where PnL standard deviation is extremely small to avoid numerical blow-ups.
                    """
                )

            d1, d2 = st.columns(2)

            # ------------------------------------------------------------
            # Column 2.1 — PnL risk (Std) vs volatility parameter
            # ------------------------------------------------------------
            d1.caption(f"PnL Standard Deviation vs {x_label}")
            d1.line_chart(df_sweep.set_index(x_col)["pnl_std"])

            with d1.expander("Description", expanded=False):
                st.write(
                    f"""
                    Measures how dispersion of outcomes changes with the volatility assumption.
                    Useful for seeing whether misspecification increases variability even if mean PnL looks acceptable.
                    """
                )

            # ------------------------------------------------------------
            # Column 2.2 — Downside risk (5% quantile) vs volatility parameter
            # ------------------------------------------------------------
            d2.caption(f"PnL 5% Quantile (Downside) vs {x_label}")
            d2.line_chart(df_sweep.set_index(x_col)["pnl_p5"])

            with d2.expander("Description", expanded=False):
                st.write(
                    f"""
                    Focuses on **tail outcomes** (bad scenarios). This is often more informative than the mean:
                    it tells you how wrong volatility assumptions can hurt you in stressed paths.
                    """
                )

            e1, e2, e3 = st.columns(3)

            # ------------------------------------------------------------
            # Column 3.1 — Tail width (p95 - p5) vs volatility parameter
            # ------------------------------------------------------------
            e1.caption(f"Tail Width (p95 − p5) vs {x_label}")
            e1.line_chart(df_sweep.set_index(x_col)["pnl_range90"])

            with e1.expander("Description", expanded=False):
                st.write(
                    f"""
                    Shows the **width of the central 90% of outcomes**. A wider band means more tail uncertainty.
                    This is often easier to interpret than standard deviation for hedging distributions.
                    """
                )

            # ------------------------------------------------------------
            # Column 3.2 — Trading intensity (turnover) vs volatility parameter
            # ------------------------------------------------------------
            e2.caption(f"Mean Turnover vs {x_label}")
            e2.line_chart(df_sweep.set_index(x_col)["turnover_mean"])

            with e2.expander("Description", expanded=False):
                st.write(
                    f"""
                    Turnover measures how much trading the hedge requires. It’s a proxy for **operational load**
                    and sensitivity to transaction costs/slippage.
                    """
                )

            # ------------------------------------------------------------
            # Column 3.3 — Max hedge stress proxy (optional but very useful)
            # ------------------------------------------------------------
            e3.caption(f"Max |Delta| (95% quantile) vs {x_label}")
            e3.line_chart(df_sweep.set_index(x_col)["max_delta_p95"])

            with e3.expander("Description", expanded=False):
                st.write(
                    f"""
                    Shows how extreme the hedge ratio becomes in stressed paths.
                    Large values indicate more aggressive hedging requirements (and more sensitivity to discretization).
                    """
                )


    # ============================================================
    # TRANSACTION COSTS SWEEP: PLOTTING
    # ============================================================
    if st.session_state.sweep_label == "Transaction costs":
        c1, c2 = st.columns(2)

        # ------------------------------------------------------------
        # Column 1.1 — Risk sensitivity to transaction costs
        # ------------------------------------------------------------
        c1.caption("PnL Standard Deviation vs Transaction Costs (bps)")
        c1.line_chart(df_sweep.set_index("transaction_bps")["pnl_std"])

        with c1.expander("Description", expanded=False):
            st.write(
                """
                Shows how transaction costs indirectly affect risk by limiting effective hedging.
                Higher costs discourage trading, increasing residual risk.
                """
            )

        # ------------------------------------------------------------
        # Column 1.2 — Distribution shift under transaction costs
        # ------------------------------------------------------------
        c2.caption("PnL 5th and 95th Quantile vs Transaction Costs (bps)")

        base = alt.Chart(df_sweep).encode(x="transaction_bps:Q")

        band = base.mark_area(opacity=0.25).encode(
            y="pnl_p5:Q",
            y2="pnl_p95:Q"
        )

        line = base.mark_line().encode(
            y="pnl_mean:Q"
        )

        c2.altair_chart(band + line, use_container_width=True)

        with c2.expander("Description", expanded=False):
            st.write(
                """
                Demonstrates how transaction costs shift and widen the PnL distribution,
                increasing downside risk even when the hedging strategy remains unchanged.
                """
            )

        d1, d2, d3 = st.columns(3)

        # ------------------------------------------------------------
        # Column 2.1 — Absolute cost paid
        # ------------------------------------------------------------
        d1.caption("Average Transaction Costs Paid vs Transaction Costs (bps)")
        d1.line_chart(df_sweep.set_index("transaction_bps")["tc_mean"])

        with d1.expander("Description", expanded=False):
            st.write(
                """
                Displays the realized transaction costs paid by the strategy.
                Confirms linear or nonlinear scaling between quoted fees and actual costs.
                """
            )

        # ------------------------------------------------------------
        # Column 2.2 — Risk–cost efficiency
        # ------------------------------------------------------------
        d2.caption("Risk–Cost Frontier")

        xmin = df_sweep["tc_mean"].min()
        xmax = df_sweep["tc_mean"].max()

        chart = alt.Chart(df_sweep).mark_circle(size=80, opacity=0.6).encode(
            x=alt.X("tc_mean:Q", title="Mean total transaction costs",
                    scale = alt.Scale(domain=[xmin,xmax])),
            y=alt.Y("pnl_std:Q", title="PnL standard deviation"),
            tooltip=["transaction_bps", "pnl_std", "tc_mean", "turnover_mean", "cost_share_mean"])

        d2.altair_chart(chart, use_container_width=True)

        with d2.expander("Description", expanded=False):
            st.write(
                """
                Highlights the efficiency trade-off between risk reduction and transaction costs.
                Useful for identifying cost regimes where hedging becomes economically inefficient.
                """
            )

        # ------------------------------------------------------------
        # Column 2.3 — Tail-risk amplification
        # ------------------------------------------------------------
        d3.caption("Tail Risk vs Transaction Costs (bps)")

        df_sweep = df_sweep.copy()
        df_sweep["pnl_range90"] = df_sweep["pnl_p95"] - df_sweep["pnl_p5"]
        d3.line_chart(df_sweep.set_index("transaction_bps")["pnl_range90"])

        with d3.expander("Description", expanded=False):
            st.write(
                """
                Shows how transaction costs amplify tail risk by reducing hedge responsiveness.
                Particularly relevant for stress scenarios and worst-case analysis.
                """
            )

        with st.expander("Raw DataFrame", expanded=False):
            st.write(df_sweep)
