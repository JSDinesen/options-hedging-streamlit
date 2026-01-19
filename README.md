# Option Hedging Simulator (Python)

A Python + Streamlit application for exploring **dynamic delta hedging** of European options under Black–Scholes assumptions.

The project focuses on **numerical implementation, data handling, and visual analysis**, using simulated data.

## What it does
- Simulates stock price paths using Geometric Brownian Motion
- Prices European options and computes Greeks (delta) via Black–Scholes
- Applies discrete-time delta hedging with configurable parameters
- Runs **single-path simulations**, **Monte Carlo simulations**, and **parameter sweeps**
- Tracks PnL, transaction costs, hedge positions, and portfolio value
- Visualises results interactively using Streamlit

## Tech stack
- Python
- NumPy, pandas
- Streamlit
- matplotlib / Altair / Plotly

## Live demo
👉 https://options-hedging-app-ss6jwgrbu6gxxnv2dpxymm.streamlit.app

## Motivation
Built as a personal project to deepen my Python skills and gain hands-on experience with quantitative simulation, numerical methods, and end-to-end analytical tool development beyond coursework.

## Notes
- Uses **simulated data** (educational / exploratory purpose)
- Designed for clarity and experimentation rather than production use
