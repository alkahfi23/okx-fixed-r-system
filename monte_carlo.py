import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# CONFIG
# ===============================
START_BALANCE = 10_000
RISK_PER_TRADE = 0.01      # 1% risk
N_TRADES = 300             # jumlah trade per simulasi
N_SIMULATIONS = 5000       # jumlah simulasi

# Outcome R-multiple
outcomes = np.array([-1.0, 0.8, 2.0])
probabilities = np.array([0.55, 0.20, 0.25])

# ===============================
# MONTE CARLO ENGINE
# ===============================
def run_simulation():
    balance = START_BALANCE
    equity = [balance]

    for _ in range(N_TRADES):
        r = np.random.choice(outcomes, p=probabilities)
        risk_amount = balance * RISK_PER_TRADE
        balance += r * risk_amount
        equity.append(balance)

    return np.array(equity)

# ===============================
# RUN SIMULATIONS
# ===============================
curves = np.array([run_simulation() for _ in range(N_SIMULATIONS)])
final_balances = curves[:, -1]

# ===============================
# METRICS
# ===============================
def max_drawdown(equity):
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return dd.min()

drawdowns = np.array([max_drawdown(curve) for curve in curves])

# ===============================
# RESULTS
# ===============================
print("====== MONTE CARLO RESULT ======")
print(f"Simulations      : {N_SIMULATIONS}")
print(f"Trades / sim     : {N_TRADES}")
print(f"Risk / trade     : {RISK_PER_TRADE*100:.1f}%")
print("--------------------------------")
print(f"Median Final Bal : ${np.median(final_balances):,.0f}")
print(f"Worst Final Bal  : ${final_balances.min():,.0f}")
print(f"Best Final Bal   : ${final_balances.max():,.0f}")
print("--------------------------------")
print(f"Median Max DD    : {np.median(drawdowns)*100:.1f}%")
print(f"Worst Max DD     : {drawdowns.min()*100:.1f}%")
print("--------------------------------")
print(f"Ruin Prob (<50%) : {(final_balances < START_BALANCE*0.5).mean()*100:.2f}%")

# ===============================
# PLOTS
# ===============================
plt.figure(figsize=(10,6))
for i in range(50):
    plt.plot(curves[i], alpha=0.2)
plt.title("Monte Carlo Equity Curves (Sample)")
plt.xlabel("Trades")
plt.ylabel("Equity")
plt.show()

plt.figure(figsize=(8,5))
plt.hist(final_balances, bins=50)
plt.title("Final Balance Distribution")
plt.show()

