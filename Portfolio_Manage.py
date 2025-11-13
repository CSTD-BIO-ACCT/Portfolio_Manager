import numpy as np
import pandas as pd
import streamlit as st
import numpy_financial as npf

st.set_page_config(page_title="Case Study - Hasso Plattner Foundation (10y)", layout="wide")

#PRIMARY INPUTS FOR SIMULATION
st.sidebar.header("Primary Inputs")
nav0_eur = st.sidebar.slider("Initial NAV (EUR)", min_value=10000, max_value = 2000000, value=1000000, step=1000)
cashflows_affect_investments = st.sidebar.checkbox("Cashflows Affect Investments?", value=True)
w_equity = st.sidebar.slider("Initial Equity Allocation (rest Bonds)", 0.0, 100.0, 50.0, 0.1)/100
mu_s = st.sidebar.slider("Annualised Stock Growth", -10.0, 20.0, 8.0, 0.1)/100
sigma_s = st.sidebar.number_input("Stock Volatility", value=16.8, step=0.1, format="%.3f")/100
mu_b = st.sidebar.slider("Annualised Bond Growth",  -10.0, 10.0, 4.4, 0.1)/100
sigma_b = st.sidebar.number_input("Bond Volatility", value=4.8, step=0.1, format="%.3f")/100
rho = st.sidebar.slider("Stock Bond Correlation", -0.99, 0.99, -0.2, 0.01)

#ECONOMIC INPUTS TO CALCULATE MORTGAGE RATE AND FOREIGN EXCHANGE
st.sidebar.header("Economic Conditions")
ecb = st.sidebar.slider("ECB Policy rate (%)", -10.00, 10.00, 2.15, 0.05)/100
fed = st.sidebar.slider("Federal Funds rate (%)", -10.00, 10.00, 4.00, 0.05)/100
fx0 = st.sidebar.number_input("Starting FX USD per EUR", value=1.15, step=0.01, format="%.2f")
sigma_fx = st.sidebar.number_input("FX volatility", value=0.05, step=0.005, format="%.3f")

#Calculate rf_de and mortgage rates from ECB rate
rf_de = ecb + 0.0055
mortgage_de = rf_de + 0.015

st.sidebar.subheader("Bakery Model Settings")
bakery_yield_Y = st.sidebar.slider("Annual Yield (%)", 0.0, 0.05, 0.02, step=0.005)
bakery_bow_shape = st.sidebar.slider("Bow Shape: ", 1.0, 30.0, 5.0, step = 0.1)
years_of_purchase = st.sidebar.number_input("Years of bakery purchases", 1, 10, 5)

st.sidebar.header("Simulation")
years = 10
steps_per_year = st.sidebar.selectbox("Steps per year", [1,4,12], index=2)
n_paths = st.sidebar.slider("Monte Carlo paths", 500, 20000, 5000, 100)
rebalance = st.sidebar.checkbox("Rebalance yearly to target weights?", value=True)

def cholesky_from_corr(rho):
    cov = np.array([[1.0, rho],[rho,1.0]])
    return np.linalg.cholesky(cov)

#Calcultes the Bakery Cashflows with volatility
def bakery_cashflows(nav0, n_paths=1, years=10):
    commit = nav0 / 3
    IRR_variance = max(0.1, 0.005*bakery_bow_shape)
    IRR_rates = np.random.normal(1.2, IRR_variance, size=n_paths)
    invest = np.array([commit / years_of_purchase if t < years_of_purchase else 0.0 for t in range(years)])
    cashflows = np.zeros((n_paths, years))
    B = max(1.0, bakery_bow_shape)
    for p in range(n_paths):
        nav = np.zeros(years + 1)
        payouts = np.zeros(years)
        IRR_rate = IRR_rates[p]
        for t in range(1, years + 1):
            nav_g = nav[t - 1] * (IRR_rate)
            rd = (min(t, years)/years)**B
            rate = max(bakery_yield_Y, rd)
            d = rate * nav_g
            if t == years:
                d = nav_g + invest[t - 1]  # clean-out
                nav[t] = 0.0
            else:
                nav[t] = nav_g + invest[t - 1] - d
            payouts[t - 1] = d
        cashflows[p, :] = payouts - invest
    return cashflows

#Calculates Donation Cash Flow based on current NAV
def donation(nav0, nav_prev_y):
    return -.025 * nav0 - (0.025 * nav_prev_y)
#Calculates Apartment Cashflow for 10y, requires market_rate the market interest rate, since unlevered returns accounts for appreciation + yield
#This could be reshaped similar to the bow in Bakery
def apartment_cashflows(nav0):
    mortgage_payment = npf.pmt(mortgage_de, 3, 3*nav0/21)
    #In this case Apartment Cashflows interact with Stock/Bond Portfolio, positive are reinvested, negative are drawn down, the unlevered returns are "realised" yearly and are taken out and put into portfolio
    if cashflows_affect_investments:
        inflows = [mortgage_payment  if t < 3 else 0 for t in range(10)]
        unlevered_returns = [(10*nav0/21)*(0.05)  for t in range(10)]
        return [inflows[t] + unlevered_returns[t] for t in range(10)]
    else:
        outflows = [mortgage_payment if t < 3 else 0 for t in range(10)]
        unlevered_returns = [(10*nav0/21)*(1.05**t) for t in range(10)]
        return [0.05*10*nav0/21 + outflows[t]  if t == 0 else unlevered_returns[t] + outflows[t] - unlevered_returns[t-1] for t in range(10)]

#This function works to find 3 primary things:
#A) It finds the Growth path using Euler-Naruyama
#B) It simulates this n_paths number of times
#C) It uses this to predict USD/EUR outflow given an initial investment
def simulate(
    nav0_eur, fx0, w_equity,
    mu_s, sigma_s, mu_b, sigma_b, rho,
    ecb, fed, sigma_fx,
    years, steps_per_year, n_paths,
    rebalance, rf_de=0.02, seed=42
):
    rng = np.random.default_rng(seed)
    #Number of Steps - Also modify dt for this
    n_steps = years * steps_per_year
    dt = 1.0 / steps_per_year
    #convert to log for GBM
    mu_s = np.log1p(mu_s)
    mu_b = np.log1p(mu_b)
    #FX modeled as EUR per USD
    fx0_eur_per_usd = 1.0 / fx0
    mu_fx = (fed - ecb)
    drift_fx = (2*mu_fx - 0.5 * sigma_fx**2) * dt
    #Geometric Brownian Motion Initialise
    drift_s = (mu_s - 0.5 * sigma_s**2) * dt
    drift_b = (mu_b - 0.5 * sigma_b**2) * dt
    #Correlation between stock/bonds
    L = cholesky_from_corr(rho)
    #Initialize array for ForEx
    Y = np.full((n_paths, n_steps + 1), fx0_eur_per_usd)
    #Convert initial EUR NAV to USD
    nav0_usd = nav0_eur * fx0
    #Initial equity/bond values
    eq_val = np.full(n_paths, w_equity * nav0_usd)
    bd_val = np.full(n_paths, (1.0 - w_equity) * nav0_usd)
    #Store total portfolio value (USD) in empty array
    port_usd = np.zeros((n_paths, n_steps + 1))
    port_usd[:, 0] = eq_val + bd_val
    #Calculate Growth Path
    for t in range(1, n_steps + 1):
        #Draw correlated shocks for equity & bonds
        z = rng.normal(size=(n_paths, 2))
        eps = z @ L.T
        #Stochastic Variables initialised accordingly
        dWs = np.sqrt(dt) * eps[:, 0]
        dWb = np.sqrt(dt) * eps[:, 1]
        #Updates price of Equtiy and Bonds using SDE
        eq_val *= np.exp(drift_s + sigma_s * dWs)
        bd_val *= np.exp(drift_b + sigma_b * dWb)
        #Rebalance at end of each year if chosen
        if rebalance and (t % steps_per_year == 0):
            total = eq_val + bd_val
            eq_val = w_equity * total
            bd_val = (1.0 - w_equity) * total
        #Calculate eq_val + bd_val at time step in USD and store
        port_usd[:, t] = eq_val + bd_val
        #Forex Path and store for each year - to convert to EUR each year
        zfx = rng.normal(size=n_paths)
        Y[:, t] = Y[:, t - 1] * np.exp(drift_fx + sigma_fx * np.sqrt(dt) * zfx)
    #Extract yearly values
    year_idx = [i * steps_per_year for i in range(0, years + 1)]
    port_usd_yr = port_usd[:, year_idx]
    Y_yr = Y[:, year_idx]
    #Convert to EUR
    port_eur_yr = port_usd_yr * Y_yr
    #Compute annualized EUR returns
    ann_rets_eur = port_eur_yr[:, 1:] / port_eur_yr[:, :-1] - 1.0
    mean_ann = float(np.mean(ann_rets_eur))
    vol_ann = float(np.std(ann_rets_eur, ddof=1))
    #Total return over 10y
    terminal_values = port_eur_yr[:, -1]
    tot_ret = float(np.mean(terminal_values / port_eur_yr[:, 0] - 1.0))
    tot_vol = float(np.std(terminal_values / port_eur_yr[:, 0] - 1.0, ddof=1))
    #Terminal Confidence interval in EUR
    term_ci = (
        float(np.percentile(terminal_values, 5)),
        float(np.percentile(terminal_values, 95)),
    )
    #Sharpe ratio
    sharpe = (mean_ann - rf_de) / vol_ann if vol_ann > 0 else float("nan")
    usd_growth_index = port_usd_yr / port_usd_yr[:, [0]]
    return {
        "annual_return_mean_eur": mean_ann,
        "annual_vol_eur": vol_ann,
        "sharpe_vs_rf_de": sharpe,
        "total_return_10y_eur_mean": tot_ret,
        "total_return_10y_eur_vol": tot_vol,
        "terminal_value_CI_5_95_eur": term_ci,
        "rf_de": rf_de,
        "mortgage_de": rf_de + 0.015,
        "portfolio_eur_paths": port_eur_yr,
        "portfolio_usd_paths": port_usd_yr,
        "usd_growth_index_paths": usd_growth_index,
        "fx_eur_per_usd_paths": Y_yr,
    }

#This function simulates the NAV path alongside equity
#If cashflows affect investments it interacts, uses traced growth values so its independent
#Uses FX path to convert to USD and back to trace NAVs
#Uses Cashflows to affect compounding
#Returns adjusted paths and also NAVs/avgs
def simulate_with_cashflows(
    nav0_eur_total,
    fx0, w_equity, mu_s, sigma_s, mu_b, sigma_b, rho, ecb, fed, sigma_fx,
    initial_shock = 0.00,
    years=10, steps_per_year=12, n_paths=10_000,
    seed=42,
    rebalance = False,
    cashflows_affect_investments=True
):
    #Account for Shock (needed for part 3-4)
    portfolio_initial_nav = 2*nav0_eur_total/3 - initial_shock*nav0_eur_total
    base = simulate(
        nav0_eur= portfolio_initial_nav,
        fx0=fx0, w_equity=w_equity,
        mu_s=mu_s, sigma_s=sigma_s, mu_b=mu_b, sigma_b=sigma_b,
        rho=rho, ecb=ecb, fed=fed, sigma_fx=sigma_fx,
        years=years, steps_per_year=steps_per_year, n_paths=n_paths,
        rebalance=rebalance, seed=seed
    )
    #Get the base growth paths and forex rates
    port_usd_raw = base["portfolio_usd_paths"]           #shape (n_paths, years+1)
    fx_eur_per_usd = base["fx_eur_per_usd_paths"]        #same shape
    #Build EUR cashflow series (length=years)
    bakery_eur = bakery_cashflows(nav0_eur_total, n_paths=n_paths, years=years)
    apt_eur = apartment_cashflows(nav0_eur_total)[:years]
    #We'll compute donation using the *total* EUR NAV at prior year-end
    #Year 0 total NAV is nav0_eur_total (portfolio + property already embedded in split)
    #We construct adjusted NAV path iteratively to feed donations
    years_plus_1 = port_usd_raw.shape[1]
    #Adjusted USD path that includes cashflows and compounds with raw growth rates
    adj_usd = np.zeros_like(port_usd_raw)
    adj_usd[:, 0] = port_usd_raw[:, 0]  # start
    #Also track adjusted EUR NAV (for donation rule bookkeeping and output)
    adj_eur = np.zeros_like(port_usd_raw)
    adj_eur[:, 0] = adj_usd[:, 0] * fx_eur_per_usd[:, 0] + nav0_eur_total/3 #add nav_0/3 cause intiial only does it on 2/3 NAV - shock
    #Keep EUR cashflow components per year, broadcast to paths on apply
    bakery_eur_arr = np.array(bakery_eur)
    apt_eur_arr = np.array(apt_eur)                # (years,)
    don_eur_arr = np.zeros(years)                  # to fill based on adj_eur
    cash_usd = np.zeros_like(adj_usd)
    #Time Steps modify all paths at once
    for t in range(1, years_plus_1):
        #Yearly Portfolio USD Growth Rate over Steps
        g = port_usd_raw[:, t] / port_usd_raw[:, t - 1]
        #Donation (EUR) yearly based on previous year over Steps
        don_eur = -0.025 * nav0_eur_total - 0.025 * adj_eur[:, t - 1]
        don_eur_arr[t - 1] = np.mean(don_eur)
        #Total cashflow this year (EUR) = bakery cashflow + apt cash flow + donation
        cf_eur_vec = (bakery_eur[:, t - 1] + apt_eur_arr[t - 1]) + don_eur
        #Convert to USD at year-end FX
        cf_usd_vec = cf_eur_vec / fx_eur_per_usd[:, t]
        if cashflows_affect_investments:
            #Flows reinvested and thus impact compounding
            adj_usd[:, t] = adj_usd[:, t - 1] * g + cf_usd_vec
            cash_usd[:, t] = 0.0  # not tracked separately
        else:
            #Flows not reinvested and thus portfolio evolves independently we treat Negative/Positive Flows as Phantoms
            adj_usd[:, t] = adj_usd[:, t - 1] * g  #just the invested portion grows
            cash_usd[:, t] = cash_usd[:, t - 1] + cf_usd_vec  # accumulate cash separately
        #Combined NAV (investments + accumulated cash)
        nav_total_usd = adj_usd[:, t] + cash_usd[:, t]
        #Translate to EUR -> adj eur reupdated for next year's donation -> Adj_eur is the one tracking NAV!!!
        adj_eur[:, t] = nav_total_usd * fx_eur_per_usd[:, t]  + nav0_eur_total/3

    #Return everything useful
    out = dict(base)  #keep original stats/paths
    out.update({
        #Adjusted paths (what you use for "actual NAV" after flows)
        "nav_eur_paths_adjusted": adj_eur, #per-year EUR NAV incl. cashflows
        #Cashflow components (EUR, per year)
        "cashflows_bakery_eur_per_year": bakery_eur_arr.mean(axis=0).tolist(),
        "cashflows_apartment_eur_per_year": apt_eur_arr.tolist(),
        "cashflows_donation_mean_eur_per_year": don_eur_arr.tolist(),
        #Raw USD growth rates per path
        "portfolio_usd_growth_factors": (port_usd_raw[:, 1:] / port_usd_raw[:, :-1]),
    })
    return out

#Initial Simulation for Part 2
res  = simulate_with_cashflows(
    nav0_eur_total = nav0_eur,
    fx0 = fx0, w_equity = w_equity, mu_s = mu_s, sigma_s = sigma_s,
    mu_b = mu_b, sigma_b = sigma_b, rho = rho, ecb = ecb, fed = fed, sigma_fx = sigma_fx,
    years=years, steps_per_year=steps_per_year, n_paths=n_paths,
    initial_shock=0.00,
    seed=42,
    rebalance = rebalance,
    cashflows_affect_investments= cashflows_affect_investments
)

st.title("10-year Portfolio Model (EUR base)")
st.markdown(f"""
- Portfolio Paths and FX Rates are Tracked utilising a Stochastic Differential Equation with Euler-Naruyama steps - Wiener process assumes independence of increment steps. Stock-Bond correlations utilise Cholesky method to generate correlated random variables at each time step. 
- Mean annual volatiltiy and return expectations are used as inputs to model the SDE. ECB Rate affects German Risk Free Return rate (10y Yield) and with the US Federal Reserve rate, controls drift in Forex (in this model). 
- The SDE used in this model follows the general form outlined below:
""")
st.latex(r"""
dS_t = \mu S_t dt + \sigma S_t dW_t
""")
st.markdown("""
- Here S_t represents the stock price, mu the expectation and sigma the drift.
""")
st.subheader("Portfolio Modeling: The Bond Component")

st.markdown("""
In this model, the value of the aggregated **bond portfolio** ($B_t$) is also assumed to follow a **Geometric Brownian Motion (GBM)**. This treats the bond portfolio's total value as an asset whose returns are log-normally distributed, driven by your input $\mu_b$ and $\sigma_b$.
The bond portfolio's value, $B_t$, is governed by the SDE:
""")
st.latex(r"""
\frac{dB_t}{B_t} = \mu_b dt + \sigma_b dW_{t,b}
""")
st.markdown("""
### Correlated Shocks
To ensure a realistic simulation, the random shocks driving the stock ($dW_{t,s}$) and bond ($dW_{t,b}$) values are correlated using your input **Stock-Bond Correlation ($\rho$)**.
* We construct a correlation matrix $\mathbf{\Sigma}$:
""")
st.latex(r"""
\mathbf{\Sigma} = \begin{pmatrix} 1 & \rho \\ \rho & 1 \end{pmatrix}
""")
st.markdown("""
* The **Cholesky decomposition** is applied ($\mathbf{\Sigma} = \mathbf{L}\mathbf{L}^T$) to transform independent standard normal random variables into the correlated shocks ($\epsilon_s, \epsilon_b$):
""")
st.latex(r"""
\begin{pmatrix} \epsilon_s \\ \epsilon_b \end{pmatrix} = \mathbf{L} \begin{pmatrix} Z_1 \\ Z_2 \end{pmatrix}
""")

#----- PART 1: PORTFOLIO METRICS -------
st.header("Part 1: Portfolio Analytics")
st.markdown("""
In the Base Model, we assume Stock Price grows at 8.0%, with volatility 16.8%, annualised bond growth of 4.40% and volatility of 4.80%. These are from leading Analyst's current market forecasts with a 10 year outlook in USD weighted alongside past performances of Treasuries and MSCI World ETF. To account for currency deviations and foreign exchange rates, we assume a beginning foreign exchange rate of 1.15 USD per EUR and assume appreciation of USD when the federal funds rate is higher than the ECB deposit rate with yearly volatility of 5%. Adjusting the Bow for the Bakery's realisation schedule increases volatility while increasing expected returns given a 20% IRR. 
""")
c1,c2,c3 = st.columns(3)
c1.metric("Annual Return (EUR)", f"{res['annual_return_mean_eur']*100:.2f}%")
c2.metric("Annual Volatility (EUR)", f"{res['annual_vol_eur']*100:.2f}%")
c3.metric("Sharpe vs RF(DE)", f"{res['sharpe_vs_rf_de']:.2f}")

st.subheader("10-Year Results (EUR)")
st.write(f"Average total return over 10y: **{res['total_return_10y_eur_mean']*100:.2f}%**")
st.write(f"Total return stdev (paths): **{res['total_return_10y_eur_vol']*100:.2f}%**")
st.write(f"Terminal EUR value CI (5–95%): ${res['terminal_value_CI_5_95_eur'][0]:,.0f} – ${res['terminal_value_CI_5_95_eur'][1]:,.0f}")

st.subheader("Rates (derived)")
st.write(f"German risk-free (approx): **{rf_de*100:.2f}%**")
st.write(f"German mortgage rate (approx): **{mortgage_de*100:.2f}%**")

st.subheader("Equity/Bonds Growth Paths")
mean_growth_df = pd.DataFrame({
    "Mean Portfolio Growth" : np.mean(res["portfolio_eur_paths"], axis=0),
    "Lower 5% Portfolio Growth": np.percentile(res["portfolio_eur_paths"], 5, axis = 0),
    "Highest 95% Portfolio Growth": np.percentile(res["portfolio_eur_paths"], 95, axis=0),
})
mean_growth_df.index = np.arange(0, 11)
years_ofrandom = np.arange(0, 11)
colA, colB = st.columns(2)
with colA:
    st.subheader("EUR Paths CI")
    st.line_chart(mean_growth_df)
with colB:
    st.subheader("Random EUR Paths")
    sample_idx = np.random.choice(res["portfolio_eur_paths"].shape[0], size=25, replace=False)
    sample_df = pd.DataFrame(res["portfolio_eur_paths"][sample_idx].T, index=years_ofrandom)
    st.line_chart(sample_df, use_container_width=True)
#----- PART 2: NAV Metrics -------
st.header("Part 2: Cash Flow Modeling")
st.header("Cashflow Schedules (EUR per year)")
cf_df = pd.DataFrame({
    "Bakery": res["cashflows_bakery_eur_per_year"],
    "Apartment": res["cashflows_apartment_eur_per_year"],
    "Donations": res["cashflows_donation_mean_eur_per_year"],
})
cf_df.index = np.arange(1, len(cf_df) + 1)
st.bar_chart(cf_df)

# Extract the adjusted EUR NAV paths
nav_paths_eur = res["nav_eur_paths_adjusted"]

# Compute key summary percentiles across paths per year
median_nav = np.percentile(nav_paths_eur, 50, axis=0)
p5_nav = np.percentile(nav_paths_eur, 5, axis=0)
p95_nav = np.percentile(nav_paths_eur, 95, axis=0)
mean_nav = np.mean(nav_paths_eur, axis=0)

# Build DataFrame for Streamlit chart
years_axis = np.arange(0, nav_paths_eur.shape[1])
df_nav = pd.DataFrame({
    "Median NAV": median_nav,
    "Expected NAV": mean_nav,
    "5th %ile": p5_nav,
    "95th %ile": p95_nav
}, index=years_axis)

st.header("NAV Evolution (EUR)")
st.line_chart(df_nav)
st.caption("Median and 5–95% percentile range of total EUR NAV across Monte Carlo paths.")

final_navs = nav_paths_eur[:, -1]
prob_up = np.mean(final_navs >= nav0_eur)
mean_final = np.mean(final_navs)
median_final = np.median(final_navs)
p5_final = np.percentile(final_navs, 5)
p95_final = np.percentile(final_navs, 95)
exp_growth = (mean_final / nav0_eur - 1) * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("Expected 10Y NAV", f"€{mean_final:,.0f}")
c2.metric("Expected Growth", f"{exp_growth:.2f}%")
c3.metric("Median Final NAV", f"€{median_final:,.0f}")
c4.metric("Prob(NAV ≥ Today)", f"{prob_up*100:.2f}%")

st.write(f"5th–95th percentile range: **€{p5_final:,.0f} – €{p95_final:,.0f}**")

st.caption("We use Euler–Maruyama to discretize the SDEs and Monte Carlo to run many random paths. Percentiles (5th/95th) are from np.percentile across paths.")


st.subheader("Expected Component Returns Summary (10y)")
total_cf = np.sum(cf_df, axis=0)
summary_df = pd.DataFrame({
    "Source": total_cf.index,
    "Total EUR": total_cf.values
})
st.dataframe(summary_df.style.format({"Total EUR": "€{:,.0f}"}))


#---Risk Management: find maximum tolerable shock ---
st.header("Part 3: Risk Management")

target_prob = st.slider("Select Target Probabilty for Shock: ", 0.00, 100.00, 95.00, 0.1)/100
#Just use a binary search
def find_max_shock(target_prob=0.95, tol=0.001, max_iter=20):
    lo = 0.0
    hi = 2/3
    results = []   #store(shock, probability)
    progress = st.progress(0)
    for i in range(max_iter):
        mid = 0.5 * (lo + hi)
        sim = simulate_with_cashflows(
            nav0_eur_total=nav0_eur,
            fx0=fx0, w_equity=w_equity,
            mu_s=mu_s, sigma_s=sigma_s, mu_b=mu_b, sigma_b=sigma_b,
            rho=rho, ecb=ecb, fed=fed, sigma_fx=sigma_fx,
            years=years, steps_per_year=steps_per_year,
            n_paths=n_paths, rebalance=rebalance,
            initial_shock=mid,
            cashflows_affect_investments=cashflows_affect_investments
        )
        final_navs = sim["nav_eur_paths_adjusted"][:, -1]
        prob = np.mean(final_navs >= nav0_eur)
        results.append((mid, prob))
        # Binary search update
        if prob >= target_prob:
            lo = mid  #can withstand more shock
        else:
            hi = mid  #too much shock
        progress.progress((i + 1) / max_iter)
    progress.empty()
    df_results = pd.DataFrame(results, columns=["Shock", "Probability"])
    return lo, df_results

max_shock, shock_results = find_max_shock(target_prob=target_prob)
st.metric(f"Maximum tolerable shock (≥ {target_prob*100}% prob NAV ≥ today)", f"{max_shock*100:.2f}%")
#Line chart of binary search convergence
shock_results["Shock %"] = shock_results["Shock"] * 100
shock_results.set_index("Shock %", inplace=True)
st.line_chart(shock_results["Probability"])
st.caption("Binary search over initial shocks. Each point represents one simulation; the curve shows the probability of ending with NAV ≥ today's NAV.")

#--------- PART 4: Portfolio Optimisation ------------#
st.header("Part 4: Portfolio Optimisation")
st.write(f"From Part 3: maximum tolerable shock = **{max_shock*100:.2f}%**, target probability = **{target_prob*100:.1f}%**.")
alloc_grid = np.linspace(0.0, 1.0, 100)
results = []
progress = st.progress(0)
for i, x_eq in enumerate(alloc_grid):
    sim = simulate_with_cashflows(
        nav0_eur_total=nav0_eur,
        fx0=fx0, w_equity=x_eq,
        mu_s=mu_s, sigma_s=sigma_s,
        mu_b=mu_b, sigma_b=sigma_b,
        rho=rho, ecb=ecb, fed=fed, sigma_fx=sigma_fx,
        years=years, steps_per_year=steps_per_year,
        n_paths=int(n_paths/3),
        rebalance=rebalance,
        initial_shock = max_shock,
        cashflows_affect_investments=cashflows_affect_investments
    )
    final_navs = sim["nav_eur_paths_adjusted"][:, -1]
    prob = np.mean(final_navs >= nav0_eur)
    mean_ret = np.mean(final_navs / nav0_eur - 1)
    vol_ret = np.std(final_navs / nav0_eur - 1, ddof=1)
    results.append((x_eq, prob, mean_ret, vol_ret))
    progress.progress((i + 1) / len(alloc_grid))
progress.empty()
#Convert to DataFrame
opt_df = pd.DataFrame(results, columns=["Equity Weight", "Prob_UP", "Mean_Return", "Volatility"])
opt_df["Equity Weight (%)"] = opt_df["Equity Weight"] * 100
opt_df["Prob(NAV ≥ Today) (%)"] = opt_df["Prob_UP"] * 100
opt_df["Mean 10Y Return (%)"] = opt_df["Mean_Return"] * 100
opt_df["Volatility (%)"] = opt_df["Volatility"] * 100
#Display summary
st.subheader("Allocation Results (10-Year Horizon)")
st.dataframe(
    opt_df[["Equity Weight (%)", "Prob(NAV ≥ Today) (%)", "Mean 10Y Return (%)", "Volatility (%)"]]
    .style.format({
        "Prob(NAV ≥ Today) (%)": "{:.2f}%",
        "Mean 10Y Return (%)": "{:.2f}%",
        "Volatility (%)": "{:.2f}%"
    })
)

#Plot probability curve
st.subheader("Probability of Ending Above Today's NAV vs Equity Allocation")
st.line_chart(opt_df.set_index("Equity Weight (%)")["Prob(NAV ≥ Today) (%)"])

#Find optimal allocation
best_row = opt_df.loc[opt_df["Prob_UP"].idxmax()]
best_alloc = best_row["Equity Weight"] * 100
best_prob = best_row["Prob_UP"] * 100

st.write(
    f"Optimal allocation to maximize 10-year success probability: "
    f"**{best_alloc:.1f}% equities / {100 - best_alloc:.1f}% bonds** "
    f"(Prob(NAV ≥ today) = {best_prob:.2f}%)"
)

st.markdown(f"""
- From Part 3: tolerable shock = **{max_shock*100:.2f}%**
- From Part 4: optimal equity allocation = **{best_alloc:.1f}%**
- These results together suggest that, given your drawdown tolerance,
  you should aim for roughly **{best_alloc:.0f}% equities** to maximize the chance
  of ending 10 years above today's NAV.
""")

st.caption("""
This coarse grid method runs independent Monte Carlo simulations for allocations from 0–100% equities.It’s slower than analytic results, but more robust under random noise.
""")
