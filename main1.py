import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")



ticker = input("Enter ticker symbol: ")
end = dt.date.today()
start = dt.date.today() - dt.timedelta(days=365 * 5)

# Download data
data = yf.download(ticker, start=start, end=end)
print(f"Downloaded data for {ticker}")
print(f"Data shape: {data.shape}")
print(data.head())

# Extract close prices
closeDf = pd.DataFrame()
closeDf['Close'] = data['Close']
closeDf['Date'] = data.index

# Parameters for moving averages
sma_period = 20  # Short-term SMA
ema_period = 50  # Long-term EMA

# Calculate SMA and EMA
closeDf['SMA'] = closeDf['Close'].rolling(window=sma_period).mean()
closeDf['EMA'] = closeDf['Close'].ewm(span=ema_period).mean()

# Generate trading signals
closeDf['Signal'] = 0
closeDf['Signal'][sma_period:] = np.where(
    closeDf['SMA'][sma_period:] > closeDf['EMA'][sma_period:], 1, 0
)

# Generate trading positions (1 for buy, -1 for sell, 0 for hold)
closeDf['Position'] = closeDf['Signal'].diff()

# Calculate returns
closeDf['Returns'] = closeDf['Close'].pct_change()
closeDf['Strategy_Returns'] = closeDf['Returns'] * closeDf['Signal'].shift(1)

# Calculate cumulative returns
closeDf['Cumulative_Returns'] = (1 + closeDf['Returns']).cumprod()
closeDf['Cumulative_Strategy_Returns'] = (1 + closeDf['Strategy_Returns']).cumprod()

# Remove NaN values
closeDf = closeDf.dropna()

print(f"\nBacktest Results for {ticker}")
print("=" * 50)


def calculate_metrics(returns):
    total_return = (returns.iloc[-1] - 1) * 100
    annual_return = ((returns.iloc[-1]) ** (252 / len(returns)) - 1) * 100
    volatility = returns.pct_change().std() * np.sqrt(252) * 100
    sharpe_ratio = (annual_return - 2) / volatility if volatility != 0 else 0  # Assuming 2% risk-free rate
    max_drawdown = ((returns / returns.expanding().max()) - 1).min() * 100

    return {
        'Total Return (%)': round(total_return, 2),
        'Annual Return (%)': round(annual_return, 2),
        'Volatility (%)': round(volatility, 2),
        'Sharpe Ratio': round(sharpe_ratio, 2),
        'Max Drawdown (%)': round(max_drawdown, 2)
    }


buy_hold_metrics = calculate_metrics(closeDf['Cumulative_Returns'])
strategy_metrics = calculate_metrics(closeDf['Cumulative_Strategy_Returns'])

print("\nBuy and Hold Strategy:")
for key, value in buy_hold_metrics.items():
    print(f"{key}: {value}")

print(f"\nSMA({sma_period}) - EMA({ema_period}) Crossover Strategy:")
for key, value in strategy_metrics.items():
    print(f"{key}: {value}")

# Trading Statistics
buy_signals = len(closeDf[closeDf['Position'] == 1])
sell_signals = len(closeDf[closeDf['Position'] == -1])
total_trades = buy_signals + sell_signals

print(f"\nTrading Statistics:")
print(f"Total Buy Signals: {buy_signals}")
print(f"Total Sell Signals: {sell_signals}")
print(f"Total Trades: {total_trades}")

# Calculate win rate
if total_trades > 0:
    winning_trades = len(closeDf[(closeDf['Position'] != 0) & (closeDf['Strategy_Returns'] > 0)])
    win_rate = (winning_trades / total_trades) * 100
    print(f"Win Rate: {win_rate:.2f}%")

# Plotting
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle(f'{ticker} - SMA-EMA Crossover Strategy Analysis', fontsize=16)

# Plot 1: Price with moving averages and signals
ax1.plot(closeDf.index, closeDf['Close'], label='Close Price', linewidth=1, alpha=0.8)
ax1.plot(closeDf.index, closeDf['SMA'], label=f'SMA({sma_period})', linewidth=1.5, alpha=0.8)
ax1.plot(closeDf.index, closeDf['EMA'], label=f'EMA({ema_period})', linewidth=1.5, alpha=0.8)

# Add buy/sell signals
buy_signals_data = closeDf[closeDf['Position'] == 1]
sell_signals_data = closeDf[closeDf['Position'] == -1]

ax1.scatter(buy_signals_data.index, buy_signals_data['Close'],
            color='green', marker='^', s=100, label='Buy Signal', alpha=0.8)
ax1.scatter(sell_signals_data.index, sell_signals_data['Close'],
            color='red', marker='v', s=100, label='Sell Signal', alpha=0.8)

ax1.set_title('Price Chart with Moving Averages and Signals')
ax1.set_ylabel('Price')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Cumulative returns comparison
ax2.plot(closeDf.index, closeDf['Cumulative_Returns'],
         label='Buy & Hold', linewidth=2, alpha=0.8)
ax2.plot(closeDf.index, closeDf['Cumulative_Strategy_Returns'],
         label='SMA-EMA Strategy', linewidth=2, alpha=0.8)
ax2.set_title('Cumulative Returns Comparison')
ax2.set_ylabel('Cumulative Returns')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Daily returns distribution
ax3.hist(closeDf['Returns'].dropna(), bins=50, alpha=0.7, label='Buy & Hold', density=True)
ax3.hist(closeDf['Strategy_Returns'].dropna(), bins=50, alpha=0.7, label='Strategy', density=True)
ax3.set_title('Daily Returns Distribution')
ax3.set_xlabel('Returns')
ax3.set_ylabel('Density')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Rolling Sharpe ratio (252-day window)
rolling_window = 252
if len(closeDf) > rolling_window:
    rolling_sharpe_bh = (closeDf['Returns'].rolling(rolling_window).mean() * 252) / \
                        (closeDf['Returns'].rolling(rolling_window).std() * np.sqrt(252))
    rolling_sharpe_strategy = (closeDf['Strategy_Returns'].rolling(rolling_window).mean() * 252) / \
                              (closeDf['Strategy_Returns'].rolling(rolling_window).std() * np.sqrt(252))

    ax4.plot(closeDf.index, rolling_sharpe_bh, label='Buy & Hold', alpha=0.8)
    ax4.plot(closeDf.index, rolling_sharpe_strategy, label='Strategy', alpha=0.8)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.set_title(f'Rolling Sharpe Ratio ({rolling_window} days)')
    ax4.set_ylabel('Sharpe Ratio')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
else:
    ax4.text(0.5, 0.5, 'Insufficient data for\nrolling Sharpe ratio',
             transform=ax4.transAxes, ha='center', va='center')
    ax4.set_title('Rolling Sharpe Ratio')

plt.tight_layout()
plt.show()

# Create a summary table
summary_df = pd.DataFrame({
    'Buy & Hold': buy_hold_metrics,
    'SMA-EMA Strategy': strategy_metrics
})

print("\n" + "=" * 60)
print("PERFORMANCE COMPARISON TABLE")
print("=" * 60)
print(summary_df)

# Export results to CSV (optional)
results_df = closeDf[['Close', 'SMA', 'EMA', 'Signal', 'Position',
                      'Returns', 'Strategy_Returns', 'Cumulative_Returns',
                      'Cumulative_Strategy_Returns']].copy()


#results_df.to_csv(f'{ticker}_sma_ema_backtest_results.csv')

print(f"\nBacktest completed successfully!")
print(
    f"Strategy Performance: {strategy_metrics['Total Return (%)']}% vs Buy & Hold: {buy_hold_metrics['Total Return (%)']}%")