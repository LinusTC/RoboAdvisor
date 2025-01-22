from fetchData import fetch_data, fetch_data1
# Example list of assets
assets = ["AAPL", "GOOGL", "MSFT", "AMZN", "NFLX"]

# Set the portfolio size and maximum number of iterations
portfolio_size = 5
max_iters = 10

# Fetch the data
raw_data, simulations, errors = fetch_data(assets,"51rd-rru9cH8Dt2ZBroe", portfolio_size, max_iters)

# Check if raw data was successfully fetched and then print it
if raw_data is not None:
    print (simulations)
else:
    print("No valid data available.")

# Optionally, also print out errors if there were any during data fetching
if errors:
    print("Errors encountered with the following assets:", errors)
