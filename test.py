import quandl

quandl.ApiConfig.api_key = '51rd-rru9cH8Dt2ZBroe'

try:
    data = quandl.get_table('WIKI/PRICES', ticker=['TSLA'])
    print(data)

except quandl.errors.quandl_error.NotFoundError:
    print("The requested dataset or table could not be found.")

except quandl.errors.quandl_error.ForbiddenError:
    print("Authentication failed or you do not have permissions to access this data.")

except Exception as e:
    print("An error occurred: ", str(e))

