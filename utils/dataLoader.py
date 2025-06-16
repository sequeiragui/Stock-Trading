import yfinance as yf

class DataLoader:
    def __init__(self):
        self.qqq_data = yf.download("QQQ", start="2020-01-01", end="2024-12-31")
        print(self.qqq_data.head())

    def get_data(self):
        return self.qqq_data


dataload = DataLoader()
print(dataload.get_data().shape)  