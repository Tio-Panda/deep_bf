from deep_bf.data_handler import DataLoader

dl = DataLoader("../../../rf_data")
print(dl.df.head())

test = dl.get_defined_pwdata("MYO001", "RF")
print(test.data.shapes)
