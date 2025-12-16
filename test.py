if __name__ == "__main__":
    from deep_bf.data_handler import DataLoader, rf2iq

    dl = DataLoader("../../../rf_data")
    # print(dl.df.head())

    test = dl.get_defined_pwdata("MYO001", "RF")
    # print(test.data.shape)
    # print(test.fc, test.fs)

    fs = test.fs
    fc = test.fc

    rf = test.data
    
    iq = rf2iq(rf, fs, fc)
    print(iq.shape)

