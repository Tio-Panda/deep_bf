import h5py
import ast
from deep_bf.constants.bf import PWDataType
import pandas as pd
from pathlib import Path
import numpy as np
from scipy.signal import hilbert, convolve
import importlib.resources

from .data_classes import PWData, RFData, IQSplitData
# from data_handlers.pwData import PWData, IQData, RFData

class DataLoader():
    def __init__(self, data_path:str):
        """
        Load ultrasound PW dataset from PICMUS, CUBDL or custom data as PWData objects for reconstruction.

        Args:
            data_path (str): Filesystem path to the directory with PICMUS or CUBDL HDF5 files.
            df_path (str): Filesystem path to dataset csv.
        """
        self.path = Path(data_path)
        
        with importlib.resources.files("deep_bf.data_handler.data").joinpath("data.csv").open("r") as f:
            self.df = pd.read_csv(f)
            self.df["path"] = None

        # === Mapping the names of datasets with their filesystem path in the df ===
        mapping: dict[str, str] = {}
        dataset_names = set(self.df["name"])

        for f in self.path.rglob('*.hdf5'):
            stem = f.stem
            if stem in dataset_names:
                mapping[stem] = str(f)

        self.df["path"] = self.df["name"].map(mapping.get)


    def get_defined_pwdata(self, selected_name: str, mode: str):
        df = self.df.query("name == @selected_name").iloc[0]

        name = df["name"]
        source = df["source"]

        params = {}
        params["name"] = df["name"]
        params["source"] = df["source"]

        # params["na"] = df["na"]
        # params["nc"] = df["nc"]
        # params["ns"] = df["ns"]

        params["pitch"] = df["pitch"] / 1000 # [mm] -> [m]
        params["angles_range"] = np.array(ast.literal_eval(df["angles_range"]))
        params["zlims"] = np.array(ast.literal_eval(df["zlims"]))
        params["fc"] = df["fc"] * 1e6 # [MHz] -> [Hz]
        
        path = Path(df["path"])
        with h5py.File(path, mode="r") as f:
            if source == "PICMUS":
                f = f["US"]["US_DATASET0000"]

            #for key in list(f.keys()):
                #print(f[key])


            aperture_width = params["pitch"] * df["nc"]
            rad_range = np.radians(params["angles_range"])
            angles = np.linspace(rad_range[0], rad_range[1], num=df["na"], dtype=np.float32)

            params["c0"] = np.array(f["sound_speed"]).item()
            # params["fc"] = np.array(f["modulation_frequency"]).item()
    
            # Frequency
            if name[0:3] == "UFL":
                params["fs"] = np.array(f["channel_data_sampling_frequency"]).item()
                params["fdemod"] = np.array(f["modulation_frequency"]).item()
            else:
                params["fs"] = np.array(f["sampling_frequency"]).item()     
                params["fdemod"] = 0
            
            # T0
            if name[0:3] in ("TSH", "MYO", "EUT", "INS"):
                params["t0"] = ((aperture_width/2) / params["c0"]) * np.abs(np.sin(angles))
                if name[0:3] == "EUT":
                    params["t0"] += 10 / params["fs"]

            elif name[0:3]  == "UFL":
                params["t0"] = -1 * np.array(f["channel_data_t0"], dtype=np.float32)
                if params["t0"].size == 1:
                    params["t0"] = np.ones_like(angles) * params["t0"]

            elif name[0:3] == "OSL":
                params["t0"] = -1 * np.array(f["start_time"], dtype=np.float32)
                params["t0"] = np.transpose(params["t0"])
            
            elif name[0:3] == "JHU":
                params["t0"] = np.array(f["time_zero"], dtype=np.float32)
                if name[0:3] == "JHU":
                    params["t0"] -= 10 / params["fdemod"]

            else:
                params["t0"] = np.zeros(df["na"])

            # Data

            if source == "CUBDL":
                data = np.array(f["channel_data"], dtype="float32")
            else:
                data = np.array(f["data"]["real"], dtype="float32")

            if name[0:3] == "TSH":
                data = np.reshape(data, (128, df["na"], -1))
                data = np.transpose(data, (1, 0, 2))

            if mode == PWDataType.IQ_SPLIT:
                idata = data
                qdata = np.imag(hilbert(idata, axis=-1))

                # if name[0:3] == "UFL":
                #     _data = idata + 1j * qdata
                #     phase = np.reshape(np.arange(idata.shape[2], dtype="float"), (1, 1, -1))
                #     phase *= params["fdemod"] / params["fs"]
                #     _data *= np.exp(-2j * np.pi * phase)
                #     dsfactor = int(np.floor(params["fs"] / params["fc"]))
                #     kernel = np.ones((1, 1, dsfactor), dtype="float") / dsfactor
                #     _data = convolve(_data, kernel, "same")
                #     _data = _data[:, :, ::dsfactor]
                #     params["fs"] /= dsfactor
                #
                #     idata = np.real(_data)
                #     qdata = np.imag(_data)

                params["data"] = np.stack([idata, qdata], axis=-1)
            else:
                params["data"] = data

            # TODO: implementar IQ Demodulada

            params["na"], params["nc"], params["ns"] = data.shape

            if mode == PWDataType.RF:
                pw = RFData(**params)
            else:
                pw = IQSplitData(**params)

            # TODO: Agrear la version IQComplex

        return pw
        
    #def get_custom_pwdata(self, y todos los parametros necesarios):

    #TODO: Funciones auxiliares para obtener por source y otros metodos.

    def get_df(self) -> pd.DataFrame:
        return self.df
