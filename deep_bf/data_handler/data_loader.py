import h5py
import ast
import pandas as pd
from pathlib import Path
import numpy as np
from scipy.signal import hilbert
import importlib.resources

from .data_classes import PWData, RFData, IQData
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


    def get_defined_pwdata(self, selected_name: str, mode: str) -> RFData|IQData:
        df = self.df.query("name == @selected_name").iloc[0]

        name = df["name"]
        source = df["source"]

        params = {}
        params["name"] = df["name"]
        params["source"] = df["source"]
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

            params["c0"] = np.array(f["sound_speed"]).item()
            params["fdemod"] = np.array(f["modulation_frequency"]).item()

            if name[0:3] == "UFL":
                params["fs"] = np.array(f["channel_data_sampling_frequency"]).item()
            else:
                params["fs"] = np.array(f["sampling_frequency"]).item()     

            aperture_width = params["pitch"] * df["nc"]
            rad_range = np.radians(params["angles_range"])
            angles = np.linspace(rad_range[0], rad_range[1], num=df["na"], dtype=np.float32)

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
            
            
            if source == "CUBDL":
                data = np.array(f["channel_data"], dtype="float32")
            elif source == "PICMUS":
                data = np.array(f["data"]["real"], dtype="float32")
            else:
                raise ValueError("Source options are: 'CUBDL' and 'PICMUS'")
            
            if name[0:3] == "TSH":
                data = np.reshape(data, (128, df["na"], -1))
                data = np.transpose(data, (1, 0, 2))

            if mode == "RF":
                params["rfdata"] = data
            else:
                if source == "CUBDL":
                    params["iqdata"] = np.stack((data, np.imag(hilbert(data, axis=-1))), axis=-1)
                    params["fdemod"] = 0
                elif source == "PICMUS":
                    params["iqdata"] = np.stack((data, np.array(f["data"]["imag"])), axis=-1)
                else:
                    raise ValueError("Source options are: 'CUBDL' and 'PICMUS'")

            params["n_angles"], params["n_channels"], params["n_samples"] = data.shape

            if mode == "RF":
                pw = RFData(**params)
            else:
                pw = IQData(**params)

        return pw
        
    #def get_custom_pwdata(self, y todos los parametros necesarios):

    #TODO: Funciones auxiliares para obtener por source y otros metodos.

    def get_df(self) -> pd.DataFrame:
        return self.df

    def get_defined_pwdata2(self, selected_name: str, mode: str) -> RFData:
        # 1. Obtener metadata
        df_row = self.df.query("name == @selected_name").iloc[0]
        source = df_row["source"]
        acq_num = int(''.join(filter(str.isdigit, selected_name)))
        
        path = Path(df_row["path"])
        
        # Parámetros base
        params = {
            "name": selected_name,
            "source": source,
            "pitch": df_row["pitch"] / 1000,
            "angles_range": np.array(ast.literal_eval(df_row["angles_range"])),
            "zlims": np.array(ast.literal_eval(df_row["zlims"]))
        }
        
        with h5py.File(path, mode="r") as f_root:
            # Manejo de estructura interna
            f = f_root["US"]["US_DATASET0000"] if source == "PICMUS" else f_root
            source = selected_name[0:3]

            # --- A. Velocidad del Sonido (c0) Calibrada ---
            # Mantenemos esto igual, es física, no formato.
            c0 = 1540.0
            if source == "MYO":
                c_map = {1: 1580, 2: 1583, 3: 1578, 4: 1572, 5: 1562}
                c0 = c_map.get(acq_num, 1581)
            elif source == "UFL":
                c_map = {1: 1526, 2: 1523, 4: 1523, 5: 1523}
                c0 = c_map.get(acq_num, 1525)
            elif source == "EUT":
                c_map = {1: 1603, 2: 1618, 3: 1607, 4: 1614, 5: 1495}
                c0 = c_map.get(acq_num, 1479)
            elif source == "INS":
                # Lógica simplificada INS (expandir si es necesario todos los casos)
                c0 = 1540 # Default general, ajustar según tabla si se requiere precisión extrema
            elif source == "OSL":
                 c_map = {2:1536, 3:1543, 4:1538, 5:1539, 6:1541, 7:1540}
                 c0 = c_map.get(acq_num, 1540)
            else:
                c0 = np.array(f["sound_speed"]).item()
            
            params["c0"] = float(c0)

            # --- B. Carga de RF Cruda ---
            # UFL tiene nombres de keys distintos
            if source == "UFL":
                fs = np.array(f["channel_data_sampling_frequency"]).item()
                # NOTA: En RF puro, fc es informativa, no se usa para demodular
                fc = np.array(f["modulation_frequency"]).item() 
                raw_data = np.array(f["channel_data"], dtype="float32")
                
                # Ángulos en UFL vienen en grados
                if "angles" in f:
                    angles = np.array(f["angles"]) * np.pi / 180
                else:
                    rad_range = np.radians(params["angles_range"])
                    angles = np.linspace(rad_range[0], rad_range[1], num=df_row["n_angles"])

            elif source == "PICMUS":
                fs = np.array(f["sampling_frequency"]).item()
                fc = np.array(f["modulation_frequency"]).item()
                raw_data = np.array(f["data"]["real"], dtype="float32")
                angles = np.linspace(np.radians(params["angles_range"][0]), 
                                     np.radians(params["angles_range"][1]), 
                                     num=df_row["n_angles"])
            else:
                # Estándar CUBDL (MYO, EUT, INS, OSL, JHU, TSH)
                fs = np.array(f["sampling_frequency"]).item()
                fc = np.array(f["modulation_frequency"]).item()
                raw_data = np.array(f["channel_data"], dtype="float32")
                
                if "angles" in f:
                    angles = np.array(f["angles"])
                elif "transmit_direction" in f:
                    angles = np.array(f["transmit_direction"])[:, 0]
                else:
                    rad_range = np.radians(params["angles_range"])
                    angles = np.linspace(rad_range[0], rad_range[1], num=df_row["n_angles"])

            # Ajuste de dimensiones específico para TSH
            if source == "TSH":
                 raw_data = np.reshape(raw_data, (128, len(angles), -1))
                 raw_data = np.transpose(raw_data, (1, 0, 2))

            params["fs"] = float(fs)
            params["fc"] = float(fc)
            params["rfdata"] = raw_data
            params["fdemod"] = 0 # Siempre 0 en RF puro

            # --- C. Corrección de Tiempo Cero (t0) ---
            n_channels = raw_data.shape[1]
            xpos = np.arange(n_channels) * params["pitch"]
            xpos -= np.mean(xpos)
            
            t0 = np.zeros(len(angles), dtype=np.float32)

            if source in ["MYO", "TSH"]:
                for i, a in enumerate(angles):
                    t0[i] = xpos[-1] * np.abs(np.sin(a)) / c0
            
            elif source == "UFL":
                t0_file = -1 * np.array(f["channel_data_t0"], dtype="float32")
                t0 = np.ones_like(angles) * t0_file if t0_file.size == 1 else t0_file

            elif source == "EUT":
                t0_start = np.array(f["start_time"], dtype="float32")[0]
                t0[:] = t0_start
                for i, a in enumerate(angles):
                    t0[i] += xpos[-1] * np.abs(np.sin(a)) / c0
                t0 += 10 / fc 

            elif source == "INS":
                n_angles = raw_data.shape[0]
                angles = np.linspace(-16, 16, n_angles) * np.pi / 180
                t0 = -1 * np.array(f["start_time"], dtype="float32")[0]
                for i, a in enumerate(angles):
                    t0[i] += xpos[-1] * np.abs(np.sin(a)) / c0

            elif source == "OSL":
                t0 = -1 * np.array(f["start_time"], dtype="float32")[0]
            
            elif source == "JHU":
                t0 = -1 * np.array(f["time_zero"], dtype=np.float32)
                c0 = np.array(f["sound_speed"]).item()
                params["c0"] = float(c0)

            params["t0"] = t0
            
            # Dimensiones finales
            params["n_angles"], params["n_channels"], params["n_samples"] = raw_data.shape

            return RFData(**params)
