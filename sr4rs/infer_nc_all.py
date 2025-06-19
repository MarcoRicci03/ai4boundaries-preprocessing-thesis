# infer.py
import os, warnings, logging, argparse, pathlib

import tqdm


# ---------- CLI prima di TensorFlow ----------
parser = argparse.ArgumentParser(description="SR4RS inference – salva tutti gli output")
parser.add_argument("--gpu", default="7", help="indice GPU o lista (es. '0' o '0,2')")
parser.add_argument("--model_dir", default='models/sr4rs_sentinel2_bands4328_france2020_savedmodel', help="Directory SavedModel")
parser.add_argument("--input_dir", default='../../datasets/AI4B/sentinel2/images', help="LR input directory")
parser.add_argument("--bands",     default="B2,B3,B4,B8", help="Virgola‑separata delle bande da usare (ordine = canali modello)")
parser.add_argument("--exclude",   default="NDVI,spatial_ref", help="Variabili da saltare (sempre escluse dal processamento)")
parser.add_argument("--output_path", default='../../datasets/AI4B_SR/sentinel2/images', help="Cartella di destinazione NC HR")
parser.add_argument("--scale", type=int, default=4, help="Fattore SR (default 4)")
parser.add_argument("--margin_lr", type=int, default=16, help="Pixel LR pad‑crop (default 16)")
args = parser.parse_args()

# ---------- env vars & silenziamento log ----------
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
try:
    from absl import logging as absl_log
    absl_log.set_verbosity(absl_log.ERROR)
    absl_log.set_stderrthreshold("error")
except ImportError:
    pass
warnings.filterwarnings("ignore", category=FutureWarning, module="tensorflow")
warnings.filterwarnings("ignore", category=UserWarning,   module="tensorflow")

# ---------- import TensorFlow & GDAL ----------
import numpy as np
import xarray as xr
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
print("GPU visibili a TensorFlow:", gpus)

def sr_inference(lr_arr: np.ndarray, model, scale: int, margin_lr: int) -> np.ndarray:
    """Applica il modello SR su una singola immagine (H,W,B).

    Ritorna (H*scale, W*scale, B) con crop centrale per compensare il
    valid‑padding del modello.
    """
    lr_pad = np.pad(lr_arr,
                    ((margin_lr, margin_lr), (margin_lr, margin_lr), (0, 0)),
                    mode="reflect")
    sr_pad = list(model(tf.constant(lr_pad[None], tf.float32)).values())[0].numpy()[0]

    exp_h, exp_w = lr_arr.shape[0] * scale, lr_arr.shape[1] * scale
    dh, dw = sr_pad.shape[0] - exp_h, sr_pad.shape[1] - exp_w
    if dh < 0 or dw < 0:
        raise ValueError("Output SR più piccolo dell'atteso – aumenta margin_lr")

    start_h, start_w = dh // 2, dw // 2
    return sr_pad[start_h:start_h + exp_h, start_w:start_w + exp_w, :]

# ---------- inferenza ----------
def main():
    input_path = pathlib.Path(args.input_dir)
    output_path = pathlib.Path(args.output_path)
    if not input_path.is_dir():
        raise ValueError(f"Input deve essere una cartella: {input_path}")
    if not output_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)

    nc_files = list(input_path.rglob("*.nc"))
    if not nc_files:
        raise ValueError(f"Nessun file .nc trovato in {input_path}")
    
    log.info("Trovati %d file .nc nella directory %s", len(nc_files), input_path)

    model = tf.saved_model.load(args.model_dir).signatures["serving_default"]
    

    for nc_file in tqdm.tqdm(nc_files, desc="Processing NC Files", unit="file"):
        # Apri il dataset NetCDF
        ds = xr.open_dataset(nc_file)

        bands   = [b.strip() for b in args.bands.split(",") if b.strip()]
        exclude = {e.strip() for e in args.exclude.split(",")}

        missing = [b for b in bands if b not in ds.data_vars]
        if missing:
            log.warning("Bande mancanti in %s: %s. Saltato.", nc_file, missing)
            continue
        for b in bands:
            if not {"time", "y", "x"}.issubset(ds[b].dims):
                log.warning("Dimensioni non valide per %s in %s. Saltato.", b, nc_file)
                continue

        time_len, H, W = ds[bands[0]].shape

        # -------- inferenza tempo per tempo ---------------------------
        sr_stack = {b: [] for b in bands}
        for t in range(time_len):
            lr = np.stack([ds[b].isel(time=t).values for b in bands], axis=-1)
            sr = sr_inference(lr, model, args.scale, args.margin_lr)  # (H*sc,W*sc,B)
            for i, b in enumerate(bands):
                sr_stack[b].append(sr[..., i])
        sr_stack = {b: np.stack(v, axis=0) for b, v in sr_stack.items()}

        # NDVI ---------------------------------------------------------
        ndvi_stack = None
        if {"B8","B4"}.issubset(bands):
            nir, red = sr_stack["B8"].astype("float32"), sr_stack["B4"].astype("float32")
            denom = nir + red
            ndvi_stack = np.where(denom != 0, (nir - red)/denom, np.nan)

        # -------- coordinate HR con linspace (mantiene orientamento) ---
        y_lr, x_lr = ds["y"].values, ds["x"].values
        H_hr, W_hr = sr_stack[bands[0]].shape[1:]
        y_hr = np.linspace(y_lr[0], y_lr[-1], H_hr)
        x_hr = np.linspace(x_lr[0], x_lr[-1], W_hr)

        # -------- costruiamo dataset output ---------------------------
        data_vars = {f"{b}": (("time","y","x"), sr_stack[b]) for b in bands}
        if ndvi_stack is not None:
            data_vars["NDVI"] = (("time","y","x"), ndvi_stack)
        for v in ds.data_vars:
            if v not in bands and v not in exclude:
                data_vars[v] = ds[v]

        ds_hr = xr.Dataset(
            data_vars=data_vars,
            coords={"time": ds["time"].values, "y": y_hr, "x": x_hr},
            attrs=ds.attrs,
        )
        enc = {k: {"zlib": True, "complevel": 3} for k in data_vars} 

        # Costruiamo il percorso di output mantendo la struttura originale
        # Cambiamo il nome del file al posto di essere NL_1335_S2_10m_256.nc -> NL_1335_S2_2-5m_256.nc
        rel_path = nc_file.relative_to(input_path)
        new_filename = rel_path.name.replace("10m", "2_5m")
        output_file = output_path / rel_path.parent / new_filename
        output_file.parent.mkdir(parents=True, exist_ok=True)

        ds_hr.to_netcdf(output_file, format="NETCDF4", encoding=enc)

if __name__ == "__main__":
    main()