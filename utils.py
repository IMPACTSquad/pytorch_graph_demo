import importlib
from glob import glob
import numpy as np

if importlib.util.find_spec("rasterio"):
    import rasterio


def open_dem(no_data_val=-3.40e38):
    try:
        dem_tif = rasterio.open("data/DEM.tif").read(1)
        dem_tif[dem_tif < no_data_val] = np.nan
    except NameError:  # if could not install rasterio, fall back to saved numpy array
        dem_tif = np.load("data/DEM.npy")
    return dem_tif


def open_land_cover(no_data_val=[0, 128]):
    try:
        land_cover_tif = rasterio.open("data/corine/land_cover_level_2.tif").read(1).astype(float)
        for ndv in no_data_val:
            land_cover_tif[land_cover_tif == ndv] = np.nan
    except NameError:  # if could not install rasterio, fall back to saved numpy array
        land_cover_tif = np.load("data/corine/lc_level_2.npy")
    return land_cover_tif


def open_land_cover_colormap():
    with open("data/corine/Legend/qgis_cmap_level_2.clr") as f:
        lines = f.readlines()[1:-1]  # first and last are "Confused"/"No data" i.e. classes not used
    lines = [l.strip().split(",") for l in lines]
    return {label[-1]: np.array([float(v) / 255 for v in label[1:-2]]) for label in lines}


def open_pm25(no_data_val=-999):
    files = glob("data/PM2_5/*.tif")
    pm25 = []
    try:
        for f in files:
            with rasterio.open(f) as src:
                pm25.append(src.read(1))
        pm25 = np.stack(pm25, axis=-1)
        pm25[pm25 == no_data_val] = np.nan
    except NameError:  # if could not install rasterio, fall back to saved numpy array
        pm25 = np.load("data/PM2_5/pm25.npy")
    return pm25


def get_masks(ops, seed, train_ratio=0.3, val_ratio=0.3):
    np.random.seed(seed)
    N = len(ops.flatten())

    # don't put any no data pixels in any set (i.e. should contribute to any loss values)
    idxs = np.argwhere(ops.flatten() != -999).flatten()

    # shuffle the indices
    idxs = np.random.choice(idxs, size=len(idxs), replace=False)

    # take first train_ratio * len(idxs) indices for training, and so on
    masks = (
        idxs[: int(len(idxs) * train_ratio)],
        idxs[int(len(idxs) * train_ratio) : int(len(idxs) * (train_ratio + val_ratio))],
        idxs[int(len(idxs) * (train_ratio + val_ratio)) :],
    )

    mask_tr = np.zeros(N).astype(bool)
    mask_va = np.zeros(N).astype(bool)
    mask_te = np.zeros(N).astype(bool)

    mask_tr[masks[0]] = True
    mask_va[masks[1]] = True
    mask_te[masks[2]] = True
    return mask_tr, mask_va, mask_te


def add_colorbar(fig, img, one_ax, x_shift=0.2, height_scale=0.95):
    bounds = one_ax.get_position().bounds
    bounds = (
        bounds[0] + x_shift,
        (3 - height_scale) * bounds[1] / 2,
        bounds[2],
        bounds[3] * height_scale,
    )
    cbar = fig.add_axes(bounds)
    cbar.axis("off")
    fig.colorbar(img, ax=cbar)


def axes_off(axes):
    if axes.ndim == 2:
        for ax_splice in axes:
            for ax in ax_splice:
                ax.set_xticks([])
                ax.set_yticks([])
    elif axes.ndim == 1:
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
    else:
        ax.set_xticks([])
        ax.set_yticks([])
