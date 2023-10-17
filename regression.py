import numpy as np
import matplotlib.pyplot as plt
import torch

import utils
import train


def main():
    trained_model, data, masks = train.train_regression()
    mask_tr, mask_va, mask_te = masks
    trained_model.eval()  # double check that model is in evaluation mode
    with torch.no_grad():  # don't track gradients during validation
        predictions = trained_model(data).cpu().numpy().flatten()

    # undo flattening
    ip = utils.open_pm25()
    gt = utils.open_dem()
    mask = ~np.all(np.isnan(ip), axis=-1) * ~np.isnan(gt)
    predictions_unflattened = np.zeros(ip.shape[:2]) * np.nan
    predictions_unflattened[mask] = predictions.flatten()

    # reverse normalization (i.e. DEM was normalized to [0, 1] during training, we want to reverse that)
    gt_unnormalized = gt[mask]
    op_min, op_max = np.nanmin(gt_unnormalized), np.nanmax(gt_unnormalized)
    predictions_unflattened = predictions_unflattened * (op_max - op_min) + op_min

    vmin = min(np.nanmin(gt), predictions_unflattened.min())
    vmax = max(np.nanmax(gt), predictions_unflattened.max())

    fig, ax = plt.subplots(2, 2, sharey=True, sharex=True, figsize=(12, 5))
    cbar = ax[0, 0].matshow(gt, vmin=vmin, vmax=vmax)
    utils.add_colorbar(fig, cbar, ax[0, 0], x_shift=0.06)
    ax[0, 0].set_title("Ground truth DEM")
    cbar = ax[0, 1].matshow(predictions_unflattened, vmin=vmin, vmax=vmax)
    utils.add_colorbar(fig, cbar, ax[0, 1], x_shift=0.06)
    ax[0, 1].set_title("Predicted DEM using PM2.5 time series")

    gt_flattened = gt[mask]
    gt_flattened[~mask_te] = np.nan  # show only test pixels
    gt_unflattened = np.zeros(ip.shape[:2]) * np.nan
    gt_unflattened[mask] = gt_flattened.flatten()
    cbar = ax[1, 0].matshow(gt_unflattened, vmin=vmin, vmax=vmax)
    utils.add_colorbar(fig, cbar, ax[1, 0], x_shift=0.06)
    ax[1, 0].set_title("Ground truth DEM (test nodes only)")
    predictions[~mask_te] = np.nan  # show only test pixels
    predictions_unflattened = np.zeros(ip.shape[:2]) * np.nan
    predictions_unflattened[mask] = predictions.flatten()
    predictions_unflattened = predictions_unflattened * (op_max - op_min) + op_min
    cbar = ax[1, 1].matshow(predictions_unflattened, vmin=vmin, vmax=vmax)
    utils.add_colorbar(fig, cbar, ax[1, 1], x_shift=0.06)
    ax[1, 1].set_title("Predicted DEM (test nodes only)")

    plt.show()


if __name__ == "__main__":
    main()
