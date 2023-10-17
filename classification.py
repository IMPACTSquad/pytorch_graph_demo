import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import train
import utils


def add_colorbar(fig, img, class_names, one_ax, x_shift=0.2):
    bounds = one_ax.get_position().bounds
    bounds = (bounds[0] + x_shift, *bounds[1:])
    # make a new axes upon which the cbar will be drawn
    cbar = fig.add_axes(bounds)
    cbar.axis("off")
    cbar = fig.colorbar(img, ax=cbar, ticks=np.arange(len(class_names)))
    cbar.ax.set_yticklabels(class_names)


def unflatten(mask, flattened, outside_value=np.nan, dtype=float):
    # when training we flattened the data in space, now we need to unflatten it so that rows/cols represent lines of lat/lon
    unflattened = np.ones_like(mask, dtype=dtype) * outside_value
    unflattened[mask] = flattened
    return unflattened


def main():
    trained_model, data, masks = train.train_classification()
    mask_tr, mask_va, mask_te = masks

    trained_model.eval()  # double check that model is in evaluation mode
    with torch.no_grad():  # don't track gradients during validation
        predictions = trained_model(data).cpu().numpy()

    # undo flattening
    ip = utils.open_pm25()
    gt = utils.open_land_cover() - 1
    mask = ~np.all(np.isnan(ip), axis=-1) * ~np.isnan(gt)
    predictions = unflatten(mask, np.argmax(predictions, axis=1))
    mask_tr = unflatten(mask, mask_tr, outside_value=False, dtype=bool)
    mask_va = unflatten(mask, mask_va, outside_value=False, dtype=bool)
    mask_te = unflatten(mask, mask_te, outside_value=False, dtype=bool)

    class_colormap = utils.open_land_cover_colormap()
    mappable = ListedColormap(class_colormap.values())
    fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(15, 10))

    training_labels = gt.copy()
    training_labels[~mask_tr] = np.nan
    validation_labels = gt.copy()
    validation_labels[~mask_va] = np.nan
    test_labels = gt.copy()
    test_labels[~mask_te] = np.nan
    training_predictions = predictions.copy()
    training_predictions[~mask_tr] = np.nan
    validation_predictions = predictions.copy()
    validation_predictions[~mask_va] = np.nan
    test_predictions = predictions.copy()
    test_predictions[~mask_te] = np.nan

    ax[0, 0].matshow(training_labels, cmap=mappable, vmin=-0.5, vmax=len(class_colormap) - 0.5)
    ax[0, 0].set_title("Ground truth (training)")
    ax[0, 1].matshow(validation_labels, cmap=mappable, vmin=-0.5, vmax=len(class_colormap) - 0.5)
    ax[0, 1].set_title("Ground truth (val.)")
    img = ax[0, 2].matshow(test_labels, cmap=mappable, vmin=-0.5, vmax=len(class_colormap) - 0.5)
    ax[0, 2].set_title("Ground truth (test)")
    add_colorbar(fig, img, class_colormap.keys(), ax[0, 2], x_shift=0.02)

    ax[1, 0].matshow(training_predictions, cmap=mappable, vmin=-0.5, vmax=len(class_colormap) - 0.5)
    ax[1, 0].set_title("Predictions (training)")
    ax[1, 1].matshow(validation_predictions, cmap=mappable, vmin=-0.5, vmax=len(class_colormap) - 0.5)
    ax[1, 1].set_title("Predictions (val.)")
    img = ax[1, 2].matshow(test_predictions, cmap=mappable, vmin=-0.5, vmax=len(class_colormap) - 0.5)
    ax[1, 2].set_xticks([])
    ax[1, 2].set_title("Predictions (test)")
    add_colorbar(fig, img, class_colormap.keys(), ax[1, 2], x_shift=0.02)

    mappable = ListedColormap([[1.0, 0, 0], [0, 1.0, 0]])
    ax[2, 0].matshow(
        np.where(mask * mask_tr, training_labels == training_predictions, np.nan), cmap=mappable, vmin=-0.5, vmax=1.5
    )
    ax[2, 0].set_title("Errors (training)")
    ax[2, 1].matshow(
        np.where(mask * mask_va, validation_labels == validation_predictions, np.nan),
        cmap=mappable,
        vmin=-0.5,
        vmax=1.5,
    )
    ax[2, 1].set_title("Errors (val.)")
    img = ax[2, 2].matshow(
        np.where(mask * mask_te, test_labels == test_predictions, np.nan), cmap=mappable, vmin=-0.5, vmax=1.5
    )
    ax[2, 2].set_title("Errors (test)")
    add_colorbar(fig, img, ["Differing labels", "Matching labels"], ax[2, 2], x_shift=0.02)
    utils.axes_off(ax)

    plt.show()


if __name__ == "__main__":
    main()
