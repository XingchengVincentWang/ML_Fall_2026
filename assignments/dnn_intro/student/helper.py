import os
import json
import math

import numpy as np
import matplotlib.pyplot as plt


class Helper():
  """Utilities for the ships-in-satellite-images assignments.

  Put this file in the same directory as your notebook.
  The data file `shipsnet.json` is searched for in several places (see CANDIDATE_DIRS);
  if it is somewhere else (e.g., your Google Drive), pass the folder explicitly:
      helper = Helper(data_dir="/content/drive/MyDrive/my_folder")
  """
  DATASET = "shipsnet.json"

  CANDIDATE_DIRS = [
      "./Data",
      ".",
      "../resource/asnlib/publicdata/ships_in_satellite_images/data",
      "/content/Data",
      "/content",
      "/content/drive/MyDrive/Data",
      "/content/drive/MyDrive/Colab Notebooks/Data",
  ]

  def __init__(self, data_dir=None):
    dirs = ([data_dir] if data_dir else []) + self.CANDIDATE_DIRS
    for d in dirs:
      if os.path.isfile(os.path.join(d, self.DATASET)):
        self.DATA_DIR = d
        break
    else:
      raise FileNotFoundError(
          "Cannot find {f}. Upload it next to your notebook (or to a Data/ folder), "
          "or set DATA_DIR to the folder that contains it.".format(f=self.DATASET))

  def getData(self):
    """Returns (data, labels): data has shape (N, 80, 80, 3), uint8; labels has shape (N,), values 0/1."""
    return self.json_to_numpy(os.path.join(self.DATA_DIR, self.DATASET))

  def json_to_numpy(self, json_file):
    with open(json_file) as f:
      dataset = json.load(f)

    data = np.array(dataset['data']).astype('uint8')
    labels = np.array(dataset['labels']).astype('uint8')

    # The file stores each image channel-first (3, 80, 80); we want (80, 80, 3)
    data = data.reshape([-1, 3, 80, 80]).transpose([0, 2, 3, 1])
    return data, labels

  def showData(self, data, labels, num_cols=5, cmap=None):
    """Plot a grid of images with their labels as titles."""
    num_rows = math.ceil(data.shape[0] / num_cols)

    fig = plt.figure(figsize=(10, 10))
    for i in range(0, data.shape[0]):
      img, img_label = data[i], labels[i]
      ax = fig.add_subplot(num_rows, num_cols, i + 1)
      _ = ax.set_axis_off()
      _ = ax.set_title(img_label)
      _ = plt.imshow(img, cmap=cmap)
    fig.tight_layout()
    return fig

  def showWeights(self, W, num_cols=4, titles=None, shared_scale=False, suptitle=None):
    """Plot weight vectors as 2D images.

    W: array of shape (n_pixels, n_vectors), n_pixels a perfect square (e.g., 6400 = 80 x 80).
       Column k is the weight vector of unit k.  (If your toolkit stores weights as (n_units, n_pixels),
       pass the transpose.)
    Gray = 0, white = positive, black = negative. By default each image has its own symmetric color scale
    (a unit whose weights are all tiny then looks like speckle); with shared_scale=True all images use one scale.
    titles: optional list of titles (a single vector is titled "weights" by default); suptitle: optional figure title.
    """
    W = np.asarray(W)
    if W.ndim == 1:
      W = W[:, None]
    dim = round(W.shape[0] ** 0.5)
    assert dim * dim == W.shape[0], "number of weights per unit must be a perfect square"
    n = W.shape[1]
    num_rows = math.ceil(n / num_cols)

    fig = plt.figure(figsize=(2.5 * min(n, num_cols), 2.5 * num_rows))
    for k in range(n):
      ax = fig.add_subplot(num_rows, min(n, num_cols), k + 1)
      scale = np.abs(W).max() if shared_scale else np.abs(W[:, k]).max()
      _ = ax.imshow(W[:, k].reshape(dim, dim), cmap="gray", vmin=-scale, vmax=scale, interpolation="nearest")
      _ = ax.set_axis_off()
      default = "weights" if n == 1 else "unit {k}".format(k=k)
      _ = ax.set_title(titles[k] if titles is not None else default, fontsize=9)
    if suptitle:
      fig.suptitle(suptitle)
    fig.tight_layout()
    return fig
