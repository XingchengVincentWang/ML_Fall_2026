import os
import json
import math

import numpy as np
import matplotlib.pyplot as plt


class EarlyStopper():
  """Early stopping for a hand-written training loop (toolkit-neutral).

  Keras users: use keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=..., patience=..., restore_best_weights=True)
  instead; this class implements the same rule for loops you write yourself (e.g., PyTorch).

  Usage, once per epoch after computing the validation loss:
      stopper = EarlyStopper(patience=5, min_delta=1e-3)
      for epoch in range(max_epochs):
          ...train one epoch, compute val_loss...
          if stopper.step(val_loss, copy.deepcopy(model.state_dict())):   # state is optional
              break
      model.load_state_dict(stopper.best_state)      # restore the best weights

  The loss counts as improved only if it is lower than the best so far by more than min_delta.
  step() returns True when there has been no improvement for `patience` epochs in a row.
  """
  def __init__(self, patience=5, min_delta=1e-3):
    self.patience, self.min_delta = patience, min_delta
    self.best = float("inf")
    self.best_epoch = 0          # 1-based epoch of the best validation loss
    self.best_state = None
    self.epochs_seen = 0
    self.bad_epochs = 0

  def step(self, val_loss, state=None):
    self.epochs_seen += 1
    if val_loss < self.best - self.min_delta:
      self.best, self.best_epoch, self.best_state, self.bad_epochs = val_loss, self.epochs_seen, state, 0
    else:
      self.bad_epochs += 1
    return self.bad_epochs >= self.patience


class Helper():
  """Utilities for the ships-in-satellite-images assignments.

  Put this file in the same directory as your notebook.
  The data file `shipsnet.json` is searched for in several places (see CANDIDATE_DIRS);
  if it is somewhere else (e.g., your Google Drive), pass the folder explicitly:
      helper = Helper(data_dir="/content/drive/MyDrive/my_folder")
  """

  EarlyStopper = EarlyStopper      # so that helper.EarlyStopper(...) works on a Helper instance
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

  def describe_hardware(self):
    """Print whether a GPU is visible to PyTorch and/or TensorFlow (whichever you have installed)."""
    found = False
    try:
      import torch
      found = True
      print("PyTorch", torch.__version__, "| GPU available:", torch.cuda.is_available(),
            ("| " + torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "")
    except ImportError:
      pass
    try:
      import tensorflow as tf
      found = True
      gpus = tf.config.list_physical_devices("GPU")
      print("TensorFlow", tf.__version__, "| GPUs:", [g.name for g in gpus] if gpus else "none")
    except ImportError:
      pass
    if not found:
      print("Neither PyTorch nor TensorFlow is installed here.")

  def showFilters(self, W, num_cols=8, titles=None, suptitle=None):
    """Show convolution filters as small images.

    W: array of shape (k, k, c_in, n_filters) (Keras layout). PyTorch stores (n_filters, c_in, k, k): pass
       W.transpose(2, 3, 1, 0) of the numpy array. With 3 input channels each filter is drawn in color
       (its own min-max scaling, so only the pattern of dark and bright matters, not the absolute values);
       otherwise the first input channel is drawn in gray.
    """
    W = np.asarray(W, dtype=float)
    k, _, c_in, n = W.shape
    num_rows = math.ceil(n / num_cols)
    fig = plt.figure(figsize=(1.6 * min(n, num_cols), 1.6 * num_rows))
    for i in range(n):
      f = W[:, :, :, i]
      f = (f - f.min()) / (f.max() - f.min() + 1e-12)
      ax = fig.add_subplot(num_rows, min(n, num_cols), i + 1)
      if c_in == 3:
        _ = ax.imshow(f, interpolation="nearest")
      else:
        _ = ax.imshow(f[:, :, 0], cmap="gray", interpolation="nearest")
      _ = ax.set_axis_off()
      _ = ax.set_title(titles[i] if titles is not None else str(i), fontsize=8)
    if suptitle:
      fig.suptitle(suptitle)
    fig.tight_layout()
    return fig
