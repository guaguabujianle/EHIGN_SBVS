import numpy as np
from sklearn.metrics import roc_auc_score, cohen_kappa_score
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, precision_score, recall_score, accuracy_score, precision_recall_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, cohen_kappa_score, auc, roc_curve

import numpy as np

def calculate_enrichment_factor(y_true, y_pred, percentage):
    # Get the total number of samples
    T = len(y_true)

    # Get the total number of positive samples
    A = np.sum(y_true)

    # Calculate N, the number of samples in the top "percentage" of predictions
    N = int(T * (percentage / 100))

    # Get indices of predictions sorted by score in descending order
    indices_sorted = np.argsort(y_pred)[::-1]

    # Get "N" indices corresponding to the top "percentage"
    top_indices = indices_sorted[:N]

    # Count the number of true positives in the top "percentage"
    a = np.sum(y_true[top_indices])

    # Calculate and return the enrichment factor
    return (a / N) / (A / T)


def range_logAUC(true_y, predicted_score, FPR_range=(0.001, 0.1)):
    """
    Author: Yunchao "Lance" Liu (lanceknight26@gmail.com)
    Calculate logAUC in a certain FPR range (default range: [0.001, 0.1]).
    This was used by previous methods [1] and the reason is that only a
    small percentage of samples can be selected for experimental tests in
    consideration of cost. This means only molecules with very high
    predicted score can be worth testing, i.e., the decision
    threshold is high. And the high decision threshold corresponds to the
    left side of the ROC curve, i.e., those FPRs with small values. Also,
    because the threshold cannot be predetermined, the area under the curve
    is used to consolidate all possible thresholds within a certain FPR
    range. Finally, the logarithm is used to bias smaller FPRs. The higher
    the logAUC[0.001, 0.1], the better the performance.
    A perfect classifer gets a logAUC[0.001, 0.1] ) of 1, while a random
    classifer gets a logAUC[0.001, 0.1] ) of around 0.0215 (See [2])
    References:
    [1] Mysinger, M.M. and B.K. Shoichet, Rapid Context-Dependent Ligand
    Desolvation in Molecular Docking. Journal of Chemical Information and
    Modeling, 2010. 50(9): p. 1561-1573.
    [2] Mendenhall, J. and J. Meiler, Improving quantitative
    structure–activity relationship models using Artificial Neural Networks
    trained with dropout. Journal of computer-aided molecular design,
    2016. 30(2): p. 177-189.
    :param true_y: numpy array of the ground truth. Values are either 0 (
    inactive) or 1(active).
    :param predicted_score: numpy array of the predicted score (The
    score does not have to be between 0 and 1)
    :param FPR_range: the range for calculating the logAUC formated in
    (x, y) with x being the lower bound and y being the upper bound
    :return: a numpy array of logAUC of size [1,1]
    """

    # FPR range validity check
    if FPR_range == None:
        raise Exception('FPR range cannot be None')
    lower_bound = np.log10(FPR_range[0])
    upper_bound = np.log10(FPR_range[1])
    if (lower_bound >= upper_bound):
        raise Exception('FPR upper_bound must be greater than lower_bound')

    # Get the data points' coordinates. log_fpr is the x coordinate, tpr is
    # the y coordinate.
    fpr, tpr, thresholds = roc_curve(true_y, predicted_score, pos_label=1)
    log_fpr = np.log10(fpr)

    # Intecept the curve at the two ends of the region, i.e., lower_bound,
    # and upper_bound
    tpr = np.append(tpr, np.interp([lower_bound, upper_bound], log_fpr, tpr))
    log_fpr = np.append(log_fpr, [lower_bound, upper_bound])

    # Sort both x-, y-coordinates array
    x = np.sort(log_fpr)
    y = np.sort(tpr)

    # For visulization of the plot before trimming, uncomment the following
    # line, with proper libray imported
    # plt.plot(x, y)
    # For visulization of the plot in the trimmed area, uncomment the
    # following line
    # plt.xlim(lower_bound, upper_bound)

    # Get the index of the lower and upper bounds
    lower_bound_idx = np.where(x == lower_bound)[-1][-1]
    upper_bound_idx = np.where(x == upper_bound)[-1][-1]

    # Create a new array trimmed at the lower and upper bound
    trim_x = x[lower_bound_idx:upper_bound_idx + 1]
    trim_y = y[lower_bound_idx:upper_bound_idx + 1]

    area = auc(trim_x, trim_y) / 2

    return area

def enrichment_score(y_true: np.ndarray, y_pred: np.ndarray, alpha: float = [0.001, 0.005, 0.01, 0.05]):
  try:
    from rdkit.ML.Scoring.Scoring import CalcEnrichment
  except ModuleNotFoundError:
    raise ImportError("This function requires RDKit to be installed.")

  # validation
  assert len(y_true) == len(y_pred), 'Number of examples do not match'
  assert np.array_equal(
      np.unique(y_true).astype(int),
      [0, 1]), ('Class labels must be binary: %s' % np.unique(y_true))

  yt = np.asarray(y_true)
  yp = np.asarray(y_pred)

  yt = yt.flatten()
  yp = yp.flatten()  # Index 1 because one_hot predictions

  scores = list(zip(yt, yp))
  scores = sorted(scores, key=lambda pair: pair[1], reverse=True)

  return CalcEnrichment(scores, 0, alpha)


def bedroc_score(y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 20.0):
  """Compute BEDROC metric.
  BEDROC metric implemented according to Truchon and Bayley that modifies
  the ROC score by allowing for a factor of early recognition.
  Please confirm details from [1]_.
  Parameters
  ----------
  y_true: np.ndarray
    Binary class labels. 1 for positive class, 0 otherwise
  y_pred: np.ndarray
    Predicted labels
  alpha: float, default 20.0
    Early recognition parameter
  Returns
  -------
  float
    Value in [0, 1] that indicates the degree of early recognition
  Notes
  -----
  This function requires RDKit to be installed.
  References
  ----------
  .. [1] Truchon et al. "Evaluating virtual screening methods: good and bad metrics
     for the “early recognition” problem." Journal of chemical information and modeling
     47.2 (2007): 488-508.
  """
  try:
    from rdkit.ML.Scoring.Scoring import CalcBEDROC
  except ModuleNotFoundError:
    raise ImportError("This function requires RDKit to be installed.")

  # validation
  assert len(y_true) == len(y_pred), 'Number of examples do not match'
  assert np.array_equal(
      np.unique(y_true).astype(int),
      [0, 1]), ('Class labels must be binary: %s' % np.unique(y_true))

  yt = np.asarray(y_true)
  yp = np.asarray(y_pred)

  yt = yt.flatten()
  yp = yp.flatten()  # Index 1 because one_hot predictions

  scores = list(zip(yt, yp))
  scores = sorted(scores, key=lambda pair: pair[1], reverse=True)

  return CalcBEDROC(scores, 0, alpha)


def get_cindex(gt, pred):
    gt_mask = gt.reshape((1, -1)) > gt.reshape((-1, 1))
    diff = pred.reshape((1, -1)) - pred.reshape((-1, 1))
    h_one = (diff > 0)
    h_half = (diff == 0)
    CI = np.sum(gt_mask * h_one * 1.0 + gt_mask * h_half * 0.5) / np.sum(gt_mask)

    return CI

def get_k(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs*y_pred) / float(sum(y_pred*y_pred))

def squared_error_zero(y_obs,y_pred):
    k = get_k(y_obs,y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k*y_pred)) * (y_obs - (k* y_pred)))
    down= sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))

    return 1 - (upp / float(down))


def get_rm2(ys_orig,ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute((r2*r2)-(r02*r02))))

def r_squared_error(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean) )

    return mult / float(y_obs_sq * y_pred_sq)

def positive(y_true):
    return np.sum((y_true == 1))

def negative(y_true):
    return np.sum((y_true == 0))

def true_positive(y_true, y_pred):
    return np.sum(np.bitwise_and(y_true == 1, y_pred == 1))

def false_positive(y_true, y_pred):
    return np.sum(np.bitwise_and(y_true == 0, y_pred == 1))

def true_negative(y_true, y_pred):
    return np.sum(np.bitwise_and(y_true == 0, y_pred == 0))

def false_negative(y_true, y_pred):
    return np.sum(np.bitwise_and(y_true == 1, y_pred == 0))

def accuracy(y_true, y_pred):
    sample_count = 1.
    for s in y_true.shape:
        sample_count *= s

    return np.sum((y_true == y_pred)) / sample_count

def sensitive(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    p = positive(y_true) + 1e-9
    return tp / p

def specificity(y_true, y_pred):
    tn = true_negative(y_true, y_pred)
    n = negative(y_true) + 1e-9
    return tn / n

def precision(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    return tp / (tp + fp)

def recall(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    return tp / (tp + fn)

def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    reca = recall(y_true, y_pred)
    fs = (2 * prec * reca) / (prec + reca)
    return fs

auc_score = roc_auc_score
kappa_score = cohen_kappa_score

if __name__ == '__main__':
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 1])
    y_pred = np.array([1, 1, 0, 1, 0, 1, 0, 1])

    sens = sensitive(y_true, y_pred)
    spec = specificity(y_true, y_pred)
    prec = precision(y_true, y_pred)
    reca = recall(y_true, y_pred)
    fs = f1_score(y_true, y_pred)

    print(sens)
    print(spec)
    print(prec)
    print(reca)
    print(fs)

# %%