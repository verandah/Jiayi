# Copyright (C) 2017-2022  Cleanlab Inc.
# This file is part of cleanlab.
#
# cleanlab is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cleanlab is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with cleanlab.  If not, see <https://www.gnu.org/licenses/>.


"""
Methods to rank/order data by cleanlab's `label quality score`.
Except for :py:func:`order_label_issues <cleanlab.rank.order_label_issues>`, which operates only on the subset of the data identified
as potential label issues/errors, the methods in this module can be used on whichever subset
of the dataset you choose (including the entire dataset) and provide a `label quality score` for
every example. You can then do something like: ``np.argsort(label_quality_score)`` to obtain ranked
indices of individual data.

CAUTION: These label quality scores are computed based on `pred_probs` from your model that must be out-of-sample!
You should never provide predictions on the same examples used to train the model,
as these will be overfit and unsuitable for finding label-errors.
To obtain out-of-sample predicted probabilities for every datapoint in your dataset, you can use :ref:`cross-validation <pred_probs_cross_val>`.
Alternatively it is ok if your model was trained on a separate dataset and you are only evaluating
labels in data that was previously held-out.
"""

import pandas as pd
import numpy as np
from typing import List
import warnings
from cleanlab.internal.label_quality_utils import (
    _subtract_confident_thresholds,
    get_normalized_entropy,
)


def order_label_issues(
    label_issues_mask: np.array,
    labels: np.array,
    pred_probs: np.array,
    *,
    rank_by: str = "self_confidence",
    rank_by_kwargs: dict = {},
) -> np.array:
    """Sorts label issues by label quality score.

    Default label quality score is "self_confidence".

    Parameters
    ----------
    label_issues_mask : np.array
      A boolean mask for the entire dataset where ``True`` represents a label
      issue and ``False`` represents an example that is accurately labeled with
      high confidence.

    labels : np.array
      Labels in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>` function.

    pred_probs : np.array (shape (N, K))
      Predicted-probabilities in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>` function.

    rank_by : str, optional
      Score by which to order label error indices (in increasing order). See
      the `method` argument of :py:func:`get_label_quality_scores
      <cleanlab.rank.get_label_quality_scores>`.

    rank_by_kwargs : dict, optional
      Optional keyword arguments to pass into :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>` function.
      Accepted args include `adjust_pred_probs`.

    Returns
    -------
    label_issues_idx : np.array
      Return an array of the indices of the label issues, ordered by the label-quality scoring method
      passed to `rank_by`.

    """

    assert len(pred_probs) == len(labels)

    # Convert bool mask to index mask
    label_issues_idx = np.arange(len(labels))[label_issues_mask]

    # Calculate label quality scores
    label_quality_scores = get_label_quality_scores(
        labels, pred_probs, method=rank_by, **rank_by_kwargs
    )

    # Get label quality scores for label issues
    label_quality_scores_issues = label_quality_scores[label_issues_mask]

    return label_issues_idx[np.argsort(label_quality_scores_issues)]


def get_label_quality_scores(
    labels: np.array,
    pred_probs: np.array,
    *,
    method: str = "self_confidence",
    adjust_pred_probs: bool = False,
) -> np.array:
    """Returns label quality scores for each datapoint.

    This is a function to compute label-quality scores for classification datasets,
    where lower scores indicate labels less likely to be correct.

    Score is between 0 and 1.

    1 - clean label (given label is likely correct).
    0 - dirty label (given label is likely incorrect).

    Parameters
    ----------
    labels : np.array
      A discrete vector of noisy labels, i.e. some labels may be erroneous.
      *Format requirements*: for dataset with K classes, labels must be in 0, 1, ..., K-1.

    pred_probs : np.array, optional
      An array of shape ``(N, K)`` of model-predicted probabilities,
      ``P(label=k|x)``. Each row of this matrix corresponds
      to an example `x` and contains the model-predicted probabilities that
      `x` belongs to each possible class, for each of the K classes. The
      columns must be ordered such that these probabilities correspond to
      class 0, 1, ..., K-1.

      **Caution**: `pred_probs` from your model must be out-of-sample!
      You should never provide predictions on the same examples used to train the model,
      as these will be overfit and unsuitable for finding label-errors.
      To obtain out-of-sample predicted probabilities for every datapoint in your dataset, you can use :ref:`cross-validation <pred_probs_cross_val>`.
      Alternatively it is ok if your model was trained on a separate dataset and you are only evaluating
      data that was previously held-out.

    method : {"self_confidence", "normalized_margin", "confidence_weighted_entropy"}, default="self_confidence"
      Label quality scoring method.

      Letting ``k = labels[i]`` and ``P = pred_probs[i]`` denote the given label and predicted class-probabilities
      for datapoint *i*, its score can either be:

      - ``'normalized_margin'``: ``P[k] - max_{k' != k}[ P[k'] ]``
      - ``'self_confidence'``: ``P[k]``
      - ``'confidence_weighted_entropy'``: ``entropy(P) / self_confidence``

      Let ``C = {0, 1, ..., K}`` denote the classification task's specified set of classes.

      The normalized_margin score works better for identifying class conditional label errors,
      i.e. examples for which another label in C is appropriate but the given label is not.

      The self_confidence score works better for identifying alternative label issues corresponding
      to bad examples that are: not from any of the classes in C, well-described by 2 or more labels in C,
      or generally just out-of-distribution (ie. anomalous outliers).

    adjust_pred_probs : bool, optional
      Account for class imbalance in the label-quality scoring by adjusting predicted probabilities
      via subtraction of class confident thresholds and renormalization.
      Set this to ``True`` if you prefer to account for class-imbalance.
      See `Northcutt et al., 2021 <https://jair.org/index.php/jair/article/view/12125>`_.

    Returns
    -------
    label_quality_scores : np.array
      Scores are between 0 and 1 where lower scores indicate labels less likely to be correct.

    See Also
    --------
    get_self_confidence_for_each_label
    get_normalized_margin_for_each_label
    get_confidence_weighted_entropy_for_each_label

    """

    # Available scoring functions to choose from
    scoring_funcs = {
        "self_confidence": get_self_confidence_for_each_label,
        "normalized_margin": get_normalized_margin_for_each_label,
        "confidence_weighted_entropy": get_confidence_weighted_entropy_for_each_label,
    }

    # Select scoring function
    try:
        scoring_func = scoring_funcs[method]
    except KeyError:
        raise ValueError(
            f"""
            {method} is not a valid scoring method for rank_by!
            Please choose a valid rank_by: self_confidence, normalized_margin, confidence_weighted_entropy
            """
        )

    # Adjust predicted probabilities
    if adjust_pred_probs:

        # Check if adjust_pred_probs is supported for the chosen method
        if method == "confidence_weighted_entropy":
            raise ValueError(f"adjust_pred_probs is not currently supported for {method}.")

        pred_probs = _subtract_confident_thresholds(labels, pred_probs)

    # Pass keyword arguments for scoring function
    input = {"labels": labels, "pred_probs": pred_probs}

    # Calculate scores
    label_quality_scores = scoring_func(**input)

    return label_quality_scores


def get_label_quality_ensemble_scores(
    labels: np.array,
    pred_probs_list: List[np.array],
    *,
    method: str = "self_confidence",
    adjust_pred_probs: bool = False,
    weight_ensemble_members_by: str = "accuracy",
    custom_weights: np.array = None,
    verbose: bool = True,
) -> np.array:
    """Returns label quality scores based on predictions from an ensemble of models.

    This is a function to compute label-quality scores for classification datasets,
    where lower scores indicate labels less likely to be correct.

    Ensemble scoring requires a list of pred_probs from each model in the ensemble.

    For each pred_probs in list, compute label quality score.
    Take the average of the scores with the chosen weighting scheme determined by `weight_ensemble_members_by`.

    Score is between 0 and 1:

    - 1 --- clean label (given label is likely correct).
    - 0 --- dirty label (given label is likely incorrect).

    Parameters
    ----------
    labels : np.array
      Labels in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>` function.

    pred_probs_list : List[np.array]
      Each element in this list should be an array of pred_probs in the same format
      expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>` function.
      Each element of `pred_probs_list` corresponds to the predictions from one model for all examples.

    method : {"self_confidence", "normalized_margin", "confidence_weighted_entropy"}, default="self_confidence"
      Label quality scoring method. See :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>`
      for scenarios on when to use each method.

    adjust_pred_probs : bool, optional
      `adjust_pred_probs` in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>` function.

    weight_ensemble_members_by : {"uniform", "accuracy", "custom"}, default="accuracy"
      Weighting scheme used to aggregate scores from each model:

      - "uniform": take the simple average of scores
      - "accuracy": take weighted average of scores, weighted by model accuracy
      - "custom": take weighted average of scores using custom weights that the user passes to the custom_weights parameter.

    custom_weights : np.array, default=None
      Weights used to aggregate scores from each model if weight_ensemble_members_by="custom".
      Length of this array must match the number of models: len(pred_probs_list).

    verbose : bool, default=True
      Set to ``False`` to suppress all print statements.

    Returns
    -------
    label_quality_scores : np.array

    See Also
    --------
    get_label_quality_scores

    """

    # Check pred_probs_list for errors
    assert isinstance(
        pred_probs_list, list
    ), f"pred_probs_list needs to be a list. Provided pred_probs_list is a {type(pred_probs_list)}"

    assert len(pred_probs_list) > 0, "pred_probs_list is empty."

    if len(pred_probs_list) == 1:
        warnings.warn(
            """
            pred_probs_list only has one element.
            Consider using get_label_quality_scores() if you only have a single array of pred_probs.
            """
        )

    # Raise ValueError if user passed custom_weights array but did not choose weight_ensemble_members_by="custom"
    if custom_weights is not None and weight_ensemble_members_by != "custom":
        raise ValueError(
            f"""
            custom_weights provided but weight_ensemble_members_by is not "custom"!
            """
        )

    # Generate scores for each model's pred_probs
    scores_list = []
    accuracy_list = []
    for pred_probs in pred_probs_list:

        # Calculate scores and accuracy
        scores = get_label_quality_scores(
            labels=labels,
            pred_probs=pred_probs,
            method=method,
            adjust_pred_probs=adjust_pred_probs,
        )
        scores_list.append(scores)

        # Only compute if weighting by accuracy
        if weight_ensemble_members_by == "accuracy":
            accuracy = (pred_probs.argmax(axis=1) == labels).mean()
            accuracy_list.append(accuracy)

    if verbose:
        print(f"Weighting scheme for ensemble: {weight_ensemble_members_by}")

    # Transform list of scores into an array of shape (N, M) where M is the number of models in the ensemble
    scores_ensemble = np.vstack(scores_list).T

    # Aggregate scores with chosen weighting scheme
    if weight_ensemble_members_by == "uniform":
        label_quality_scores = scores_ensemble.mean(axis=1)  # Uniform weights (simple average)

    elif weight_ensemble_members_by == "accuracy":
        weights = np.array(accuracy_list) / sum(accuracy_list)  # Weight by relative accuracy
        if verbose:
            print("Ensemble members will be weighted by: their relative accuracy")
            for i, acc in enumerate(accuracy_list):
                print(f"  Model {i} accuracy : {acc}")
                print(f"  Model {i} weights  : {weights[i]}")

        # Aggregate scores with weighted average
        label_quality_scores = (scores_ensemble * weights).sum(axis=1)

    elif weight_ensemble_members_by == "custom":

        # Check custom_weights for errors
        assert (
            custom_weights is not None
        ), "custom_weights is None! Please pass a valid custom_weights."

        assert len(custom_weights) == len(
            pred_probs_list
        ), "Length of custom_weights array must match the number of models: len(pred_probs_list)."

        # Aggregate scores with custom weights
        label_quality_scores = (scores_ensemble * custom_weights).sum(axis=1)

    else:
        raise ValueError(
            f"""
            {weight_ensemble_members_by} is not a valid weighting method for weight_ensemble_members_by!
            Please choose a valid weight_ensemble_members_by: uniform, accuracy, custom
            """
        )

    return label_quality_scores


def get_self_confidence_for_each_label(
    labels: np.array,
    pred_probs: np.array,
) -> np.array:
    """Returns the self-confidence label-quality score for each datapoint.

    This is a function to compute label-quality scores for classification datasets,
    where lower scores indicate labels less likely to be correct.

    The self-confidence is the holdout probability that an example belongs to
    its given class label.

    Self-confidence works better for finding out-of-distribution (OOD) examples, weird examples, bad examples,
    multi-label, and other types of label errors.

    Parameters
    ----------
    labels : np.array
      Labels in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>` function.

    pred_probs : np.array
      Predicted-probabilities in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>` function.

    Returns
    -------
    label_quality_scores : np.array
      An array of holdout probabilities that each example in `pred_probs` belongs to its
      label.

    """

    # np.mean is used so that this works for multi-labels (list of lists)
    label_quality_scores = np.array([np.mean(pred_probs[i, l]) for i, l in enumerate(labels)])
    return label_quality_scores


def get_normalized_margin_for_each_label(
    labels: np.array,
    pred_probs: np.array,
) -> np.array:
    """Returns the "normalized margin" label-quality score for each datapoint.

    This is a function to compute label-quality scores for classification datasets,
    where lower scores indicate labels less likely to be correct.

    Letting k denote the given label for a datapoint, the normalized margin is
    ``(p(label = k) - max(p(label != k)))``, i.e. the probability
    of the given label minus the probability of the argmax label that is not
    the given label. This gives you an idea of how likely an example is BOTH
    its given label AND not another label, and therefore, scores its likelihood
    of being a good label or a label error.

    Normalized margin works better for finding class conditional label errors where
    there is another label in the class that is better than the given label.

    Parameters
    ----------
    labels : np.array
      Labels in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>` function.

    pred_probs : np.array
      Predicted-probabilities in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>` function.

    Returns
    -------
    label_quality_scores : np.array
      An array of scores (between 0 and 1) for each example of its likelihood of
      being correctly labeled. ``normalized_margin = prob_label - max_prob_not_label``
    """

    self_confidence = get_self_confidence_for_each_label(labels, pred_probs)
    max_prob_not_label = np.array(
        [max(np.delete(pred_probs[i], l, -1)) for i, l in enumerate(labels)]
    )
    label_quality_scores = (self_confidence - max_prob_not_label + 1) / 2
    return label_quality_scores


def get_confidence_weighted_entropy_for_each_label(
    labels: np.array, pred_probs: np.array
) -> np.array:
    """Returns the "confidence weighted entropy" label-quality score for each datapoint.

    This is a function to compute label-quality scores for classification datasets,
    where lower scores indicate labels less likely to be correct.

    "confidence weighted entropy" is the normalized entropy divided by "self-confidence".

    Parameters
    ----------
    labels : np.array
      Labels in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>` function.

    pred_probs : np.array
      Predicted-probabilities in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>` function.

    Returns
    -------
    label_quality_scores : np.array
      An array of scores (between 0 and 1) for each example of its likelihood of
      being correctly labeled.
    """

    MIN_ALLOWED = 1e-6  # lower-bound clipping threshold to prevents 0 in logs and division
    self_confidence = get_self_confidence_for_each_label(labels, pred_probs)
    self_confidence = np.clip(self_confidence, a_min=MIN_ALLOWED, a_max=None)

    # Divide entropy by self confidence
    label_quality_scores = get_normalized_entropy(**{"pred_probs": pred_probs}) / self_confidence

    # Rescale
    clipped_scores = np.clip(label_quality_scores, a_min=MIN_ALLOWED, a_max=None)
    label_quality_scores = np.log(label_quality_scores + 1) / clipped_scores

    return label_quality_scores


def vote2score_1d(
    votes_1d: np.array,
    pred_probs_1d: np.array,
    method: str = "self_confidence",
    adjust_pred_probs: bool = False
) -> np.array:
    """
    Parameters
    ----------
    votes_1d : np.array (shape (M',))
      M' different Labels given by annotators for one sample.

    pred_probs_1d : np.array (shape (K,))
      To an example `x`, pred_probs_1d is an array of model-predicted probabilities that `x` belongs to each
      possible class, for each of the K classes. For example, any row of Predicted-probabilities in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>` function.

    method : str
      `method` in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>` function

    adjust_pred_probs : bool, optional
      `adjust_pred_probs` in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>` function.

    Returns
    -------
    label_quality_scores : np.array (shape(M,))
    Scores are between 0 and 1 where lower scores indicate labels less likely to be correct.

    """
    res = get_label_quality_scores(
        labels = votes_1d[:, None],
        pred_probs=np.tile(pred_probs_1d,(votes_1d.shape[0],1)),
        method=method,
        adjust_pred_probs=adjust_pred_probs
    )
    return res

def vote2score_2d(
    votes_2d: np.array,
    pred_probs_2d: np.array,
    method: str = "self_confidence",
    adjust_pred_probs: bool = False
) -> np.array:
    """
    Parameters
    ----------
    votes_2d : np.array (shape (N, M))
      Labels given by M annotators for N samples.
      Each row corresponding to one sample.

    pred_probs_2d : np.array (shape (N, K))
      To an example `x`, each row of pred_probs_2d is an array of model-predicted probabilities that `x` belongs to each possible class, for each of the K classes. Predicted-probabilities in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>` function.

    method : str
      `method` in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>` function

    adjust_pred_probs : bool, optional
      `adjust_pred_probs` in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>` function.

    Returns
    -------
    label_quality_scores : np.array (shape(M, N))
    Scores are between 0 and 1 where lower scores indicate labels less likely to be correct.

    """
    votes_scores_2d = np.zeros_like(votes_2d, dtype=float)
    N = votes_2d.shape[0]

    for idx in range(N):
        # get unique labels from annotators' labels
        votes_unique_1d = np.unique(votes_2d[idx])
        # compute scores for each unique labels
        votes_unique_1d_scores =vote2score_1d(
            votes_unique_1d,
            pred_probs_2d[idx],
            method,
            adjust_pred_probs=adjust_pred_probs
        )
        # in each row of votes_scores_2d matrix, record score for corresponding annotator's label
        votes_scores_2d[idx, :] = \
            votes_unique_1d_scores[np.where(votes_2d[idx][:, None] == votes_unique_1d[None, :])[1]]
    return votes_scores_2d

def vote2freq_1d(
    votes_1d: np.array
) -> np.array:
    """
    Parameters
    ----------
    votes_1d : np.array (shape (M',))
      M' different Labels given by annotators for one sample.

    Returns
    -------
    labels_freq : np.array (shape(M',))
    values are between 0 and 1 where value indicates the fraction of annotators agree on the corresponding label.

    """

    votes_unique, votes_count = np.unique(votes_1d, return_counts=True)
    return votes_count[np.where(votes_1d[:, None] == votes_unique[None, :])[1]]/votes_1d.shape[0]

def vote2freq_2d(
    votes_2d: np.array
) -> np.array:
    """
    Parameters
    ----------
    votes_2d : np.array (shape (M, N))
      each row corresponds to M' different Labels given by annotators for the current sample.

    Returns
    -------
    labels_freq : np.array (shape(M, N))
      values are between 0 and 1 where value indicates the fraction of annotators agree on the corresponding label.
    """
    return np.apply_along_axis(vote2freq_1d, 1, votes_2d)

def get_overall_score_2d(scores: np.array, labels: np.array, w: float = 1) -> np.array:
    """
    compute overall score for each sample.

    Parameters
    ----------
    labels: np.array (shape(M, N))

    scores: np.array (shape(M, N))
      A 2d matrix of scores. Each row corresponds to one sample. scores[i][j]: score of the annotator_j's label of the sample i.

    Returns
    ----------
    overall_score: np.array (shape(N))
    """
    # check if all scores are in range [0,1]
    if not ((scores <= 1) & (scores >= 0)).all():
        raise ValueError("scores are not in interval [0,1]")

    MIN_ALLOWED = 1e-6
    MAX_ALLOWED = 2**30

    N, M = np.shape(labels)[0], np.shape(scores)[1]
    overall_score = [ 0 for _ in range(N)]
    for idx in range(N):
        labels_unique_1d, index_unique_1d = np.unique(labels[idx], return_index= True)

        if np.shape(index_unique_1d) == 1:
            oR = (scores_label_unique_1d[-1])/np.clip((1 - scores_label_unique_1d[-1]), a_min=MIN_ALLOWED, a_max=MAX_ALLOWED)
        else:
            scores_label_unique_1d = sorted(scores[idx][index_unique_1d])
            oR = (scores_label_unique_1d[-1] - scores_label_unique_1d[-2])/np.clip((1 - scores_label_unique_1d[-1]), a_min=MIN_ALLOWED, a_max=MAX_ALLOWED)

        overall_score[idx] = 2/np.pi * np.arctan(w*oR)

    return overall_score

def get_label_weighted_score(
    label_quality_scores: np.array,
    label_agreements: np.array,
    alpha: float = 0.5,
    beta: float = 1,
    weighted_method: str = "weighted_arithmetic_mean"
) -> np.array:
    """
    compute a weighted score from label_quality_scores and label_agreement

    Parameters
    ----------
    label_quality_scores : np.array (shape(N, M))
      each row represents scores for annotator's label.

    label_agreements : np.array (shape(N, M))
      each row represents agreement for annotator's label.

    alpha : np.float
      a positive number between 0 and 1. alpha indicates how much weight we want to put on annotators. Used in weighted arithmetic mean.

    beta : np.float
      a positive number between 0 and 1. beta indicates how much weight we want to put on annotators. Used in weighted harmonic mean.

    weighted_method: str
        "weighted_arithmetic_mean": a*label_agreement + (1-a)*label_quality_score
        "weighted_harmonic_mean": (1+b*b)label_agreement*label_quality_score / (b*b*label_quality_score + label_agreement)

    Returns
    ----------
    weighted_scores: np.array (shape(N, M))
    """

    if weighted_method == "weighted_arithmetic_mean":
        return alpha*label_agreements + (1-alpha)*label_quality_scores
    elif weighted_method == "weighted_harmonic_mean":
        return (1+beta**2)*label_agreements*label_quality_scores/(beta**2*label_quality_scores + label_agreements)
    else:
        raise NotImplementedError(
            f"""weighted method {weighted_method} is not implemented. Choose weighted method from "weighted_arithmetic_mean" or "weighted_harmonic_mean". """
            )

def get_multiannotator_label_quality_scores(
    labels: np.array,
    pred_probs: np.array,
    alpha: float = 0.5,
    beta: float = 1,
    weighted_method: str = "weighted_arithmetic_mean",
    w: float = 1,
    method: str = "self_confidence",
    adjust_pred_probs: bool = False,
) -> pd.DataFrame:
    """Returns label_quality_scores, label overall score, sample_agreement, for each datapoint.

    Parameters
    ----------
    labels : np.array
        2D numpy array of (multiple) given labels for each example.
		labels[i][j] = given label for i-th example by j-th annotator, i is an integer in {0,...,K-1},
        j is an integer in {0,..., M-1}.

    pred_probs : np.array, optional
      same format as for get_label_quality_scores().

    alpha: float
      `alpha` in the same format expected by the :py:func:`get_label_weighted_scores <cleanlab.rank.get_label_weighted_scores>` function
      By default, alpha = 0.5

    beta: float
      `beta` in the same format expected by the :py:func:`get_label_weighted_scores <cleanlab.rank.get_label_weighted_scores>` function
      By default, beta = 1

    weighted_method: str
      `weighted_method` in the same format expected by the :py:func:`get_label_weighted_scores <cleanlab.rank.get_label_weighted_scores>` function
      {"weighted arithmetic mean", "weighted harmonic mean"}

    w: float
      `w` in the same format expected by the :py:func:`get_overall_scores_2d <cleanlab.rank.get_overall_scores_2d>` function
      By default, w = 1

    method : str
      `method` in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>` function

    adjust_pred_probs : bool, optional
      `adjust_pred_probs` in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>` function.

    Returns
    -------
    overall_score: pandas DataFrame (shape(N, M+2))
    df : pandas DataFrame in which each row corresponds to one example, with columns:
        score_annotator1, score_annotator2, ..., score_annotatorM, annotator_agreement, overall_score

    """

    # Pass keyword arguments for scoring function
    try:
        M = np.shape(labels)[1]
    except IndexError:
        raise IndexError(
            f"""
            labels matrix is not a 2d np.array. Please use a 2d np.array as labels matrix.
            """
        )

    M, N1, K, N = np.shape(labels)[1], np.shape(labels)[0], np.shape(pred_probs)[1], np.shape(pred_probs)[0]

    # check if input matrixes are valid. labels and pred_probs should have the same number of samples.
    if N1 != N:
        raise IndexError("The number of rows in labels matrix is not the same as the number of rows in pred_probs matrix. The number of rows is the number of samples. Please review input: labels and pred_prods")

    # check if parameters are valid.
    if w <= 0:
        raise ValueError(f"""w is required to be positive""")

    if weighted_method == "weighted_arithmetic_mean" and (alpha >1 or alpha < 0):
        raise ValueError(f"""alpha = {alpha} is not in range [0, 1]""")

    beta = np.clip(beta, a_min = None, a_max=2**15)
    if weighted_method == "weighted_harmonic_mean" and (beta < 0):
        raise ValueError(f"""beta = {beta} is not in range [0, infinity] """)

    # get quality score for each annotator's label
    label_quality_scores = vote2score_2d(labels, pred_probs, method = method, adjust_pred_probs=adjust_pred_probs)

    # get fraction of agreement for each annotator's label
    label_agreements = vote2freq_2d(labels)

    # get sample agreement
    sample_agreement = np.max(label_agreements, axis=1)

    # get weighted score for each label, weigted score combine the label_agreement and label_quality_score
    label_weighted_scores = get_label_weighted_score(label_quality_scores, label_agreements, alpha, beta, weighted_method)

    # get overall score for each sample
    overall_scores = get_overall_score_2d(label_weighted_scores, labels, w)

    # generate report as a pandas frame
    report = np.column_stack((label_quality_scores, sample_agreement, overall_scores))

    report_df = pd.DataFrame(report, columns = ["score_annotator_"+str(i) for i in range(M)] + ["agreement", "overall_score"], index = ["sample_"+ str(j) for j in range(N)])

    return report_df
