use linalg::vectors::floats::FloatsVector;
use ndarray::Zip;
use ndarray_stats::SummaryStatisticsExt;
use wasm_bindgen::prelude::*;

use crate::types::VecF;

fn weighted_sum(sample_score: &VecF, sample_weights: Option<&VecF>, normalize: bool) -> f64 {
    if normalize {
        match sample_weights {
            Some(sw) => sample_score.weighted_mean(sw).unwrap(),
            None => sample_score.mean().unwrap(),
        }
    } else {
        match sample_weights {
            Some(sw) => sample_score.dot(sw),
            None => sample_score.sum(),
        }
    }
}

// Accuracy classification score
//
// Params,
//   y_true: the correct labels
//   y_pred: the predicted labels
//   normalize: if true, return the fraction of correctly classified samples,
//              else, the number of correctly classified samples
//   sample_weights: weights for each sample
pub fn accuracy_score(
    y_true: &VecF,
    y_pred: &VecF,
    normalize: bool,
    sample_weights: Option<&VecF>,
) -> f64 {
    // ignoring multilabel preds for now
    let score =
        VecF::from_iter(
            y_true
                .iter()
                .zip(y_pred.iter())
                .map(|(&t, &p)| if t == p { 1.0 } else { 0.0 }),
        );

    weighted_sum(&score, sample_weights, normalize)
}

// Accuracy classification score
//
// Params,
//   y_true: the correct labels
//   y_pred: the predicted labels
//   normalize: if true, return the fraction of correctly classified samples,
//              else, the number of correctly classified samples
//   sample_weights: weights for each sample
#[wasm_bindgen(js_name = accuracyScore)]
pub fn accuracy_score_js(y_true: &FloatsVector, y_pred: &FloatsVector, normalize: bool) -> f64 {
    accuracy_score(&y_true.data, &y_pred.data, normalize, None)
}

// Accuracy classification score
//
// Params,
//   y_true: the correct labels
//   y_pred: the predicted labels
//   normalize: if true, return the fraction of correctly classified samples,
//              else, the number of correctly classified samples
#[wasm_bindgen(js_name = accuracyScoreWithSampleWeights)]
pub fn accuracy_score_js_with_sample_weights(
    y_true: &FloatsVector,
    y_pred: &FloatsVector,
    normalize: bool,
    sample_weights: &FloatsVector,
) -> f64 {
    accuracy_score(
        &y_true.data,
        &y_pred.data,
        normalize,
        Some(&sample_weights.data),
    )
}
