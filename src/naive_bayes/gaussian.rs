use linalg::{matrices::floats::FloatsMatrix, vectors::floats::FloatsVector};
use ndarray::Axis;
use ndarray_stats::SummaryStatisticsExt;
use wasm_bindgen::prelude::*;

use crate::{
    base::Classifier,
    types::{MatF, VecF, VecI},
    utils,
};

use super::BaseNaiveBayes;

#[wasm_bindgen]
pub struct GaussianNaiveBayes {
    // Prior probabilities of the classes. If specified the priors are not
    // adjusted according to the data.
    priors: Option<VecF>,

    // Portion of the largest variance of all features that is added to
    // variances for calculation stability.
    var_smoothing: f64,

    // Has this classifier been fitted
    fitted: bool,

    // Number of training samples observed within each class
    class_count: Option<VecI>,

    // Probability of each class
    class_prior: Option<VecF>,

    // Class labels known to the classifier
    classes: Option<VecF>,

    // Absolute additive value to variances
    epsilon: Option<f64>,

    // Number of features seen during the "fit"
    n_features_in: Option<i32>,

    // Variance of each feature per class
    var: Option<MatF>,

    // Mean of each feature per class
    theta: Option<MatF>,
}

// Getters
impl GaussianNaiveBayes {
    fn class_prior(&self) -> Result<&VecF, String> {
        Ok(self.class_prior.as_ref().ok_or("Not fitted yet")?)
    }

    fn var(&self) -> Result<&MatF, String> {
        Ok(self.var.as_ref().ok_or("Not fitted yet")?)
    }

    fn theta(&self) -> Result<&MatF, String> {
        Ok(self.theta.as_ref().ok_or("Not fitted yet")?)
    }
}

// impl Estimator for GaussianNaiveBayes {
//     fn get_params(&self) -> Params;
//
//     fn set_params(&mut self, parameters: Params);
//
//     fn set_features(&mut self, x: linalg::Linalg);
//
//     fn get_n_features(&self) -> i32;
//
//     fn set_feature_names(&mut self, x: Vec<String>);
//
//     fn get_features_names(&self);
//
//     fn validate_data(&self, x: linalg::Linalg, y: linalg::Linalg);
// }

impl Classifier for GaussianNaiveBayes {}

impl BaseNaiveBayes for GaussianNaiveBayes {
    fn classes(&self) -> Result<&VecF, String> {
        Ok(self.classes.as_ref().ok_or("Not fitted yet")?)
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }

    // Compute the unnormalized posterior log probability of x, i.e. log P(c) + log P(x|c)
    //
    // Params
    //   x: input array (n_classes, n_samples)
    fn joint_log_likelihood(&self, x: &MatF) -> Result<MatF, String> {
        let classes = self.classes()?;
        let class_prior = self.class_prior()?;
        let var = self.var()?;
        let theta = self.theta()?;
        let mut joint_log_likelihood = MatF::default((classes.len(), x.nrows()));

        for i in 0..classes.len() {
            let joint_i = class_prior[i];
            let mut n_ij =
                -0.5 * (&(x - &theta.row(i)).map(|x| x.powi(2)) / &var.row(i)).sum_axis(Axis(1));
            n_ij -= (&var.row(i) * (std::f64::consts::PI * 2.0))
                .map(|x| x.ln())
                .sum()
                * 0.5;

            (&n_ij + joint_i).assign_to(joint_log_likelihood.row_mut(i));
        }

        Ok(joint_log_likelihood.reversed_axes())
    }

    fn check_x(&self, x: &MatF) -> Result<(), String> {
        todo!()
    }
}

impl GaussianNaiveBayes {
    // Compute online update of Gaussian mean and variance
    //
    // Params
    //   n_past: Number of samples represented in the old mean and variance (int)
    //   mu: means for Gaussians in the original set (number of Gaussians, )
    //   var: variances for Gaussians in the original set (number of Gaussians, )
    //   x: training points
    //   sample_weight: weights applied to the individual sample (n_sample, )
    //
    // Returns
    //   total_mu: Updated mean for each Gaussian over the combined set (number of Gaussians, )
    //   total_var: Updated variance for each Gaussian over the combined set (number of Gaussians, )
    fn update_mean_variance(
        n_past: i32,
        mu: &VecF,
        var: &VecF,
        x: &MatF,
        sample_weight: Option<&VecF>,
    ) -> Result<(VecF, VecF), String> {
        // if x.nrows() == 0 {
        //     return Ok((mu, var));
        // }

        // let (n_new, new_mu, new_var) = match sample_weight {
        //     Some(sw) => (
        //         sw.sum(),
        //         x.weighted_mean_axis(Axis(0), &sw),
        //         x.weighted_var_axis(Axis(0), &sw, 0.0),
        //     ),
        //     None => (
        //         x.nrows() as f64,
        //         x.var_axis(Axis(0), 0.0),
        //         x.mean_axis(Axis(0)),
        //     ),
        // };

        todo!()
    }

    // Actual implementation of the Gaussian Naive Bayes fitting
    //
    // Parameters
    //   x: training data (n_samples, n_features)
    //   y: target values (n_samples, )
    //   classes: List of all the possible classes (n_features, )
    //   sample_weight: weights applied to individual samples (n_samples, )
    fn partial_fit(
        &mut self,
        x: &MatF,
        y: &VecF,
        classes: &VecF,
        sample_weight: Option<&VecF>,
    ) -> Result<(), String> {
        todo!()
    }

    // Actual implementation of the Gaussian Naive Bayes fitting with reset
    //
    // Parameters
    //   x: training data
    //   y: target values
    //   classes: List of all the possible classes
    //   sample_weight: weights applied to individual samples
    fn partial_fit_reset(
        &mut self,
        x: &MatF,
        y: &VecF,
        classes: &VecF,
        sample_weight: Option<&VecF>,
    ) -> Result<(), String> {
        self.classes = None;

        self.partial_fit(x, y, classes, sample_weight)?;

        Ok(())
    }

    // Fit the Gaussian Naive Bayes classifier according to x and y
    //
    // Parameters
    //   x: training vectors (n_samples, n_features)
    //   y: target values (n_samples)
    //   sample_weight: weights applied to individual samples (n_samples, ) (optional)
    fn fit(&mut self, x: &MatF, y: &VecF, sample_weight: Option<&VecF>) -> Result<(), String> {
        // TODO validation for x and y

        let classes = utils::remove_depulicates_fv(y);

        self.partial_fit_reset(x, y, &classes, sample_weight)?;

        Ok(())
    }
}

#[wasm_bindgen]
impl GaussianNaiveBayes {
    // Construct a new GaussianNaiveBayes classifier
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            priors: None,
            var_smoothing: 1e-9,
            fitted: true,
            class_count: None,
            class_prior: None,
            classes: None,
            epsilon: None,
            n_features_in: None,
            var: None,
            theta: None,
        }
    }

    // Construct a new GaussianNaiveBayes classifier with the given priors and
    // var smoothing
    //
    // Parameters
    //   priors: Prior probabilities of the classes (n_classes, )
    //   var_smoothing: Portion of the largest variance of all features that is added to
    //                  variances for calculation stability. (Default is 1e-9)
    #[wasm_bindgen(js_name = newWithParams)]
    pub fn new_with_params(priors: &FloatsVector, var_smoothing: f64) -> Self {
        Self {
            priors: Some(priors.data.clone()),
            var_smoothing,
            fitted: true,
            class_count: None,
            class_prior: None,
            classes: None,
            epsilon: None,
            n_features_in: None,
            var: None,
            theta: None,
        }
    }

    // Fit the Gaussian Naive Bayes classifier according to x and y
    //
    // Parameters
    //   x: training vectors (n_samples, n_features)
    //   y: target values (n_samples, )
    #[wasm_bindgen(js_name = fit)]
    pub fn fit_js(&mut self, x: &FloatsMatrix, y: &FloatsVector) -> Result<(), JsValue> {
        // TODO validation for x and y
        self.fit(&x.data, &y.data, None)?;

        Ok(())
    }

    // Fit the Gaussian Naive Bayes classifier according to x and y
    //
    // Parameters
    //   x: training vectors (n_samples, n_features)
    //   y: target values (n_samples, )
    //   sample_weight: weights applied to individual samples (n_samples, )
    #[wasm_bindgen(js_name = fitWithSampleWeight)]
    pub fn fit_with_sample_weight_js(
        &mut self,
        x: &FloatsMatrix,
        y: &FloatsVector,
        sample_weight: &FloatsVector,
    ) -> Result<(), JsValue> {
        // TODO validation for x and y

        self.fit(&x.data, &y.data, Some(&sample_weight.data))?;

        Ok(())
    }

    // Perform classification on the given test vectors
    //
    // Parameters
    //   x: input samples (n_samples, n_features)
    #[wasm_bindgen(js_name = predict)]
    pub fn predict_to_js(&self, x: &FloatsMatrix) -> Result<FloatsVector, JsValue> {
        let y = self.predict(&x.data)?;

        Ok(FloatsVector { data: y })
    }
}
