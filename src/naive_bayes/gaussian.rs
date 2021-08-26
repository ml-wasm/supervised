use linalg::{matrices::floats::FloatsMatrix, vectors::floats::FloatsVector};
use ndarray::{s, Axis, Zip};
use ndarray_stats::{QuantileExt, SummaryStatisticsExt};
use wasm_bindgen::prelude::*;

use crate::{
    base::Classifier,
    types::{MatF, VecF, VecI},
    utils,
};

use super::BaseNaiveBayes;

#[wasm_bindgen]
#[derive(Debug)]
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
    class_count: Option<VecF>,

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

    fn priors(&self) -> Result<&VecF, String> {
        Ok(self.priors.as_ref().ok_or("Not fitted yet")?)
    }

    fn epsilon(&self) -> Result<&f64, String> {
        Ok(self.epsilon.as_ref().ok_or("Not fitted yet")?)
    }

    fn class_count(&self) -> Result<&VecF, String> {
        Ok(self.class_count.as_ref().ok_or("Not fitted yet")?)
    }

    fn class_prior_mut(&mut self) -> Result<&mut VecF, String> {
        Ok(self.class_prior.as_mut().ok_or("Not fitted yet")?)
    }

    fn var_mut(&mut self) -> Result<&mut MatF, String> {
        Ok(self.var.as_mut().ok_or("Not fitted yet")?)
    }

    fn theta_mut(&mut self) -> Result<&mut MatF, String> {
        Ok(self.theta.as_mut().ok_or("Not fitted yet")?)
    }

    fn priors_mut(&mut self) -> Result<&mut VecF, String> {
        Ok(self.priors.as_mut().ok_or("Not fitted yet")?)
    }

    fn epsilon_mut(&mut self) -> Result<&mut f64, String> {
        Ok(self.epsilon.as_mut().ok_or("Not fitted yet")?)
    }

    fn class_count_mut(&mut self) -> Result<&mut VecF, String> {
        Ok(self.class_count.as_mut().ok_or("Not fitted yet")?)
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
            let joint_i = class_prior[i].ln();
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
        Ok(())
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
        if x.nrows() == 0 {
            return Ok((mu.clone(), var.clone()));
        }

        let (n_new, new_mu, new_var) = match sample_weight {
            Some(sw) => (
                sw.sum(),
                x.weighted_mean_axis(Axis(0), &sw).unwrap(),
                x.weighted_var_axis(Axis(0), &sw, 0.0).unwrap(),
            ),
            None => (
                x.nrows() as f64,
                x.mean_axis(Axis(0)).unwrap(),
                x.var_axis(Axis(0), 0.0),
            ),
        };

        if n_past == 0 {
            return Ok((new_mu, new_var));
        }

        let n_past = n_past as f64; // convenience
        let n_total = n_past + n_new;

        let total_mu = (n_new * &new_mu + n_past * mu) / n_total;

        let old_ssd = var * n_past;
        let new_ssd = &new_var * n_new;
        let total_ssd =
            &(mu - &new_mu).map(|x| x.powi(2)) * (n_new * n_past / n_total) + &new_ssd + &old_ssd;
        let total_var = &total_ssd / n_total;

        Ok((total_mu, total_var))
    }

    // Actual implementation of the Gaussian Naive Bayes fitting
    //
    // Parameters
    //   x: training data (n_samples, n_features)
    //   y: target values (n_samples, )
    //   classes: List of all the possible classes, should be provided in the first call (n_features, )
    //   sample_weight: weights applied to individual samples (n_samples, )
    fn partial_fit(
        &mut self,
        x: &MatF,
        y: &VecF,
        classes: Option<&VecF>,
        sample_weight: Option<&VecF>,
    ) -> Result<(), String> {
        let first_call = self.is_fitted();
        // I believe this is same as "_check_partial_fit_first_call"
        if first_call {
            let classes = classes.ok_or("Classes need to be provided for the first call")?;
            let unique_classes = utils::remove_duplicates_fv(classes);

            if classes.len() != unique_classes.len() {
                return Err(format!(
                    "There are some duplicates in the provided classes: {:?}",
                    classes.to_vec()
                )
                .to_string());
            }

            self.classes = Some(unique_classes);
        } else {
            let classes_before = self.classes()?.to_vec();
            let classes_new_not_in_before: Vec<f64> = classes
                .unwrap()
                .iter()
                .filter(|x| !classes_before.contains(*x))
                .map(|x| *x)
                .collect();

            if classes.is_some() && classes_new_not_in_before.len() > 0 {
                return Err(format!(
                    "The following labels in the provided classes: {:?}\ndo not exist in existing classes: {:?}",
                    classes_new_not_in_before, classes.unwrap().to_vec()
                )
                .to_string());
            }
        }

        // Skipped x and y validation
        // Skipped sample weight check

        // Boosting variance by epsilon to avoid numerical errors
        self.epsilon = Some(self.var_smoothing * x.var_axis(Axis(0), 0.0).max().unwrap());

        if first_call {
            let n_features = x.ncols();
            let n_classes = self.classes()?.len();
            self.theta = Some(MatF::zeros((n_classes, n_features)));
            self.var = Some(MatF::zeros((n_classes, n_features)));
            self.class_count = Some(VecF::zeros(n_classes));

            if self.priors.is_some() {
                let priors = self.priors()?;

                if priors.len() != n_classes {
                    return Err("Number of priors must match number of classes".to_string());
                }

                if (priors.sum() - 1.0).abs() <= 1e-8 {
                    return Err("Priors should sum to one".to_string());
                }

                if priors.iter().any(|x| *x < 0.0) {
                    return Err("Priors must be non-negative".to_string());
                }

                self.class_prior = Some(priors.clone());
            } else {
                self.class_prior = Some(VecF::zeros(n_classes));
            }
        } else {
            let theta = self.theta()?;
            if x.ncols() != theta.ncols() {
                return Err(format!(
                    "Number of features {} doesn't match previous data {}",
                    x.ncols(),
                    theta.ncols()
                )
                .to_string());
            }

            self.var = Some(self.var()? - *self.epsilon()?);
        }

        let classes = self.classes()?;
        let classes_vec = classes.to_vec();
        let unique_y = utils::remove_duplicates_fv(y);
        let unique_y_not_in_classes: Vec<f64> = unique_y
            .iter()
            .filter(|x| !classes_vec.contains(x))
            .map(|x| *x)
            .collect();

        if unique_y_not_in_classes.len() > 0 {
            return Err(format!(
                "The following labels in y: {:?}\ndo not exist in classes: {:?}",
                unique_y_not_in_classes, classes,
            )
            .to_string());
        }

        let mut class_count = self.class_count()?.clone();
        let mut theta = self.theta()?.clone();
        let mut var = self.var()?.clone();

        for y_i in unique_y.iter() {
            let i = classes_vec
                .binary_search_by(|a| a.partial_cmp(&y_i).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();
            let where_equal: Vec<usize> = y
                .iter()
                .enumerate()
                .filter(|x| x.1 == y_i)
                .map(|(i, _)| i)
                .collect();

            let x_i = x.select(Axis(0), &where_equal);

            let (sw_i, n_i) = match sample_weight {
                Some(sw) => {
                    let sw_i = sw.select(Axis(0), &where_equal);
                    let n_i = sw_i.sum();

                    (Some(sw_i), n_i)
                }
                None => (None, x_i.nrows() as f64),
            };

            let (new_theta, new_var) = GaussianNaiveBayes::update_mean_variance(
                class_count[i] as i32,
                &theta.row(i).to_owned(),
                &var.row(i).to_owned(),
                &x_i,
                sw_i.as_ref(),
            )?;

            theta.slice_mut(s![i, ..]).assign(&new_theta);
            var.slice_mut(s![i, ..]).assign(&new_var);
            class_count[i] += n_i;
        }

        var += *self.epsilon()?;

        if self.priors.is_none() {
            self.class_prior = Some(&class_count / class_count.sum());
        }

        self.theta = Some(theta);
        self.var = Some(var);
        self.class_count = Some(class_count);

        Ok(())
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
        classes: Option<&VecF>,
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

        let classes = utils::remove_duplicates_fv(y);

        self.partial_fit_reset(x, y, Some(&classes), sample_weight)?;

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
    // Returns
    //   c: predicted target values for x (n_samples, )
    #[wasm_bindgen(js_name = predict)]
    pub fn predict_js(&self, x: &FloatsMatrix) -> Result<FloatsVector, JsValue> {
        Ok(FloatsVector {
            data: self.predict(&x.data)?,
        })
    }

    // Return the log probability estimates for the input matrix x
    //
    // Params
    //   x: input samples (n_samples, n_features)
    // Returns
    //   c: log probability of the samples for each class in the model. The columns
    //      are in the same order as the `classes` (n_samples, n_features)
    #[wasm_bindgen(js_name = predictLogProba)]
    pub fn predict_log_proba_js(&self, x: &FloatsMatrix) -> Result<FloatsMatrix, JsValue> {
        Ok(FloatsMatrix {
            data: self.predict_log_proba(&x.data)?,
        })
    }

    // Return the probability estimates for the input matrix x
    //
    // Params
    //   x: input samples (n_samples, n_features)
    // Returns
    //   c: probability of the samples for each class in the model. The columns
    //      are in the same order as the `classes` (n_samples, n_features)
    #[wasm_bindgen(js_name = predictProba)]
    pub fn predict_proba_js(&self, x: &FloatsMatrix) -> Result<FloatsMatrix, JsValue> {
        Ok(FloatsMatrix {
            data: self.predict_proba(&x.data)?,
        })
    }

    #[wasm_bindgen(js_name = toString)]
    pub fn to_string(&self) -> String {
        format!("{:#?}", self)
    }
}
