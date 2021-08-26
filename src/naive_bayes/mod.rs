use ndarray::{Array2, Axis};
use ndarray_stats::QuantileExt;

use crate::{
    types::{MatF, VecF},
    utils,
};

mod gaussian;

trait BaseNaiveBayes {
    fn classes(&self) -> Result<&VecF, String>;

    // Probably move this to the Estimator trait later
    fn is_fitted(&self) -> bool;

    fn joint_log_likelihood(&self, x: &MatF) -> Result<MatF, String>;

    fn check_x(&self, x: &MatF) -> Result<(), String>;

    // Perform classification on the given test vectors
    //
    // Parameters
    //   x: input samples (n_samples, n_features)
    // Returns
    //   c: predicted target values for x (n_samples, )
    fn predict(&self, x: &MatF) -> Result<VecF, String> {
        if !self.is_fitted() {
            return Err("This Naive Bayes classifier has not been fitted yet".to_string());
        }

        self.check_x(x)?;

        let jll = self.joint_log_likelihood(x)?;

        let classes = self.classes()?;

        Ok(VecF::from_iter(
            jll.rows().into_iter().map(|x| classes[x.argmax().unwrap()]),
        ))
    }

    // Return the log probability estimates for the input matrix x
    //
    // Params
    //   x: input samples (n_samples, n_features)
    // Returns
    //   c: log probability of the samples for each class in the model. The columns
    //      are in the same order as the `classes` (n_samples, n_features)
    fn predict_log_proba(&self, x: &MatF) -> Result<MatF, String> {
        if !self.is_fitted() {
            return Err("This Naive Bayes classifier has not been fitted yet".to_string());
        }

        self.check_x(x)?;

        let jll = self.joint_log_likelihood(x)?;
        let log_prob_x = (jll.map(|x| x.exp())).sum_axis(Axis(1)).map(|x| x.ln());

        // TODO check what this np.atleast_2d step is about
        Ok(&jll - &log_prob_x)
    }

    // Return the probability estimates for the input matrix x
    //
    // Params
    //   x: input samples (n_samples, n_features)
    // Returns
    //   c: probability of the samples for each class in the model. The columns
    //      are in the same order as the `classes` (n_samples, n_features)
    fn predict_proba(&self, x: &MatF) -> Result<MatF, String> {
        Ok(self.predict_log_proba(x)?.map(|x| x.exp()))
    }
}
