use linalg::{
    matrices::{floats::FloatsMatrix, integers::IntegersMatrix},
    vectors::{floats::FloatsVector, integers::IntegersVector},
};
use wasm_bindgen::JsValue;

use crate::{metrics, types};

pub enum Param {
    // Vectors
    VecF(types::VecF),
    VecI(types::VecI),

    // Matrices
    MatF(types::MatF),
    MatI(types::MatI),

    // Scalar
    F(f64),
    I(i32),

    N,
}

impl Param {
    pub fn to_jsvalue(&self) -> JsValue {
        match self {
            // Vectors
            Param::VecF(x) => JsValue::from(FloatsVector { data: x.clone() }),
            Param::VecI(x) => JsValue::from(IntegersVector { data: x.clone() }),

            // Matrices
            Param::MatF(x) => JsValue::from(FloatsMatrix { data: x.clone() }),
            Param::MatI(x) => JsValue::from(IntegersMatrix { data: x.clone() }),

            // Scalar
            Param::F(x) => JsValue::from(x.clone()),
            Param::I(x) => JsValue::from(x.clone()),

            Param::N => JsValue::NULL,
        }
    }
}

pub type Params = std::collections::HashMap<String, Param>;

pub trait Estimator {
    fn get_params(&self) -> Params;

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
}

// Maybe we don't need this
pub trait Classifier {
    fn get_estimator_type(&self) -> String {
        "classfier".to_string()
    }

    fn predict(&self, x: &types::MatF) -> Result<types::VecF, String>;

    fn score(
        &self,
        x: &types::MatF,
        y: &types::VecF,
        sample_weights: Option<&types::VecF>,
    ) -> Result<f64, String> {
        Ok(metrics::classification::accuracy_score(
            y,
            &self.predict(x)?,
            true,
            sample_weights,
        ))
    }
}
// pub trait Regressor {
//     fn get_estimator_type(&self) -> String {
//         "regressor".to_string()
//     }
//
//     fn score(&self, x: linalg::Linalg, y: linalg::Linalg) -> f64 {
//         todo!()
//     }
// }
