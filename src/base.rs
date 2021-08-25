// pub type Params = std::collections::HashMap<String, linalg::Linalg>;

pub trait Estimator {
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
}

pub trait Classifier {
    fn get_estimator_type(&self) -> String {
        "classfier".to_string()
    }

    fn score(&self, x: linalg::Linalg, y: linalg::Linalg) -> f64 {
        todo!()
    }
}
pub trait Regressor {
    fn get_estimator_type(&self) -> String {
        "regressor".to_string()
    }

    fn score(&self, x: linalg::Linalg, y: linalg::Linalg) -> f64 {
        todo!()
    }
}
