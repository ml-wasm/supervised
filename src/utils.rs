use crate::types::VecF;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: String);
}

pub fn console_log(s: String) {
    log(s)
}

pub fn remove_duplicates_fv(x: &VecF) -> VecF {
    let mut xv = x.to_vec();
    xv.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    xv.dedup();

    VecF::from_vec(xv)
}
