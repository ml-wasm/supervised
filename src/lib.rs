mod base;
mod metrics;
mod naive_bayes;
mod types;
mod utils;

use wasm_bindgen::prelude::*;

pub fn set_panic_hook() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

#[wasm_bindgen(start)]
pub fn start() {
    set_panic_hook();
}
