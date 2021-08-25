mod base;
mod naive_bayes;
mod types;
mod utils;

use wasm_bindgen::prelude::*;

pub use linalg::init_thread_pool;

pub fn set_panic_hook() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

#[wasm_bindgen(start)]
pub fn start() {
    set_panic_hook();
}
