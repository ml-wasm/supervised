mod utils;

use wasm_bindgen::prelude::*;

pub use linalg::init_thread_pool;

#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);
}

#[wasm_bindgen(start)]
pub fn start() {
    utils::set_panic_hook();
}
