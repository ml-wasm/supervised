use crate::types::VecF;

pub fn remove_depulicates_fv(x: &VecF) -> VecF {
    let mut xv = x.to_vec();
    xv.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    xv.dedup();

    VecF::from_vec(xv)
}
