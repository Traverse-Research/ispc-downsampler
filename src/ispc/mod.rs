#![allow(deref_nullptr)]

use ispc_rt::ispc_module;

use std::rc::Rc;

use crate::CachedWeight;
pub use downsample_ispc::*;
ispc_module!(downsample_ispc);

// `WeightDimensions` is a generated struct, so we cannot realistically add derivable traits to it.
// Because of this we disable the clippy warning.
#[allow(clippy::derivable_impls)]
impl Default for downsample_ispc::WeightDimensions {
    fn default() -> Self {
        Self {
            src_center: Default::default(),
            src_start: Default::default(),
            src_end: Default::default(),
        }
    }
}

pub(crate) struct WeightCollection {
    ispc_representation: downsample_ispc::WeightCollection,

    // Keep these because we need to keep them in memory
    _starts: Vec<u32>,
    _weight_counts: Vec<u32>,
    _weights: Vec<Rc<Vec<f32>>>,
    _weights_ptrs: Vec<*const f32>,
}

impl WeightCollection {
    pub(crate) fn new(mut weights: Vec<CachedWeight>) -> Rc<Self> {
        let (starts, counts): (Vec<_>, Vec<_>) = weights
            .iter()
            .map(|w| (w.start, w.coefficients.len() as u32))
            .unzip();

        let weights = weights
            .drain(..)
            .map(|w| w.coefficients)
            .collect::<Vec<_>>();

        let weights_ptrs = weights.iter().map(|v| v.as_ptr()).collect::<Vec<_>>();

        Rc::new(Self {
            ispc_representation: downsample_ispc::WeightCollection {
                starts: starts.as_ptr(),
                weight_counts: counts.as_ptr(),
                values: weights_ptrs.as_ptr(),
            },
            _starts: starts,
            _weight_counts: counts,
            _weights: weights,
            _weights_ptrs: weights_ptrs,
        })
    }

    pub(crate) fn ispc_representation(&self) -> &downsample_ispc::WeightCollection {
        &self.ispc_representation
    }
}
