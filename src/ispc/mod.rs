#![allow(deref_nullptr)]

use ispc_rt::ispc_module;

use std::{pin::Pin, rc::Rc};

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
    _starts: Pin<Vec<u32>>,
    _weight_counts: Pin<Vec<u32>>,
    _weights: Pin<Vec<Rc<Vec<f32>>>>,
    _weights_ptrs: Pin<Vec<*const f32>>,
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
            _starts: Pin::new(starts),
            _weight_counts: Pin::new(counts),
            _weights: Pin::new(weights),
            _weights_ptrs: Pin::new(weights_ptrs),
        })
    }

    pub(crate) fn ispc_representation(&self) -> &downsample_ispc::WeightCollection {
        &self.ispc_representation
    }
}

pub(crate) struct Weights {
    ispc_representation: SampleWeights,

    // Need to be kept alive because the ispc_representation holds pointers to them
    _horizontal_weights: Pin<Rc<WeightCollection>>,
    _vertical_weights: Pin<Rc<WeightCollection>>,
}

impl Weights {
    pub(crate) fn new(
        horizontal_weights: Rc<WeightCollection>,
        vertical_weights: Rc<WeightCollection>,
    ) -> Self {
        Self {
            ispc_representation: SampleWeights {
                vertical_weights: vertical_weights.ispc_representation(),
                horizontal_weights: horizontal_weights.ispc_representation(),
            },
            _vertical_weights: Pin::new(vertical_weights),
            _horizontal_weights: Pin::new(horizontal_weights),
        }
    }

    pub(crate) fn ispc_representation(&self) -> &SampleWeights {
        &self.ispc_representation
    }
}
