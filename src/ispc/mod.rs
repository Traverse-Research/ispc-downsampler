#![allow(deref_nullptr)]

use ispc_rt::ispc_module;
ispc_module!(downsample_ispc);


impl Default for downsample_ispc::CoefficientVariables {
    fn default() -> Self {
        Self { src_center: Default::default(), src_start: Default::default(), src_end: Default::default() }
    }
}
