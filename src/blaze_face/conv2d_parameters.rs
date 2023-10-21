use candle_core::Tensor;

pub(crate) struct Conv2dParameters {
    pub(crate) weight: Tensor,
    pub(crate) bias: Option<Tensor>,
}
