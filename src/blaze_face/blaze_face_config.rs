// Reference implementation:
// https://github.com/hollance/BlazeFace-PyTorch/blob/master/blazeface.py

pub(crate) struct BlazeFaceConfig {
    pub(crate) num_classes: usize,
    pub(crate) num_anchors: usize,
    pub(crate) num_coords: usize,
    pub(crate) score_clipping_thresh: f32,
    pub(crate) x_scale: f32,
    pub(crate) y_scale: f32,
    pub(crate) h_scale: f32,
    pub(crate) w_scale: f32,
    pub(crate) min_score_thresh: f32,
    pub(crate) min_suppression_threshold: f32,
}
