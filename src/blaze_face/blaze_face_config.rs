// Reference implementation:
// https://github.com/hollance/BlazeFace-PyTorch/blob/master/blazeface.py

pub struct BlazeFaceConfig {
    pub(crate) x_scale: f32,
    pub(crate) y_scale: f32,
    pub(crate) h_scale: f32,
    pub(crate) w_scale: f32,
    pub(crate) score_clipping_thresh: f32,
    pub(crate) min_score_thresh: f32,
    pub(crate) min_suppression_threshold: f32,
}

impl BlazeFaceConfig {
    pub(crate) fn back(
        score_clipping_thresh: f32,
        min_score_thresh: f32,
        min_suppression_threshold: f32,
    ) -> BlazeFaceConfig {
        BlazeFaceConfig {
            x_scale: 256.,
            y_scale: 256.,
            h_scale: 256.,
            w_scale: 256.,
            score_clipping_thresh,
            min_score_thresh,
            min_suppression_threshold,
        }
    }

    pub(crate) fn front(
        score_clipping_thresh: f32,
        min_score_thresh: f32,
        min_suppression_threshold: f32,
    ) -> BlazeFaceConfig {
        BlazeFaceConfig {
            x_scale: 128.,
            y_scale: 128.,
            h_scale: 128.,
            w_scale: 128.,
            score_clipping_thresh,
            min_score_thresh,
            min_suppression_threshold,
        }
    }
}
