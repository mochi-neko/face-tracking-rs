// Reference implementation:
// https://github.com/hollance/BlazeFace-PyTorch/blob/master/blazeface.py

use half::f16;

pub struct BlazeFaceConfig {
    pub(crate) x_scale: f16,
    pub(crate) y_scale: f16,
    pub(crate) h_scale: f16,
    pub(crate) w_scale: f16,
    pub(crate) score_clipping_thresh: f16,
    pub(crate) min_score_thresh: f16,
    pub(crate) min_suppression_threshold: f16,
}

impl BlazeFaceConfig {
    pub(crate) fn back(
        score_clipping_thresh: f32,
        min_score_thresh: f32,
        min_suppression_threshold: f32,
    ) -> BlazeFaceConfig {
        BlazeFaceConfig {
            x_scale: f16::from_f32(256.),
            y_scale: f16::from_f32(256.),
            h_scale: f16::from_f32(256.),
            w_scale: f16::from_f32(256.),
            score_clipping_thresh: f16::from_f32(score_clipping_thresh),
            min_score_thresh: f16::from_f32(min_score_thresh),
            min_suppression_threshold: f16::from_f32(min_suppression_threshold),
        }
    }

    pub(crate) fn front(
        score_clipping_thresh: f32,
        min_score_thresh: f32,
        min_suppression_threshold: f32,
    ) -> BlazeFaceConfig {
        BlazeFaceConfig {
            x_scale: f16::from_f32(128.),
            y_scale: f16::from_f32(128.),
            h_scale: f16::from_f32(128.),
            w_scale: f16::from_f32(128.),
            score_clipping_thresh: f16::from_f32(score_clipping_thresh),
            min_score_thresh: f16::from_f32(min_score_thresh),
            min_suppression_threshold: f16::from_f32(min_suppression_threshold),
        }
    }
}
