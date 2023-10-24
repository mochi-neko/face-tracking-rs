// Reference implementation:
// https://github.com/hollance/BlazeFace-PyTorch/blob/master/blazeface.py

use std::ops::Add;

use candle_core::{IndexOp, Result, Tensor};
use candle_nn::VarBuilder;
use half::f16;

use super::{
    blaze_face_back_model::BlazeFaceBackModel,
    blaze_face_config::BlazeFaceConfig,
    blaze_face_front_model::BlazeFaceFrontModel,
};

pub enum ModelType {
    Back,
    Front,
}

pub(crate) trait BlazeFaceModel {
    fn forward(
        &self,
        xs: &Tensor,
    ) -> Result<(Tensor, Tensor)>;
}

pub struct BlazeFace {
    model: Box<dyn BlazeFaceModel>,
    anchors: Tensor,
    config: BlazeFaceConfig,
}

impl BlazeFace {
    pub fn load(
        model_type: ModelType,
        variables: VarBuilder,
        anchors: Tensor,
        score_clipping_thresh: f32,
        min_score_thresh: f32,
        min_suppression_threshold: f32,
    ) -> Result<Self> {
        match model_type {
            | ModelType::Back => {
                let model = BlazeFaceBackModel::load(variables)?;
                Ok(BlazeFace {
                    model: Box::new(model),
                    anchors,
                    config: BlazeFaceConfig::back(
                        score_clipping_thresh,
                        min_score_thresh,
                        min_suppression_threshold,
                    ),
                })
            },
            | ModelType::Front => {
                let model = BlazeFaceFrontModel::load(variables)?;
                Ok(BlazeFace {
                    model: Box::new(model),
                    anchors,
                    config: BlazeFaceConfig::front(
                        score_clipping_thresh,
                        min_score_thresh,
                        min_suppression_threshold,
                    ),
                })
            },
        }
    }

    fn forward(
        &self,
        xs: &Tensor, // back:(batch_size, 3, 256, 256) or front:(batch_size, 3, 128, 128)
    ) -> Result<(Tensor, Tensor)> // (batch_size, 896, 16), (batch_size, 896, 1)
    {
        self.model.forward(xs)
    }

    fn convert_tensors_to_detections(
        &self,
        raw_boxes: &Tensor,  // (batch_size, 896, 16)
        raw_scores: &Tensor, // (batch_size, 896, 1)
    ) -> Result<Vec<Tensor>> {
        unimplemented!()
    }

    fn decode_boxes(
        raw_boxes: &Tensor, // (batch_size, 896, 16)
        anchors: &Tensor,   // (896, 4)
        config: &BlazeFaceConfig,
    ) -> Result<Tensor> // (batch_size, 896, 16)
    {
        let boxes = Tensor::zeros_like(raw_boxes)?; // (batch_size, 896, 16)

        let x_scale =
            Tensor::from_slice(&[config.x_scale], 1, raw_boxes.device())?; // (1)
        let y_scale =
            Tensor::from_slice(&[config.y_scale], 1, raw_boxes.device())?; // (1)
        let w_scale =
            Tensor::from_slice(&[config.w_scale], 1, raw_boxes.device())?; // (1)
        let h_scale =
            Tensor::from_slice(&[config.h_scale], 1, raw_boxes.device())?; // (1)

        let two = Tensor::from_slice(
            &[f16::from_f32(2.)],
            1,
            raw_boxes.device(),
        )?; // (1)

        let x_anchor = anchors.i((.., 0))?; // (896)
        let y_anchor = anchors.i((.., 1))?; // (896)
        let w_anchor = anchors.i((.., 2))?; // (896)
        let h_anchor = anchors.i((.., 3))?; // (896)

        let x_center = raw_boxes
            .i((.., .., 0))? // (batch_size, 896)
            .broadcast_div(&x_scale)? // / (1)
            .broadcast_mul(&w_anchor)? // * (896)
            .broadcast_add(&x_anchor)?; // + (896)
                                        // = (batch_size, 896)

        let y_center = raw_boxes
            .i((.., .., 1))?
            .broadcast_div(&y_scale)?
            .broadcast_mul(&h_anchor)?
            .broadcast_add(&y_anchor)?;

        let w = raw_boxes
            .i((.., .., 2))? // (batch_size, 896)
            .broadcast_div(&w_scale)? // / (1)
            .broadcast_mul(&w_anchor)?; // * (896)
                                        // = (batch_size, 896)

        let h = raw_boxes
            .i((.., .., 3))?
            .broadcast_div(&h_scale)?
            .broadcast_mul(&h_anchor)?;

        let x_min = (&x_center - w.broadcast_div(&two)?)?; // (batch_size, 896)
        let x_max = (&x_center + w.broadcast_div(&two)?)?;
        let y_min = (&y_center - h.broadcast_div(&two)?)?;
        let y_max = (&y_center + h.broadcast_div(&two)?)?;

        // Bounding box
        boxes
            .i((.., .., 0))? // (batch_size, 896)
            .add(y_min)?; // + (batch_size, 896)
        boxes
            .i((.., .., 1))?
            .add(x_min)?;
        boxes
            .i((.., .., 2))?
            .add(y_max)?;
        boxes
            .i((.., .., 3))?
            .add(x_max)?;

        // Face keypoints: right_eye, left_eye, nose, mouth, right_ear, left_ear
        for k in 0..6 {
            let offset = 4 + k * 2;

            let keypoint_x = raw_boxes
                .i((.., .., offset))? // (batch_size, 896)
                .broadcast_div(&x_scale)? // / (1)
                .broadcast_mul(&w_anchor)? // * (896)
                .broadcast_add(&x_anchor)?; // + (896)
                                            // = (batch_size, 896)

            let keypoint_y = raw_boxes
                .i((.., .., offset + 1))?
                .broadcast_div(&y_scale)?
                .broadcast_mul(&h_anchor)?
                .broadcast_add(&y_anchor)?;

            // Keypoint
            boxes
                .i((.., .., offset))
                .add(keypoint_x)?;
            boxes
                .i((.., .., offset + 1))
                .add(keypoint_y)?;
        }

        Ok(boxes) // (batch_size, 896, 16)
    }

    fn process_scores(
        &self,
        raw_scores: &Tensor,
    ) -> Result<Tensor> {
        unimplemented!()
    }

    fn weighted_non_max_suppression(
        &self,
        detection: &Tensor,
    ) -> Result<Tensor> {
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn test_forward_back() {
        // Set up the device and dtype
        let device = Device::Cpu;
        let dtype = DType::F16;
        let batch_size = 1;

        // Load the variables
        let variables = candle_nn::VarBuilder::from_pth(
            "src/blaze_face/data/blazefaceback.pth",
            dtype,
            &device,
        )
        .unwrap();

        // Load the anchors
        let anchors =
            Tensor::read_npy("src/blaze_face/data/anchorsback.npy").unwrap();
        assert_eq!(anchors.dims(), &[896, 4,]);

        // Load the model
        let model = BlazeFace::load(
            ModelType::Back,
            variables,
            anchors,
            100.,
            0.65,
            0.3,
        )
        .unwrap();

        // Set up the input Tensor
        let input = Tensor::zeros(
            (batch_size, 3, 256, 256),
            dtype,
            &device,
        )
        .unwrap();

        // Call forward method and get the output
        let output = model.forward(&input).unwrap();

        assert_eq!(output.0.dims(), &[batch_size, 896, 1,]);
        assert_eq!(output.1.dims(), &[batch_size, 896, 16,]);
    }

    #[test]
    fn test_forward_front() {
        // Set up the device and dtype
        let device = Device::Cpu;
        let dtype = DType::F16;
        let batch_size = 1;

        // Load the variables
        let variables = candle_nn::VarBuilder::from_pth(
            "src/blaze_face/data/blazeface.pth",
            dtype,
            &device,
        )
        .unwrap();

        // Load the anchors
        let anchors =
            Tensor::read_npy("src/blaze_face/data/anchors.npy").unwrap();
        assert_eq!(anchors.dims(), &[896, 4,]);

        // Load the model
        let model = BlazeFace::load(
            ModelType::Front,
            variables,
            anchors,
            100.,
            0.75,
            0.3,
        )
        .unwrap();

        // Set up the input Tensor
        let input = Tensor::zeros(
            (batch_size, 3, 128, 128),
            dtype,
            &device,
        )
        .unwrap();

        // Call forward method and get the output
        let output = model.forward(&input).unwrap();

        assert_eq!(output.0.dims(), &[batch_size, 896, 1,]);
        assert_eq!(output.1.dims(), &[batch_size, 896, 16,]);
    }

    #[test]
    fn test_decode_boxes() {
        // Set up the device and dtype
        let device = Device::Cpu;
        let dtype = DType::F16;
        let batch_size = 1;

        // Set up the anchors and configuration
        let anchors = Tensor::read_npy("src/blaze_face/data/anchorsback.npy")
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let config = BlazeFaceConfig::back(100., 0.65, 0.3);

        // Set up the input Tensor
        let input =
            Tensor::zeros((batch_size, 896, 16), dtype, &device).unwrap();

        // Decode boxes
        let boxes = BlazeFace::decode_boxes(&input, &anchors, &config).unwrap();

        assert_eq!(boxes.dims(), &[batch_size, 896, 16]);
    }
}
