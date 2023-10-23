// Reference implementation:
// https://github.com/hollance/BlazeFace-PyTorch/blob/master/blazeface.py

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use super::{
    blaze_face_back_model::BlazeFaceBackModel,
    blaze_face_config::BlazeFaceConfig,
    blaze_face_front_model::BlazeFaceFrontModel,
};

pub(crate) enum ModelType {
    Back,
    Front,
}

pub(crate) trait BlazeFaceModel {
    fn forward(
        &self,
        xs: &Tensor,
    ) -> Result<(Tensor, Tensor)>;
}

pub(crate) struct BlazeFace {
    model: Box<dyn BlazeFaceModel>,
    anchors: Tensor,
    config: BlazeFaceConfig,
}

impl BlazeFace {
    pub(crate) fn load(
        model_type: ModelType,
        variables: VarBuilder,
        anchors: Tensor,
        config: BlazeFaceConfig,
    ) -> Result<BlazeFace> {
        match model_type {
            | ModelType::Back => {
                let model = BlazeFaceBackModel::load(variables)?;
                Ok(BlazeFace {
                    model: Box::new(model),
                    anchors,
                    config,
                })
            },
            | ModelType::Front => {
                let model = BlazeFaceFrontModel::load(variables)?;
                Ok(BlazeFace {
                    model: Box::new(model),
                    anchors,
                    config,
                })
            },
        }
    }

    pub(crate) fn forward(
        &self,
        xs: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        self.model.forward(xs)
    }

    fn convert_tensors_to_detections(
        &self,
        raw_boxes: &Tensor,
        raw_scores: &Tensor,
    ) -> Result<Vec<Tensor>> {
        unimplemented!()
    }

    fn decode_boxes(
        &self,
        raw_boxes: &Tensor,
    ) -> Result<Tensor> {
        unimplemented!()
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
            BlazeFaceConfig::back(100., 0.65, 0.3),
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
            BlazeFaceConfig::back(100., 0.75, 0.3),
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
}
