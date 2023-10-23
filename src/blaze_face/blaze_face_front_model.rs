// Reference implementation:
// https://github.com/hollance/BlazeFace-PyTorch/blob/master/blazeface.py

use candle_core::{backend, Module, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, VarBuilder};

use super::{
    blaze_block::BlazeBlock, blaze_face::BlazeFaceModel,
    blaze_face_config::BlazeFaceConfig, conv2d_parameters::Conv2dParameters,
};

pub(crate) struct BlazeFaceFrontModel {
    pub(crate) config: BlazeFaceConfig,
    head: Conv2d,
    backbone_1: Vec<BlazeBlock>,
    backbone_2: Vec<BlazeBlock>,
    classifier_8: Conv2d,
    classifier_16: Conv2d,
    regressor_8: Conv2d,
    regressor_16: Conv2d,
}

impl BlazeFaceFrontModel {
    pub(crate) fn load(variables: VarBuilder) -> Result<BlazeFaceFrontModel> {
        let config = BlazeFaceConfig {
            num_classes: 1,
            num_anchors: 896,
            num_coords: 16,
            score_clipping_thresh: 100.0,
            x_scale: 128.0,
            y_scale: 128.0,
            h_scale: 128.0,
            w_scale: 128.0,
            min_score_thresh: 0.75,
            min_suppression_threshold: 0.3,
        };

        let head = Conv2d::new(
            variables.get_with_hints(
                (24, 3, 5, 5),
                "backbone1.0.weight",
                candle_nn::init::ZERO,
            )?,
            Some(variables.get_with_hints(
                (24,),
                "backbone1.0.bias",
                candle_nn::init::ZERO,
            )?),
            Conv2dConfig {
                padding: 0,
                stride: 2,
                dilation: 1,
                groups: 1,
            },
        );

        let backbone_1 = vec![
            BlazeBlock::new(
                24,
                24,
                3,
                1,
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (24, 1, 3, 3),
                        "backbone1.2.convs.0.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (24,),
                        "backbone1.2.convs.0.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (24, 24, 1, 1),
                        "backbone1.2.convs.1.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (24,),
                        "backbone1.2.convs.1.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
            )?,
            BlazeBlock::new(
                24,
                28,
                3,
                1,
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (24, 1, 3, 3),
                        "backbone1.3.convs.0.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (24,),
                        "backbone1.3.convs.0.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (28, 24, 1, 1),
                        "backbone1.3.convs.1.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (28,),
                        "backbone1.3.convs.1.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
            )?,
            BlazeBlock::new(
                28,
                32,
                3,
                2, // stride = 2
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (28, 1, 3, 3),
                        "backbone1.4.convs.0.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (28,),
                        "backbone1.4.convs.0.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (32, 28, 1, 1),
                        "backbone1.4.convs.1.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (32,),
                        "backbone1.4.convs.1.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
            )?,
            BlazeBlock::new(
                32,
                36,
                3,
                1,
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (32, 1, 3, 3),
                        "backbone1.5.convs.0.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (32,),
                        "backbone1.5.convs.0.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (36, 32, 1, 1),
                        "backbone1.5.convs.1.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (36,),
                        "backbone1.5.convs.1.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
            )?,
            BlazeBlock::new(
                36,
                42,
                3,
                1,
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (36, 1, 3, 3),
                        "backbone1.6.convs.0.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (36,),
                        "backbone1.6.convs.0.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (42, 36, 1, 1),
                        "backbone1.6.convs.1.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (42,),
                        "backbone1.6.convs.1.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
            )?,
            BlazeBlock::new(
                42,
                48,
                3,
                2, // stride = 2
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (42, 1, 3, 3),
                        "backbone1.7.convs.0.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (42,),
                        "backbone1.7.convs.0.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (48, 42, 1, 1),
                        "backbone1.7.convs.1.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (48,),
                        "backbone1.7.convs.1.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
            )?,
            BlazeBlock::new(
                48,
                56,
                3,
                1,
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (48, 1, 3, 3),
                        "backbone1.8.convs.0.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (48,),
                        "backbone1.8.convs.0.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (56, 48, 1, 1),
                        "backbone1.8.convs.1.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (56,),
                        "backbone1.8.convs.1.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
            )?,
            BlazeBlock::new(
                56,
                64,
                3,
                1,
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (56, 1, 3, 3),
                        "backbone1.9.convs.0.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (56,),
                        "backbone1.9.convs.0.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (64, 56, 1, 1),
                        "backbone1.9.convs.1.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (64,),
                        "backbone1.9.convs.1.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
            )?,
            BlazeBlock::new(
                64,
                72,
                3,
                1,
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (64, 1, 3, 3),
                        "backbone1.10.convs.0.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (64,),
                        "backbone1.10.convs.0.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (72, 64, 1, 1),
                        "backbone1.10.convs.1.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (72,),
                        "backbone1.10.convs.1.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
            )?,
            BlazeBlock::new(
                72,
                80,
                3,
                1,
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (72, 1, 3, 3),
                        "backbone1.11.convs.0.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (72,),
                        "backbone1.11.convs.0.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (80, 72, 1, 1),
                        "backbone1.11.convs.1.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (80,),
                        "backbone1.11.convs.1.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
            )?,
            BlazeBlock::new(
                80,
                88,
                3,
                1,
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (80, 1, 3, 3),
                        "backbone1.12.convs.0.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (80,),
                        "backbone1.12.convs.0.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (88, 80, 1, 1),
                        "backbone1.12.convs.1.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (88,),
                        "backbone1.12.convs.1.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
            )?,
        ];

        let backbone_2 = vec![
            BlazeBlock::new(
                88,
                96,
                3,
                2, // stride = 2
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (88, 1, 3, 3),
                        "backbone2.0.convs.0.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (88,),
                        "backbone2.0.convs.0.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (96, 88, 1, 1),
                        "backbone2.0.convs.1.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (96,),
                        "backbone2.0.convs.1.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
            )?,
            BlazeBlock::new(
                96,
                96,
                3,
                1,
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (96, 1, 3, 3),
                        "backbone2.1.convs.0.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (96,),
                        "backbone2.1.convs.0.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (96, 96, 1, 1),
                        "backbone2.1.convs.1.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (96,),
                        "backbone2.1.convs.1.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
            )?,
            BlazeBlock::new(
                96,
                96,
                3,
                1,
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (96, 1, 3, 3),
                        "backbone2.2.convs.0.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (96,),
                        "backbone2.2.convs.0.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (96, 96, 1, 1),
                        "backbone2.2.convs.1.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (96,),
                        "backbone2.2.convs.1.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
            )?,
            BlazeBlock::new(
                96,
                96,
                3,
                1,
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (96, 1, 3, 3),
                        "backbone2.3.convs.0.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (96,),
                        "backbone2.3.convs.0.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (96, 96, 1, 1),
                        "backbone2.3.convs.1.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (96,),
                        "backbone2.3.convs.1.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
            )?,
            BlazeBlock::new(
                96,
                96,
                3,
                1,
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (96, 1, 3, 3),
                        "backbone2.4.convs.0.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (96,),
                        "backbone2.4.convs.0.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (96, 96, 1, 1),
                        "backbone2.4.convs.1.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (96,),
                        "backbone2.4.convs.1.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
            )?,
        ];

        let classifier_8 = Conv2d::new(
            variables.get_with_hints(
                (2, 88, 1, 1),
                "classifier_8.weight",
                candle_nn::init::ZERO,
            )?,
            Some(variables.get_with_hints(
                (2,),
                "classifier_8.bias",
                candle_nn::init::ZERO,
            )?),
            Conv2dConfig {
                ..Default::default()
            },
        );

        let classifier_16 = Conv2d::new(
            variables.get_with_hints(
                (6, 96, 1, 1),
                "classifier_16.weight",
                candle_nn::init::ZERO,
            )?,
            Some(variables.get_with_hints(
                (6,),
                "classifier_16.bias",
                candle_nn::init::ZERO,
            )?),
            Conv2dConfig {
                ..Default::default()
            },
        );

        let regressor_8 = Conv2d::new(
            variables.get_with_hints(
                (32, 88, 1, 1),
                "regressor_8.weight",
                candle_nn::init::ZERO,
            )?,
            Some(variables.get_with_hints(
                (32,),
                "regressor_8.bias",
                candle_nn::init::ZERO,
            )?),
            Conv2dConfig {
                ..Default::default()
            },
        );

        let regressor_16 = Conv2d::new(
            variables.get_with_hints(
                (96, 96, 1, 1),
                "regressor_16.weight",
                candle_nn::init::ZERO,
            )?,
            Some(variables.get_with_hints(
                (96,),
                "regressor_16.bias",
                candle_nn::init::ZERO,
            )?),
            Conv2dConfig {
                ..Default::default()
            },
        );

        Ok(BlazeFaceFrontModel {
            config,
            head,
            backbone_1,
            backbone_2,
            classifier_8,
            classifier_16,
            regressor_8,
            regressor_16,
        })
    }

    fn forward_backbone_1(
        &self,
        x: &Tensor,
    ) -> Result<Tensor> {
        let mut x = x.clone();
        for block in &self.backbone_1 {
            x = block.forward(&x)?;
        }
        Ok(x)
    }

    fn forward_backbone_2(
        &self,
        x: &Tensor,
    ) -> Result<Tensor> {
        let mut x = x.clone();
        for block in &self.backbone_2 {
            x = block.forward(&x)?;
        }
        Ok(x)
    }
}

impl BlazeFaceModel for BlazeFaceFrontModel {
    fn forward(
        &self,
        xs: &Tensor, // (batch, 3, 128, 128)
    ) -> Result<(Tensor, Tensor)> {
        let x = xs
            .pad_with_zeros(2, 1, 2)? // height padding
            .pad_with_zeros(3, 1, 2)?; // width padding

        let batch_size = x.dims()[0];

        let x = self.head.forward(&x)?; // (batch, 24, 64, 64)
        let x = x.relu()?;
        let x = self.forward_backbone_1(&x)?; // (batch, 88, 16, 16)

        let h = self.forward_backbone_2(&x)?; // (batch, 96, 8, 8)

        let c1 = self
            .classifier_8
            .forward(&x)?; // (batch, 2, 16, 16)
        let c1 = c1.permute((0, 2, 3, 1))?; // (batch, 16, 16, 2)
        let c1 = c1.reshape((batch_size, 512, 1))?; // (batch, 512, 1)

        let c2 = self
            .classifier_16
            .forward(&h)?; // # (batch, 6, 8, 8)
        let c2 = c2.permute((0, 2, 3, 1))?; // (batch, 8, 8, 6)
        let c2 = c2.reshape((batch_size, 384, 1))?; // (batch, 384, 1)

        let c = Tensor::cat(&[c1, c2], 1)?; // (batch, 896, 1)

        let r1 = self.regressor_8.forward(&x)?; // (batch, 32, 16, 16)
        let r1 = r1.permute((0, 2, 3, 1))?; // (batch, 16, 16, 32)
        let r1 = r1.reshape((batch_size, 512, 16))?; // (batch, 512, 16)

        let r2 = self
            .regressor_16
            .forward(&h)?; // (batch, 96, 8, 8)
        let r2 = r2.permute((0, 2, 3, 1))?; // (batch, 8, 8, 96)
        let r2 = r2.reshape((batch_size, 384, 16))?; // (batch, 384, 16)

        let r = Tensor::cat(&[r1, r2], 1)?; // (batch, 896, 16)

        Ok((c, r))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn test_forward() {
        // Set up the device and dtype
        let device = Device::Cpu;
        let dtype = DType::F16;
        let batch_size = 1;

        // FIXME: Downloaded .pth file is broken from https://github.com/hollance/BlazeFace-PyTorch/blob/master/blazeface.pth
        // Load the variables
        let variables = candle_nn::VarBuilder::from_pth(
            "src/blaze_face/blazeface.pth",
            dtype,
            &device,
        )
        .unwrap();

        // Load the model
        let model = BlazeFaceFrontModel::load(variables).unwrap();

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
