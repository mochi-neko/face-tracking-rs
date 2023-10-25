// Reference implementation:
// https://github.com/hollance/BlazeFace-PyTorch/blob/master/blazeface.py

use candle_core::{Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Module, VarBuilder};

use super::{
    blaze_block::BlazeBlock, blaze_face::BlazeFaceModel,
    conv2d_parameters::Conv2dParameters, final_blaze_block::FinalBlazeBlock,
};

pub(crate) struct BlazeFaceBackModel {
    head: Conv2d,
    backbone: Vec<BlazeBlock>,
    final_block: FinalBlazeBlock,
    classifier_8: Conv2d,
    classifier_16: Conv2d,
    regressor_8: Conv2d,
    regressor_16: Conv2d,
}

impl BlazeFaceBackModel {
    pub(crate) fn load(variables: VarBuilder) -> Result<Self> {
        let head = Conv2d::new(
            variables.get_with_hints(
                (24, 3, 5, 5),
                "backbone.0.weight",
                candle_nn::init::ZERO,
            )?,
            Some(variables.get_with_hints(
                (24,),
                "backbone.0.bias",
                candle_nn::init::ZERO,
            )?),
            Conv2dConfig {
                padding: 0,
                stride: 2,
                dilation: 1,
                groups: 1,
            },
        );

        let backbone = vec![
            BlazeBlock::new(
                24,
                24,
                3,
                1,
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (24, 1, 3, 3),
                        "backbone.2.convs.0.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (24,),
                        "backbone.2.convs.0.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (24, 24, 1, 1),
                        "backbone.2.convs.1.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (24,),
                        "backbone.2.convs.1.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
            )?,
            BlazeBlock::new(
                24,
                24,
                3,
                1,
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (24, 1, 3, 3),
                        "backbone.3.convs.0.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (24,),
                        "backbone.3.convs.0.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (24, 24, 1, 1),
                        "backbone.3.convs.1.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (24,),
                        "backbone.3.convs.1.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
            )?,
            BlazeBlock::new(
                24,
                24,
                3,
                1,
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (24, 1, 3, 3),
                        "backbone.4.convs.0.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (24,),
                        "backbone.4.convs.0.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (24, 24, 1, 1),
                        "backbone.4.convs.1.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (24,),
                        "backbone.4.convs.1.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
            )?,
            BlazeBlock::new(
                24,
                24,
                3,
                1,
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (24, 1, 3, 3),
                        "backbone.5.convs.0.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (24,),
                        "backbone.5.convs.0.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (24, 24, 1, 1),
                        "backbone.5.convs.1.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (24,),
                        "backbone.5.convs.1.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
            )?,
            BlazeBlock::new(
                24,
                24,
                3,
                1,
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (24, 1, 3, 3),
                        "backbone.6.convs.0.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (24,),
                        "backbone.6.convs.0.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (24, 24, 1, 1),
                        "backbone.6.convs.1.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (24,),
                        "backbone.6.convs.1.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
            )?,
            BlazeBlock::new(
                24,
                24,
                3,
                1,
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (24, 1, 3, 3),
                        "backbone.7.convs.0.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (24,),
                        "backbone.7.convs.0.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (24, 24, 1, 1),
                        "backbone.7.convs.1.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (24,),
                        "backbone.7.convs.1.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
            )?,
            BlazeBlock::new(
                24,
                24,
                3,
                1,
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (24, 1, 3, 3),
                        "backbone.8.convs.0.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (24,),
                        "backbone.8.convs.0.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (24, 24, 1, 1),
                        "backbone.8.convs.1.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (24,),
                        "backbone.8.convs.1.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
            )?,
            BlazeBlock::new(
                24,
                24,
                3,
                2, // stride = 2
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (24, 1, 3, 3),
                        "backbone.9.convs.0.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (24,),
                        "backbone.9.convs.0.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (24, 24, 1, 1),
                        "backbone.9.convs.1.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (24,),
                        "backbone.9.convs.1.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
            )?,
            BlazeBlock::new(
                24,
                24,
                3,
                1,
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (24, 1, 3, 3),
                        "backbone.10.convs.0.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (24,),
                        "backbone.10.convs.0.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (24, 24, 1, 1),
                        "backbone.10.convs.1.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (24,),
                        "backbone.10.convs.1.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
            )?,
            BlazeBlock::new(
                24,
                24,
                3,
                1,
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (24, 1, 3, 3),
                        "backbone.11.convs.0.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (24,),
                        "backbone.11.convs.0.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (24, 24, 1, 1),
                        "backbone.11.convs.1.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (24,),
                        "backbone.11.convs.1.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
            )?,
            BlazeBlock::new(
                24,
                24,
                3,
                1,
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (24, 1, 3, 3),
                        "backbone.12.convs.0.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (24,),
                        "backbone.12.convs.0.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (24, 24, 1, 1),
                        "backbone.12.convs.1.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (24,),
                        "backbone.12.convs.1.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
            )?,
            BlazeBlock::new(
                24,
                24,
                3,
                1,
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (24, 1, 3, 3),
                        "backbone.13.convs.0.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (24,),
                        "backbone.13.convs.0.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (24, 24, 1, 1),
                        "backbone.13.convs.1.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (24,),
                        "backbone.13.convs.1.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
            )?,
            BlazeBlock::new(
                24,
                24,
                3,
                1,
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (24, 1, 3, 3),
                        "backbone.14.convs.0.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (24,),
                        "backbone.14.convs.0.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (24, 24, 1, 1),
                        "backbone.14.convs.1.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (24,),
                        "backbone.14.convs.1.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
            )?,
            BlazeBlock::new(
                24,
                24,
                3,
                1,
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (24, 1, 3, 3),
                        "backbone.15.convs.0.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (24,),
                        "backbone.15.convs.0.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (24, 24, 1, 1),
                        "backbone.15.convs.1.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (24,),
                        "backbone.15.convs.1.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
            )?,
            BlazeBlock::new(
                24,
                24,
                3,
                1,
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (24, 1, 3, 3),
                        "backbone.16.convs.0.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (24,),
                        "backbone.16.convs.0.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (24, 24, 1, 1),
                        "backbone.16.convs.1.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (24,),
                        "backbone.16.convs.1.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
            )?,
            BlazeBlock::new(
                24,
                48,
                3,
                2, // stride = 2
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (24, 1, 3, 3),
                        "backbone.17.convs.0.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (24,),
                        "backbone.17.convs.0.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (48, 24, 1, 1),
                        "backbone.17.convs.1.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (48,),
                        "backbone.17.convs.1.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
            )?,
            BlazeBlock::new(
                48,
                48,
                3,
                1,
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (48, 1, 3, 3),
                        "backbone.18.convs.0.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (48,),
                        "backbone.18.convs.0.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (48, 48, 1, 1),
                        "backbone.18.convs.1.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (48,),
                        "backbone.18.convs.1.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
            )?,
            BlazeBlock::new(
                48,
                48,
                3,
                1,
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (48, 1, 3, 3),
                        "backbone.19.convs.0.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (48,),
                        "backbone.19.convs.0.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (48, 48, 1, 1),
                        "backbone.19.convs.1.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (48,),
                        "backbone.19.convs.1.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
            )?,
            BlazeBlock::new(
                48,
                48,
                3,
                1,
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (48, 1, 3, 3),
                        "backbone.20.convs.0.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (48,),
                        "backbone.20.convs.0.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (48, 48, 1, 1),
                        "backbone.20.convs.1.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (48,),
                        "backbone.20.convs.1.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
            )?,
            BlazeBlock::new(
                48,
                48,
                3,
                1,
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (48, 1, 3, 3),
                        "backbone.21.convs.0.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (48,),
                        "backbone.21.convs.0.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (48, 48, 1, 1),
                        "backbone.21.convs.1.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (48,),
                        "backbone.21.convs.1.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
            )?,
            BlazeBlock::new(
                48,
                48,
                3,
                1,
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (48, 1, 3, 3),
                        "backbone.22.convs.0.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (48,),
                        "backbone.22.convs.0.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (48, 48, 1, 1),
                        "backbone.22.convs.1.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (48,),
                        "backbone.22.convs.1.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
            )?,
            BlazeBlock::new(
                48,
                48,
                3,
                1,
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (48, 1, 3, 3),
                        "backbone.23.convs.0.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (48,),
                        "backbone.23.convs.0.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (48, 48, 1, 1),
                        "backbone.23.convs.1.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (48,),
                        "backbone.23.convs.1.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
            )?,
            BlazeBlock::new(
                48,
                48,
                3,
                1,
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (48, 1, 3, 3),
                        "backbone.24.convs.0.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (48,),
                        "backbone.24.convs.0.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (48, 48, 1, 1),
                        "backbone.24.convs.1.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (48,),
                        "backbone.24.convs.1.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
            )?,
            BlazeBlock::new(
                48,
                96,
                3,
                2, // stride = 2
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (48, 1, 3, 3),
                        "backbone.25.convs.0.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (48,),
                        "backbone.25.convs.0.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (96, 48, 1, 1),
                        "backbone.25.convs.1.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (96,),
                        "backbone.25.convs.1.bias",
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
                        "backbone.26.convs.0.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (96,),
                        "backbone.26.convs.0.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (96, 96, 1, 1),
                        "backbone.26.convs.1.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (96,),
                        "backbone.26.convs.1.bias",
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
                        "backbone.27.convs.0.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (96,),
                        "backbone.27.convs.0.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (96, 96, 1, 1),
                        "backbone.27.convs.1.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (96,),
                        "backbone.27.convs.1.bias",
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
                        "backbone.28.convs.0.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (96,),
                        "backbone.28.convs.0.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (96, 96, 1, 1),
                        "backbone.28.convs.1.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (96,),
                        "backbone.28.convs.1.bias",
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
                        "backbone.29.convs.0.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (96,),
                        "backbone.29.convs.0.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (96, 96, 1, 1),
                        "backbone.29.convs.1.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (96,),
                        "backbone.29.convs.1.bias",
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
                        "backbone.30.convs.0.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (96,),
                        "backbone.30.convs.0.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (96, 96, 1, 1),
                        "backbone.30.convs.1.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (96,),
                        "backbone.30.convs.1.bias",
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
                        "backbone.31.convs.0.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (96,),
                        "backbone.31.convs.0.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (96, 96, 1, 1),
                        "backbone.31.convs.1.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (96,),
                        "backbone.31.convs.1.bias",
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
                        "backbone.32.convs.0.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (96,),
                        "backbone.32.convs.0.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
                Conv2dParameters {
                    weight: variables.get_with_hints(
                        (96, 96, 1, 1),
                        "backbone.32.convs.1.weight",
                        candle_nn::init::ZERO,
                    )?,
                    bias: Some(variables.get_with_hints(
                        (96,),
                        "backbone.32.convs.1.bias",
                        candle_nn::init::ZERO,
                    )?),
                },
            )?,
        ];

        let final_block = FinalBlazeBlock::new(
            96,
            Conv2dParameters {
                weight: variables.get_with_hints(
                    (96, 1, 3, 3),
                    "final.convs.0.weight",
                    candle_nn::init::ZERO,
                )?,
                bias: Some(variables.get_with_hints(
                    (96,),
                    "final.convs.0.bias",
                    candle_nn::init::ZERO,
                )?),
            },
            Conv2dParameters {
                weight: variables.get_with_hints(
                    (96, 96, 1, 1),
                    "final.convs.1.weight",
                    candle_nn::init::ZERO,
                )?,
                bias: Some(variables.get_with_hints(
                    (96,),
                    "final.convs.1.bias",
                    candle_nn::init::ZERO,
                )?),
            },
        )?;

        let classifier_8 = Conv2d::new(
            variables.get_with_hints(
                (2, 96, 1, 1),
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
                (32, 96, 1, 1),
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

        Ok(Self {
            head,
            backbone,
            final_block,
            classifier_8,
            classifier_16,
            regressor_8,
            regressor_16,
        })
    }

    fn forward_backbone(
        &self,
        x: &Tensor,
    ) -> Result<Tensor> {
        let mut x = x.clone();
        for block in &self.backbone {
            x = block.forward(&x)?;
        }
        Ok(x)
    }
}

impl BlazeFaceModel for BlazeFaceBackModel {
    fn forward(
        &self,
        xs: &Tensor, // (batch, 3, 256, 256)
    ) -> Result<(Tensor, Tensor)> // score:(batch, 896, 1), boxes:(batch, 896, 16)
    {
        let x = xs
            .pad_with_zeros(2, 1, 2)? // height padding
            .pad_with_zeros(3, 1, 2)?; // width padding

        let batch_size = x.dims()[0];

        let x = self.head.forward(&x)?; // (batch, 24, 128, 128)
        let x = x.relu()?;
        let x = self.forward_backbone(&x)?; // (batch, 96, 16, 16)

        let h = self.final_block.forward(&x)?; // (batch, 96, 8, 8)

        let c1 = self
            .classifier_8
            .forward(&x)?; // (batch, 2, 16, 16)
        let c1 = c1.permute((0, 2, 3, 1))?; // (batch, 16, 16, 2)
        let c1 = c1.reshape((batch_size, 512, 1))?; // (batch, 512, 1)

        let c2 = self
            .classifier_16
            .forward(&h)?; // (batch, 6, 8, 8)
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

        // Load the variables
        let variables = candle_nn::VarBuilder::from_pth(
            "src/blaze_face/data/blazefaceback.pth",
            dtype,
            &device,
        )
        .unwrap();

        // Load the model
        let model = BlazeFaceBackModel::load(variables).unwrap();

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
}
