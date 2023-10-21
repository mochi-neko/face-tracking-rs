// Reference implementation:
// https://github.com/hollance/BlazeFace-PyTorch/blob/master/blazeface.py

use candle_core::{Module, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig};

use super::conv2d_parameters::Conv2dParameters;

pub(crate) struct FinalBlazeBlock {
    pub(crate) conv1: Conv2d,
    pub(crate) conv2: Conv2d,
}

impl FinalBlazeBlock {
    pub(crate) fn new(
        channels: usize,
        conv1_parameters: Conv2dParameters,
        conv2_parameters: Conv2dParameters,
    ) -> Result<FinalBlazeBlock> {
        Ok(FinalBlazeBlock {
            conv1: Conv2d::new(
                conv1_parameters.weight,
                conv1_parameters.bias,
                Conv2dConfig {
                    padding: 0,
                    stride: 2,
                    dilation: 1,
                    groups: channels,
                },
            ),
            conv2: Conv2d::new(
                conv2_parameters.weight,
                conv2_parameters.bias,
                Conv2dConfig {
                    padding: 0,
                    stride: 1,
                    dilation: 1,
                    groups: 1,
                },
            ),
        })
    }
}

impl Module for FinalBlazeBlock {
    fn forward(
        &self,
        xs: &Tensor,
    ) -> Result<Tensor> {
        let h = xs.pad_with_zeros(2, 0, 2)?;
        let h = h.pad_with_zeros(3, 0, 2)?;

        let x = self.conv1.forward(&h)?;
        let x = self.conv2.forward(&x)?;
        x.relu()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn test_final_blaze_block() {
        // Set up the device
        let device = Device::Cpu;

        // Set up the configuration
        let batch_size = 1;
        let channels = 3;
        let width = 64;
        let height = 64;

        // Set up the convolution parameters
        let conv1_weight =
            Tensor::rand(0., 1., &[channels, 1, 3, 3], &device).unwrap();
        let conv1_bias = Tensor::rand(0., 1., &[channels], &device).unwrap();

        let conv2_weight = Tensor::rand(
            0.,
            1.,
            &[
                channels * 2,
                channels,
                1,
                1,
            ],
            &device,
        )
        .unwrap();
        let conv2_bias =
            Tensor::rand(0., 1., &[channels * 2], &device).unwrap();

        // Instantiate the FinalBlazeBlock
        let block = FinalBlazeBlock::new(
            channels,
            Conv2dParameters {
                weight: conv1_weight.clone(),
                bias: Some(conv1_bias.clone()),
            },
            Conv2dParameters {
                weight: conv2_weight.clone(),
                bias: Some(conv2_bias.clone()),
            },
        )
        .unwrap();

        // Set up the input Tensor
        let input = Tensor::rand(
            0.,
            1.,
            &[
                batch_size, channels, width, height,
            ],
            &device,
        )
        .unwrap();

        // Call forward method and get the output
        let output = block.forward(&input).unwrap();

        // Verify the output dimensions
        // Assuming that the output channels are twice the input channels, and the width and height are halved due to stride of 2
        assert_eq!(
            output.dims(),
            &[
                batch_size,
                channels * 2,
                width / 2,
                height / 2
            ]
        );
    }
}
