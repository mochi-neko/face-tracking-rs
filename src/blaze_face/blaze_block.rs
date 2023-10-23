// Reference implementation:
// https://github.com/hollance/BlazeFace-PyTorch/blob/master/blazeface.py

use candle_core::{Module, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig};

use super::conv2d_parameters::Conv2dParameters;

pub(crate) struct BlazeBlock {
    stride: usize,
    channel_pad: usize,
    conv1: Conv2d,
    conv2: Conv2d,
}

impl BlazeBlock {
    pub(crate) fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        conv1_parameters: Conv2dParameters,
        conv2_parameters: Conv2dParameters,
    ) -> Result<BlazeBlock> {
        let padding = if stride == 2 {
            0
        } else {
            (kernel_size - 1) / 2
        };

        Ok(BlazeBlock {
            stride,
            channel_pad: out_channels - in_channels,
            conv1: Conv2d::new(
                conv1_parameters.weight,
                conv1_parameters.bias,
                Conv2dConfig {
                    padding,
                    stride,
                    groups: in_channels,
                    ..Default::default()
                },
            ),
            conv2: Conv2d::new(
                conv2_parameters.weight,
                conv2_parameters.bias,
                Conv2dConfig {
                    ..Default::default()
                },
            ),
        })
    }
}

impl Module for BlazeBlock {
    fn forward(
        &self,
        xs: &Tensor,
    ) -> Result<Tensor> {
        let h = if self.stride == 2 {
            xs.pad_with_zeros(2, 0, 2)? // height padding
                .pad_with_zeros(3, 0, 2)? // width padding
        } else {
            xs.clone()
        };

        let x = if self.stride == 2 {
            xs.max_pool2d_with_stride(self.stride, self.stride)? // max pooling
        } else {
            xs.clone()
        };

        let x = if self.channel_pad > 0 {
            x.pad_with_zeros(1, 0, self.channel_pad)? // channel padding
        } else {
            x
        };

        let h = self.conv1.forward(&h)?;
        let h = self.conv2.forward(&h)?;
        (h + x)?.relu()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn test_blaze_block() {
        // Set up the device
        let device = Device::Cpu;

        // Set up the configuration
        let batch_size = 1;
        let in_channels = 24;
        let out_channels = 24;
        let width = 64;
        let height = 64;
        let kernel_size = 3;
        let stride = 1;

        let conv1_weight = Tensor::rand(
            0.,
            1.,
            (in_channels, 1, kernel_size, kernel_size),
            &device,
        )
        .unwrap();
        let conv1_bias = Tensor::rand(0., 1., (in_channels,), &device).unwrap();

        let conv2_weight = Tensor::rand(
            0.,
            1.,
            (out_channels, in_channels, 1, 1),
            &device,
        )
        .unwrap();
        let conv2_bias =
            Tensor::rand(0., 1., (out_channels,), &device).unwrap();

        // Instantiate the BlazeBlock
        let model = BlazeBlock::new(
            in_channels,
            out_channels,
            kernel_size,
            stride,
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

        // Set up the input tensor
        let input = Tensor::rand(
            0.,
            1.,
            &[
                batch_size,
                in_channels,
                width,
                height,
            ],
            &device,
        )
        .unwrap(); // (1, 24, 64, 64)

        // Call forward method and get the output
        let output = model.forward(&input).unwrap(); // (1, 24, 64, 64)
        assert_eq!(
            output.dims(),
            &[
                batch_size,
                out_channels,
                width,
                height,
            ]
        );
    }

    #[test]
    fn test_blaze_block_for_stride_2() {
        // Set up the device
        let device = Device::Cpu;

        // Set up the configuration
        let batch_size = 1;
        let in_channels = 24;
        let out_channels = 28;
        let width = 64;
        let height = 64;
        let kernel_size = 3;
        let stride = 2;

        let conv1_weight = Tensor::rand(
            0.,
            1.,
            (in_channels, 1, kernel_size, kernel_size),
            &device,
        )
        .unwrap();
        let conv1_bias = Tensor::rand(0., 1., (in_channels,), &device).unwrap();

        let conv2_weight = Tensor::rand(
            0.,
            1.,
            (out_channels, in_channels, 1, 1),
            &device,
        )
        .unwrap();
        let conv2_bias =
            Tensor::rand(0., 1., (out_channels,), &device).unwrap();

        // Instantiate the BlazeBlock
        let block = BlazeBlock::new(
            in_channels,
            out_channels,
            kernel_size,
            stride,
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

        // Set up the input tensor
        let input = Tensor::rand(
            0.,
            1.,
            (batch_size, in_channels, width, height),
            &device,
        )
        .unwrap(); // (1, 24, 64, 64)

        // Call forward method and get the output
        let output = block.forward(&input).unwrap(); // (1, 28, 32, 32)
        assert_eq!(
            output.dims(),
            &[
                batch_size,
                out_channels,
                width / 2,  // stride = 2
                height / 2, // stride = 2
            ]
        );
    }
}
