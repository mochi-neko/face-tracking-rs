// Reference implementation:
// https://github.com/hollance/BlazeFace-PyTorch/blob/master/blazeface.py

use candle_core::{Module, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig};

pub(crate) struct BlazeBlock {
    pub(crate) stride: usize,
    pub(crate) channel_pad: usize,
    pub(crate) conv1: Conv2d,
    pub(crate) conv2: Conv2d,
}

pub(crate) struct Conv2dParameters {
    pub(crate) weight: Tensor,
    pub(crate) bias: Option<Tensor>,
}

impl BlazeBlock {
    pub(crate) fn load(
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
                    dilation: 1, // Default
                    groups: in_channels,
                },
            ),
            conv2: Conv2d::new(
                conv2_parameters.weight,
                conv2_parameters.bias,
                Conv2dConfig {
                    padding: 0,  // Default
                    stride: 1,   // Default
                    dilation: 1, // Default
                    groups: 1,   // Default
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
            let h = xs.pad_with_zeros(2, 0, 2)?;
            h.pad_with_zeros(3, 0, 2)?
        } else {
            xs.clone()
        };

        let x = if self.stride == 2 {
            xs.max_pool2d_with_stride(self.stride, self.stride)?
        } else {
            xs.clone()
        };

        let x = if self.channel_pad > 0 {
            x.pad_with_zeros(
                1, // channel dimension
                0,
                self.channel_pad,
            )?
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
        let device = Device::Cpu;
        let batch_size = 1;
        let in_channels = 3;
        let out_channels = 16;
        let width = 64;
        let height = 64;
        let kernel_size = 3;
        let stride = 1;

        let conv1_weight = Tensor::rand(
            0.,
            1.,
            &[
                in_channels,
                1,
                kernel_size,
                kernel_size,
            ],
            &device,
        )
        .unwrap();
        let conv1_bias = Tensor::rand(0., 1., &[in_channels], &device).unwrap();

        let conv2_weight = Tensor::rand(
            0.,
            1.,
            &[
                out_channels,
                in_channels,
                1,
                1,
            ],
            &device,
        )
        .unwrap();
        let conv2_bias =
            Tensor::rand(0., 1., &[out_channels], &device).unwrap();

        let block = BlazeBlock::load(
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
        .unwrap();

        let x = block.forward(&input).unwrap();
        assert_eq!(
            x.dims(),
            &[
                batch_size,
                out_channels,
                width,
                height,
            ]
        );
    }
}
