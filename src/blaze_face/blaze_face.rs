// Reference implementation:
// https://github.com/hollance/BlazeFace-PyTorch/blob/master/blazeface.py

use candle_core::{DType, Error, IndexOp, Result, Shape, Tensor};
use candle_nn::{ops, VarBuilder};

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
        variables: &VarBuilder,
        anchors: Tensor,
        score_clipping_thresh: f32,
        min_score_thresh: f32,
        min_suppression_threshold: f32,
    ) -> Result<Self> {
        let device = variables.device();
        if !device.same_device(anchors.device()) {
            return Result::Err(Error::DeviceMismatchBinaryOp {
                lhs: device.location(),
                rhs: anchors.device().location(),
                op: "load_blaze_face",
            });
        }
        if anchors.dims() != [896, 4] {
            return Result::Err(Error::ShapeMismatchBinaryOp {
                lhs: anchors.shape().clone(),
                rhs: Shape::from_dims(&[896, 4]),
                op: "load_blaze_face",
            });
        }
        let anchors = anchors.to_dtype(DType::F32)?;

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
                        device,
                    )?,
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
                        device,
                    )?,
                })
            },
        }
    }

    fn forward(
        &self,
        images: &Tensor, // back:(batch_size, 3, 256, 256) or front:(batch_size, 3, 128, 128)
    ) -> Result<(Tensor, Tensor)> // coordinates:(batch, 896, 16), score:(batch, 896, 1)
    {
        self.model.forward(images)
    }

    pub fn predict_on_batch(
        &self,
        images: &Tensor, // (batch_size, 3, 256, 256) or (batch_size, 3, 128, 128)
    ) -> Result<Vec<Vec<Tensor>>> // Vec<(detected_faces, 17)> with length:batch_size
    {
        let (raw_boxes, raw_scores) = self.forward(images)?; // coordinates:(batch, 896, 16), score:(batch, 896, 1)

        let detections = tensors_to_detections(
            &raw_boxes,
            &raw_scores,
            &self.anchors,
            &self.config,
        )?; // Vec<(num_detections, 17)> with length:batch_size

        let mut filtered_detections = Vec::new();
        for detection in detections {
            let faces = weighted_non_max_suppression(
                &detection.contiguous()?,
                &self.config,
            )?; // Vec<(17)> with length:detected_faces
            if !faces.is_empty() {
                filtered_detections.push(faces);
            } else {
                let zeros = Tensor::zeros(
                    17,
                    detection.dtype(),
                    detection.device(),
                )?; // (17)
                filtered_detections.push(vec![zeros]);
            }
        }

        Ok(filtered_detections) // Vec<(detected_faces, 17)> with length:batch_size
    }
}

fn tensors_to_detections(
    raw_boxes: &Tensor,  // (batch_size, 896, 16)
    raw_scores: &Tensor, // (batch_size, 896, 1)
    anchors: &Tensor,    // (896, 4)
    config: &BlazeFaceConfig,
) -> Result<Vec<Tensor>> // Vec<(num_detections, 17)> with length:batch_size
{
    let detection_boxes = decode_boxes(raw_boxes, anchors, config)?; // (batch_size, 896, 16)

    raw_scores.clamp(
        -config.score_clipping_thresh,
        config.score_clipping_thresh,
    )?;

    let detection_scores = ops::sigmoid(raw_scores)?; // (batch_size, 896, 1)

    let indices = unmasked_indices(
        &detection_scores,
        config.min_score_thresh,
    )?; // (batch_size, num_detections)

    let mut output = Vec::new();
    for batch in 0..raw_boxes.dims()[0] {
        // Filtering
        let boxes = detection_boxes.i((batch, &indices.i((batch, ..))?, ..))?; // (num_detections, 16)
        let scores =
            detection_scores.i((batch, &indices.i((batch, ..))?, ..))?; // (num_detections, 1)

        if boxes.dims()[0] == 0 || scores.dims()[0] == 0 {
            output.push(Tensor::zeros(
                (0, 17),
                raw_boxes.dtype(),
                raw_boxes.device(),
            )?);
        } else {
            let detection = Tensor::cat(&[boxes, scores], 1)?; // (896, 17)
            output.push(detection);
        }
    }

    Ok(output) // Vec<(num_detections, 17)> with length:batch_size
}

fn decode_boxes(
    raw_boxes: &Tensor, // (batch_size, 896, 16)
    anchors: &Tensor,   // (896, 4)
    config: &BlazeFaceConfig,
) -> Result<Tensor> // (batch_size, 896, 16)
{
    let mut coordinates = Vec::new();
    let two = Tensor::from_slice(&[2_f32], 1, raw_boxes.device())?; // (1)

    let x_anchor = anchors.i((.., 0))?; // (896)
    let y_anchor = anchors.i((.., 1))?; // (896)
    let w_anchor = anchors.i((.., 2))?; // (896)
    let h_anchor = anchors.i((.., 3))?; // (896)

    let x_center = raw_boxes
        .i((.., .., 0))? // (batch_size, 896)
        .broadcast_div(&config.x_scale)? // / (1)
        .broadcast_mul(&w_anchor)? // * (896)
        .broadcast_add(&x_anchor)?; // + (896)
                                    // = (batch_size, 896)

    let y_center = raw_boxes
        .i((.., .., 1))?
        .broadcast_div(&config.y_scale)?
        .broadcast_mul(&h_anchor)?
        .broadcast_add(&y_anchor)?;

    let w = raw_boxes
        .i((.., .., 2))? // (batch_size, 896)
        .broadcast_div(&config.w_scale)? // / (1)
        .broadcast_mul(&w_anchor)?; // * (896)
                                    // = (batch_size, 896)

    let h = raw_boxes
        .i((.., .., 3))?
        .broadcast_div(&config.h_scale)?
        .broadcast_mul(&h_anchor)?;

    // Bounding box
    let x_min = (&x_center - w.broadcast_div(&two)?)?; // (batch_size, 896)
    let x_max = (&x_center + w.broadcast_div(&two)?)?;
    let y_min = (&y_center - h.broadcast_div(&two)?)?;
    let y_max = (&y_center + h.broadcast_div(&two)?)?;

    coordinates.push(y_min);
    coordinates.push(x_min);
    coordinates.push(y_max);
    coordinates.push(x_max);

    // Face keypoints: right_eye, left_eye, nose, mouth, right_ear, left_ear
    for k in 0..6 {
        let offset = 4 + k * 2; // 4 = bounding box, 2 = (x, y)

        let keypoint_x = raw_boxes
            .i((.., .., offset))? // (batch_size, 896)
            .broadcast_div(&config.x_scale)? // / (1)
            .broadcast_mul(&w_anchor)? // * (896)
            .broadcast_add(&x_anchor)?; // + (896)
                                        // = (batch_size, 896)

        let keypoint_y = raw_boxes
            .i((.., .., offset + 1))?
            .broadcast_div(&config.y_scale)?
            .broadcast_mul(&h_anchor)?
            .broadcast_add(&y_anchor)?;

        coordinates.push(keypoint_x);
        coordinates.push(keypoint_y);
    }

    Tensor::stack(&coordinates, 2) // (batch_size, 896, 16)
}

fn unmasked_indices(
    scores: &Tensor, // (batch_size, 896, 1)
    threshold: f32,
) -> Result<Tensor> // (batch_size, num_unmasked) of DType::U32
{
    let batch_size = scores.dims()[0];

    let mask = scores
        .ge(threshold)? // (batch_size, 896, 1) of Dtype::U8
        .squeeze(2)?; // (batch_size, 896) of Dtype::U8

    // Collect unmasked indices
    let mut indices = Vec::new();
    for batch in 0..batch_size {
        let mut batch_indices = Vec::new();
        let batch_mask = mask
            .i((batch, ..))? // (896)
            .to_vec1::<u8>()?;

        batch_mask
            .iter()
            .enumerate()
            .for_each(|(i, x)| {
                if *x == 1u8 {
                    batch_indices.push(i as u32);
                }
            });

        let batch_indices = Tensor::from_slice(
            &batch_indices,
            batch_indices.len(),
            scores.device(),
        )?; // (num_unmasked)

        indices.push(batch_indices);
    }

    Tensor::stack(&indices, 0) // (batch_size, num_unmasked)
}

fn argsort_by_score(
    detection: &Tensor, // (num_detections, 17)
) -> Result<Tensor> // (num_detections) of DType::U32
{
    let scores = detection
        .i((.., 16))? // (num_detections)
        .to_vec1::<f32>()?;

    let count = scores.len();

    // Create a vector of indices from 0 to num_detections - 1
    let mut indices: Vec<u32> = (0u32..count as u32).collect();

    // Sort the indices by descending order of scores
    indices.sort_unstable_by(|&a, &b| {
        let score_a = scores[a as usize];
        let score_b = scores[b as usize];

        // Reverse
        score_b
            .partial_cmp(&score_a)
            .unwrap()
    });

    Tensor::from_vec(indices, count, detection.device())
}

fn overlap_similarity(
    first_box: &Tensor, // (4)
    other_box: &Tensor, // (remainings, 4)
) -> Result<Tensor> // (remainings)
{
    let first_box = first_box.unsqueeze(0)?; // (1, 4)

    jaccard(&first_box, other_box)? // (1, remainings)
        .squeeze(0) // (remainings)
}

fn jaccard(
    box_a: &Tensor, // (a, 4)
    box_b: &Tensor, // (b, 4)
) -> Result<Tensor> // (a, b)
{
    let inter = intersect(box_a, box_b)?; // (a, b)

    let area_a = box_a
        .i((.., 2))?
        .sub(&box_a.i((.., 0))?)?
        .mul(
            &box_a
                .i((.., 3))?
                .sub(&box_a.i((.., 1))?)?,
        )?
        .unsqueeze(1)?
        .expand(inter.shape())?; // (a, b)

    let area_b = box_b
        .i((.., 2))?
        .sub(&box_b.i((.., 0))?)?
        .mul(
            &box_b
                .i((.., 3))?
                .sub(&box_b.i((.., 1))?)?,
        )?
        .unsqueeze(0)?
        .expand(inter.shape())?; // (a, b)

    let union = ((&area_a + &area_b)? - &inter)?; // (a, b)

    inter.div(&union) // (a, b)
}

fn intersect(
    box_a: &Tensor, // (a, 4)
    box_b: &Tensor, // (b, 4)
) -> Result<Tensor> // (a, b)
{
    let a = box_a.dims()[0];
    let b = box_b.dims()[0];

    let a_max_xy = box_a
        .i((.., 2..4))?
        .unsqueeze(1)?
        .expand(&[a, b, 2])?; // (a, b, 2)

    let b_max_xy = box_b
        .i((.., 2..4))?
        .unsqueeze(0)?
        .expand(&[a, b, 2])?; // (a, b, 2)

    let a_min_xy = box_a
        .i((.., 0..2))?
        .unsqueeze(1)?
        .expand(&[a, b, 2])?; // (a, b, 2)

    let b_min_xy = box_b
        .i((.., 0..2))?
        .unsqueeze(0)?
        .expand(&[a, b, 2])?; // (a, b, 2)

    let max_xy = Tensor::stack(&[a_max_xy, b_max_xy], 0)?.min(0)?; // (a, b, 2)
    let min_xy = Tensor::stack(&[a_min_xy, b_min_xy], 0)?.max(0)?; // (a, b, 2)
    let inter = Tensor::clamp(&(max_xy - min_xy)?, 0., f32::INFINITY)?; // (a, b, 2)

    inter
        .i((.., .., 0))?
        .mul(&inter.i((.., .., 1))?) // (a, b)
}

fn mask_indices(
    mask: &Tensor, // (masked_vector) of DType::U8
) -> Result<(Tensor, Tensor)> // (unmasked_indices), (masked_indices) of DType::U32
{
    let mut unmasked = Vec::new();
    let mut masked = Vec::new();
    for (i, x) in mask
        .to_vec1::<u8>()?
        .iter()
        .enumerate()
    {
        if *x == 1u8 {
            unmasked.push(i as u32);
        } else {
            masked.push(i as u32);
        }
    }

    let unmasked =
        Tensor::from_slice(&unmasked, unmasked.len(), mask.device())?;

    let masked = Tensor::from_slice(&masked, masked.len(), mask.device())?;

    Ok((unmasked, masked))
}

fn weighted_non_max_suppression(
    detections: &Tensor, // (num_detections, 17)
    config: &BlazeFaceConfig,
) -> Result<Vec<Tensor>> // Vector of weighted detections by non-maximum suppression
{
    if detections.dims()[0] == 0 {
        return Ok(Vec::new());
    }
    if detections.dims()[1] != 17 {
        return Err(Error::ShapeMismatchBinaryOp {
            lhs: detections.shape().clone(),
            rhs: Shape::from_dims(&[
                detections.dims()[0],
                17,
            ]),
            op: "weighted_non_max_suppression",
        });
    }

    let mut output = Vec::new();

    let mut remaining = argsort_by_score(detections)?; // (num_detections) of Dtype::U32

    while remaining.dims()[0] > 0 {
        let detection = detections.i((
            remaining.to_vec1::<u32>()?[0] as usize,
            ..,
        ))?; // (17)

        let first_box = detection.i(0..4)?; // (4)
        let other_box = detections.i((&remaining, ..4))?; // (remainings, 4) containing first_box

        let ious = overlap_similarity(&first_box, &other_box)?; // (remainings)
        let mask = ious.gt(config.min_suppression_threshold)?; // (remainings) of Dtype::U8
        let (overlapping, others) = mask_indices(&mask)?; // (unmasked_indices), (masked_indices)

        remaining = others; // (unmasked_indices)

        let mut weighted_detection = detection.clone(); // (17)

        if overlapping.dims()[0] > 1 {
            let overlapped = detections.i((&overlapping, ..))?; // (overlapped, 17)
            let coordinates = overlapped.i((.., 0..16))?; // (overlapped, 16)
            let scores = overlapped
                .i((.., 16))? // (overlapped, 1)
                .squeeze(1)?; // (overlapped)
            let total_score = scores.sum(0)?; // (1)
            let overlapped_count = Tensor::from_slice(
                &[overlapped.dims()[0] as f32],
                1,
                detections.device(),
            )?; // (1)

            let weighted_coordinates = coordinates
                .broadcast_mul(&scores)? // (overlapped, 16)
                .sum(0)? // (16)
                .div(&total_score)?; // (16)

            let weighted_score = total_score.div(&overlapped_count)?; // (1)

            weighted_detection = Tensor::cat(
                &[
                    weighted_coordinates,
                    weighted_score,
                ],
                0,
            )?; // (17)
        }

        output.push(weighted_detection);
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn test_forward_back() {
        // Set up the device and dtype
        let device = Device::Cpu;
        let dtype = DType::F32;
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
            &variables,
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

        assert_eq!(output.0.dims(), &[batch_size, 896, 16]);
        assert_eq!(output.1.dims(), &[batch_size, 896, 1]);
    }

    #[test]
    fn test_forward_front() {
        // Set up the device and dtype
        let device = Device::Cpu;
        let dtype = DType::F32;
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
            &variables,
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

        assert_eq!(output.0.dims(), &[batch_size, 896, 16]);
        assert_eq!(output.1.dims(), &[batch_size, 896, 1]);
    }

    #[test]
    fn test_decode_boxes() {
        // Set up the device and dtype
        let device = Device::Cpu;
        let dtype = DType::F32;
        let batch_size = 1;

        // Set up the anchors and configuration
        let anchors = Tensor::read_npy("src/blaze_face/data/anchorsback.npy")
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let config = BlazeFaceConfig::back(100., 0.65, 0.3, &device).unwrap();

        // Set up the input Tensor
        let input = Tensor::rand(-1., 1., (batch_size, 896, 16), &device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();

        // Decode boxes
        let boxes = decode_boxes(&input, &anchors, &config).unwrap();

        assert_eq!(boxes.dims(), &[batch_size, 896, 16]);
    }

    #[test]
    fn test_unmasked_indices() {
        // Set up the device and dtype
        let device = Device::Cpu;
        let dtype = DType::F32;
        let batch_size = 1;

        // Set up the ones Tensor
        let ones = Tensor::ones((batch_size, 896, 1), dtype, &device).unwrap();

        // Unmasked indices
        let indices = unmasked_indices(&ones, 0.5).unwrap();

        assert_eq!(indices.dims(), &[batch_size, 896]);

        // Set up the zeros Tensor
        let zeros =
            Tensor::zeros((batch_size, 896, 1), dtype, &device).unwrap();

        // Unmasked indices
        let indices = unmasked_indices(&zeros, 0.5).unwrap();

        assert_eq!(indices.dims(), &[batch_size, 0]);

        // Set up the test tensor
        let input = Tensor::from_slice(
            &[
                0.8, 0., 0., 0., 0., 0., 0., 0., 0., 0.4, //
                0., 0., 1., 0., 0., 0., 0., 0.7, 0., 0., //
                0., 0., 0., 0., 0., 0.8, 0., 0.1, 0.6, 0., //
            ],
            (batch_size, 30, 1),
            &device,
        )
        .unwrap()
        .to_dtype(dtype)
        .unwrap();

        // Unmasked indices
        let indices = unmasked_indices(&input, 0.5).unwrap();

        assert_eq!(indices.dims(), &[batch_size, 5]);

        assert_eq!(
            indices
                .squeeze(0)
                .unwrap()
                .to_vec1::<u32>()
                .unwrap(),
            &[0, 12, 17, 25, 28]
        );
    }

    #[test]
    fn test_tensors_to_detections() {
        // Set up the device and dtype
        let device = Device::Cpu;
        let dtype = DType::F32;
        let batch_size = 1;

        // Load the variables
        let variables = candle_nn::VarBuilder::from_pth(
            "src/blaze_face/data/blazefaceback.pth",
            dtype,
            &device,
        )
        .unwrap();

        // Load the anchors
        let anchors = Tensor::read_npy("src/blaze_face/data/anchorsback.npy")
            .unwrap()
            .to_dtype(dtype)
            .unwrap(); // (896, 4)
        assert_eq!(anchors.dims(), &[896, 4]);

        // Load the model
        let model = BlazeFace::load(
            ModelType::Back,
            &variables,
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
        .unwrap(); // (batch_size, 3, 256, 256)
        assert_eq!(input.dims(), &[batch_size, 3, 256, 256]);

        // Call forward method and get the output
        let (raw_boxes, raw_scores) = model.forward(&input).unwrap();
        // raw_scores: (batch_size, 896, 1), raw_boxes: (batch_size, 896, 16)
        assert_eq!(raw_boxes.dims(), &[batch_size, 896, 16]);
        assert_eq!(raw_scores.dims(), &[batch_size, 896, 1]);

        // Tensors to detections
        let detections = tensors_to_detections(
            &raw_boxes,
            &raw_scores,
            &model.anchors,
            &model.config,
        )
        .unwrap(); // Vec<(num_detections, 17)> with length:batch_size

        assert_eq!(detections.len(), batch_size);
        assert_eq!(detections[0].dims(), &[0, 17]);
    }

    #[test]
    fn test_argsort() {
        // Set up the device and dtype
        let device = Device::Cpu;
        let dtype = DType::F32;

        // Set up the input Tensor
        let right_eye = Tensor::from_slice(
            &[
                0.8, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0.4,
            ],
            17,
            &device,
        )
        .unwrap(); // (17)
        let left_eye = Tensor::from_slice(
            &[
                0., 0.7, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0.8,
            ],
            17,
            &device,
        )
        .unwrap(); // (17)
        let input = Tensor::stack(&[right_eye, left_eye], 0)
            .unwrap()
            .to_dtype(dtype)
            .unwrap(); // (2, 17)
        assert_eq!(input.dims(), &[2, 17]);
        assert_eq!(
            input
                .i((0, 16))
                .unwrap()
                .to_vec0::<f32>()
                .unwrap(),
            0.4,
        );
        assert_eq!(
            input
                .i((1, 16))
                .unwrap()
                .to_vec0::<f32>()
                .unwrap(),
            0.8,
        );

        // Sort
        let sorted = argsort_by_score(&input).unwrap();
        assert_eq!(sorted.dims()[0], 2);
        assert_eq!(
            sorted
                .to_vec1::<u32>()
                .unwrap()[0],
            1
        );
        assert_eq!(
            sorted
                .to_vec1::<u32>()
                .unwrap()[1],
            0
        );
    }

    #[test]
    fn test_intersect() {
        // Set up the device and dtype
        let device = Device::Cpu;
        let dtype = DType::F32;

        // Set up the boxes Tensors
        let box_a = Tensor::from_slice(
            &[
                0., 0., 10., 10., //
                10., 10., 20., 20., //
            ],
            (2, 4),
            &device,
        )
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
        assert_eq!(box_a.dims(), &[2, 4]);
        assert_eq!(
            box_a
                .to_vec2::<f32>()
                .unwrap(),
            vec![
                [0., 0., 10., 10.,], //
                [10., 10., 20., 20., //
            ],
            ],
        );

        let box_b = Tensor::from_slice(
            &[
                5., 5., 15., 15., //
                15., 15., 25., 25., //
            ],
            (2, 4),
            &device,
        )
        .unwrap()
        .to_dtype(dtype)
        .unwrap();

        // Intersect
        let intersect = intersect(&box_a, &box_b).unwrap(); // (2, 2)

        assert_eq!(intersect.dims(), &[2, 2]);
        assert_eq!(
            intersect
                .to_vec2::<f32>()
                .unwrap(),
            vec![
                [
                    25., // (0, 0, 10, 10) intersects (5, 5, 15, 15) with area 25
                    0.,  // (0, 0, 10, 10) does not intersect (15, 15, 25, 25)
                ],
                [
                    25., // (10, 10, 20, 20) intersects (5, 5, 15, 15) with area 25
                    25., // (10, 10, 20, 20) intersects (15, 15, 25, 25) with area 25
                ],
            ]
        );
    }

    #[test]
    fn test_jaccard() {
        // Set up the device and dtype
        let device = Device::Cpu;
        let dtype = DType::F32;

        // Set up the boxes Tensors
        let box_a = Tensor::from_slice(
            &[
                0., 0., 10., 10., //
                10., 10., 20., 20., //
            ],
            (2, 4),
            &device,
        )
        .unwrap()
        .to_dtype(dtype)
        .unwrap();

        let box_b = Tensor::from_slice(
            &[
                5., 5., 15., 15., //
                15., 15., 25., 25., //
            ],
            (2, 4),
            &device,
        )
        .unwrap()
        .to_dtype(dtype)
        .unwrap();

        // Jaccard
        let jaccard = jaccard(&box_a, &box_b).unwrap(); // (2, 2)

        assert_eq!(jaccard.dims(), &[2, 2]);
        assert_eq!(
            jaccard
                .to_vec2::<f32>()
                .unwrap(),
            vec![
                [
                    1. / 7., // = 25 / (100 + 100 - 25)
                    0.,      // = 0 / (100 + 100 - 0)
                ],
                [
                    1. / 7., // = 25 / (100 + 100 - 25)
                    1. / 7., // = 25 / (100 + 100 - 25)
                ],
            ]
        );
    }

    #[test]
    fn test_overlap_similarity() {
        // Set up the device and dtype
        let device = Device::Cpu;
        let dtype = DType::F32;

        // Set up the boxes Tensors
        let box_a = Tensor::from_slice(
            &[
                0., 0., 10., 10., //
            ],
            4,
            &device,
        )
        .unwrap()
        .to_dtype(dtype)
        .unwrap();

        let box_b = Tensor::from_slice(
            &[
                5., 5., 15., 15., //
                15., 15., 25., 25., //
            ],
            (2, 4),
            &device,
        )
        .unwrap()
        .to_dtype(dtype)
        .unwrap();

        // Overlap similarity
        let similarity = overlap_similarity(&box_a, &box_b).unwrap(); // (2)

        assert_eq!(
            similarity
                .to_vec1::<f32>()
                .unwrap(),
            vec![
                1. / 7., // = 25 / (100 + 100 - 25)
                0.,      // = 0  / (100 + 100 - 25)
            ]
        );

        let box_c = Tensor::from_slice(
            &[
                0., 0., 10., 10., //
            ],
            (1, 4),
            &device,
        )
        .unwrap()
        .to_dtype(dtype)
        .unwrap();

        let same_similarity = overlap_similarity(&box_a, &box_c).unwrap(); // (1)
        assert_eq!(
            same_similarity
                .to_vec1::<f32>()
                .unwrap(),
            vec![
                1., // = 100 / (100 + 100 - 100)
            ]
        );
    }

    #[test]
    fn test_tensor_mask() {
        let device = Device::Cpu;

        let tensor = Tensor::from_slice(
            &[
                0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.,
            ],
            11,
            &device,
        )
        .unwrap();

        let threashold = 0.4;

        let mask = tensor.gt(threashold).unwrap();
        assert_eq!(
            mask.to_vec1::<u8>().unwrap(),
            vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        );

        let unmasked = tensor.i(&mask).unwrap();
        assert_eq!(
            unmasked
                .to_vec1::<f64>()
                .unwrap(),
            vec![0., 0., 0., 0., 0., 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        );

        let unmask = tensor.le(threashold).unwrap();
        assert_eq!(
            unmask
                .to_vec1::<u8>()
                .unwrap(),
            vec![1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        );

        let masked = tensor.i(&unmask).unwrap();
        assert_eq!(
            masked
                .to_vec1::<f64>()
                .unwrap(),
            vec![0.1, 0.1, 0.1, 0.1, 0.1, 0., 0., 0., 0., 0., 0.]
        );

        let selected = tensor
            .index_select(&mask, 0)
            .unwrap();
        assert_eq!(
            selected
                .to_vec1::<f64>()
                .unwrap(),
            vec![0., 0., 0., 0., 0., 0.1, 0.1, 0.1, 0.1, 0.1, 0.1] // equals to unmasked
        );
    }

    #[test]
    fn test_mask_indices() {
        let device = Device::Cpu;
        let dtype = DType::F32;

        let similarities = Tensor::from_slice(
            &[
                0., 0.1, 0.2, 0.3, 0.4, 0.5,
            ],
            6,
            &device,
        )
        .unwrap()
        .to_dtype(dtype)
        .unwrap(); // (6)

        let threashold = 0.3_f32;

        let (unmasked, masked) = mask_indices(
            &similarities
                .gt(threashold)
                .unwrap(),
        )
        .unwrap(); // (2), (4)

        assert_eq!(
            unmasked
                .to_vec1::<u32>()
                .unwrap(),
            vec![4, 5]
        );
        assert_eq!(
            masked
                .to_vec1::<u32>()
                .unwrap(),
            vec![0, 1, 2, 3]
        );
    }

    #[test]
    fn test_weighted_non_max_suppression() {
        // Set up the device and dtype
        let device = Device::Cpu;
        let dtype = DType::F32;

        // Setup the config
        let config = BlazeFaceConfig::back(100., 0.65, 0.3, &device).unwrap();

        // Setup the detections
        let detections = Tensor::from_slice(
            &[
                0., 0., 0.8, 0.8, // Bounding box
                0.1, 0.1, // Right eye
                0.2, 0.2, // Left eye
                0.3, 0.3, // Nose
                0.5, 0.5, // Mouth
                0.6, 0.6, // Right ear
                0.7, 0.7, // Left ear
                0.8, // Score
            ],
            (1, 17),
            &device,
        )
        .unwrap()
        .to_dtype(dtype)
        .unwrap();

        // Calculate weighted non-maximum suppression
        let weighted_detections =
            weighted_non_max_suppression(&detections, &config).unwrap(); // Vec<(num_detections, 17)> with length:batch_size

        assert_eq!(weighted_detections.len(), 1);
    }

    #[test]
    fn test_tensors_to_detections_by_1face_front() {
        // Set up the device and dtype
        let device = Device::Cpu;
        let dtype = DType::F32;

        // Load the model
        let model = load_model(ModelType::Front, &device, dtype).unwrap();

        // Load the test image
        let image = image::open("test_data/1face.png").unwrap();
        let input = convert_image_to_tensor(&image, &device) // (3, 128, 128)
            .unwrap()
            .unsqueeze(0) // (1, 3, 128, 128)
            .unwrap();

        // Call forward method and get the output
        let (raw_boxes, raw_scores) = model.forward(&input).unwrap();
        // raw_boxes: (batch_size, 896, 16), raw_scores: (batch_size, 896, 1)

        // Tensors to detections
        let detections = tensors_to_detections(
            &raw_boxes,
            &raw_scores,
            &model.anchors,
            &model.config,
        )
        .unwrap(); // Vec<(num_detections, 17)> with length:batch_size

        assert_eq!(detections.len(), 1);
        assert_eq!(
            detections[0]
                .i((.., 16))
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![0.76187944]
        );
    }

    #[test]
    fn test_tensors_to_detections_by_4faces_back() {
        // Set up the device and dtype
        let device = Device::Cpu;
        let dtype = DType::F32;

        // Load the model
        let model = load_model(ModelType::Back, &device, dtype).unwrap();

        // Load the test image
        let image = image::open("test_data/4faces.png")
            .unwrap()
            .resize_exact(
                256,
                256,
                image::imageops::FilterType::Nearest,
            );
        let input = convert_image_to_tensor(&image, &device) // (3, 256, 256)
            .unwrap()
            .unsqueeze(0) // (1, 3, 256, 256)
            .unwrap();

        // Call forward method and get the output
        let (raw_boxes, raw_scores) = model.forward(&input).unwrap();
        // raw_boxes: (batch_size, 896, 16), raw_scores: (batch_size, 896, 1)

        // Tensors to detections
        let detections = tensors_to_detections(
            &raw_boxes,
            &raw_scores,
            &model.anchors,
            &model.config,
        )
        .unwrap(); // Vec<(num_detections, 17)> with length:batch_size

        assert_eq!(detections.len(), 1);
        assert_eq!(
            detections[0]
                .i((.., 16))
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![0.8581102, 0.8371221, 0.6723021]
        );
    }

    #[test]
    fn test_predict_on_batch_by_1face() {
        // Set up the device and dtype
        let device = Device::Cpu;
        let dtype = DType::F32;

        // Load the model
        let model = load_model(ModelType::Front, &device, dtype).unwrap();

        // Load the test image
        let image = image::open("test_data/1face.png").unwrap();
        let input = convert_image_to_tensor(&image, &device) // (3, 128, 128)
            .unwrap()
            .unsqueeze(0) // (1, 3, 128, 128)
            .unwrap();

        // Predict on batch
        let detections = model
            .predict_on_batch(&input)
            .unwrap(); // Vec<Vec<(17)>> with length:batch_size of length:num_detections

        assert_eq!(detections.len(), 1); // batch_size
        assert_eq!(detections[0].len(), 1); // detectec faces
        assert_eq!(detections[0][0].dims(), &[17]); // bounding box, keypoints and score
        assert_eq!(
            detections[0][0]
                .i(16)
                .unwrap()
                .to_vec0::<f32>()
                .unwrap(),
            0.76187944
        );
    }

    #[test]
    fn test_color_order() {
        let device = Device::Cpu;

        let colors = vec![
            0.1, 0.2, 0.3, // (0, 0)
            0.4, 0.5, 0.6, // (1, 0)
            0.7, 0.8, 0.9, // (0, 1)
            0.11, 0.12, 0.13, // (1, 1)
        ];

        let tensor = Tensor::from_vec(colors, (2, 2, 3), &device).unwrap();

        assert_eq!(
            tensor
                .to_vec3::<f64>()
                .unwrap(),
            vec![
                vec![
                    vec![0.1, 0.2, 0.3],
                    vec![0.4, 0.5, 0.6],
                ],
                vec![
                    vec![0.7, 0.8, 0.9],
                    vec![0.11, 0.12, 0.13],
                ],
            ]
        );

        let tensor = tensor
            .permute((2, 0, 1))
            .unwrap();

        assert_eq!(
            tensor
                .to_vec3::<f64>()
                .unwrap(),
            vec![
                vec![
                    // R
                    vec![
                        // W = 0
                        0.1, // H = 0
                        0.4  // H = 1
                    ],
                    vec![
                        // W = 1
                        0.7,  // H = 0
                        0.11  // H = 1
                    ],
                ],
                vec![
                    // G
                    vec![0.2, 0.5],
                    vec![0.8, 0.12],
                ],
                vec![
                    // G
                    vec![0.3, 0.6],
                    vec![0.9, 0.13],
                ],
            ]
        );

        let tensor = tensor
            .permute((0, 2, 1))
            .unwrap();

        assert_eq!(
            tensor
                .to_vec3::<f64>()
                .unwrap(),
            vec![
                vec![
                    // R
                    vec![
                        // H = 0
                        0.1, // W = 0
                        0.7  // W = 1
                    ],
                    vec![
                        // H = 1
                        0.4,  // W = 0
                        0.11  // W = 1
                    ],
                ],
                vec![
                    // G
                    vec![
                        // H = 0
                        0.2, // W = 0
                        0.8  // W = 1
                    ],
                    vec![
                        // H = 1
                        0.5,  // W = 0
                        0.12  // W = 1
                    ],
                ],
                vec![
                    // B
                    vec![
                        // H = 0
                        0.3, // W = 0
                        0.9  // W = 1
                    ],
                    vec![
                        // H = 1
                        0.6,  // W = 0
                        0.13  // W = 1
                    ],
                ],
            ]
        );
    }

    fn load_model(
        model_type: ModelType,
        device: &Device,
        dtype: DType,
    ) -> Result<BlazeFace> {
        let pth_path = match model_type {
            | ModelType::Back => "src/blaze_face/data/blazefaceback.pth",
            | ModelType::Front => "src/blaze_face/data/blazeface.pth",
        };

        // Load the variables
        let variables =
            candle_nn::VarBuilder::from_pth(pth_path, dtype, device)?;

        let anchor_path = match model_type {
            | ModelType::Back => "src/blaze_face/data/anchorsback.npy",
            | ModelType::Front => "src/blaze_face/data/anchors.npy",
        };

        // Load the anchors
        let anchors = Tensor::read_npy(anchor_path)? // (896, 4)
            .to_dtype(dtype)?
            .to_device(device)?;

        let min_score_thresh = match model_type {
            | ModelType::Back => 0.65,
            | ModelType::Front => 0.75,
        };

        // Load the model
        BlazeFace::load(
            model_type,
            &variables,
            anchors,
            100.,
            min_score_thresh,
            0.3,
        )
    }

    fn convert_image_to_tensor(
        image: &image::DynamicImage,
        device: &Device,
    ) -> Result<Tensor> {
        let pixels = image.to_rgb32f().to_vec();

        Tensor::from_vec(
            pixels,
            (
                image.width() as usize,
                image.height() as usize,
                3,
            ),
            device,
        )? // (width, height, channel = 3) in range [0., 1.]
        .permute((2, 1, 0))? // (3, height, width) in range [0., 1.]
        .contiguous()?
        .broadcast_mul(&Tensor::from_slice(
            &[2_f32],
            1,
            device,
        )?)? // (3, height, width) in range [0., 2.]
        .broadcast_sub(&Tensor::from_slice(
            &[1_f32],
            1,
            device,
        )?) // (3, height, width) in range [-1., 1.]
    }
}
