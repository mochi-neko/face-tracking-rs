use candle_core::{Error, Result, Shape, Tensor};

#[derive(Debug)]
pub struct BoundingBox {
    pub x_min: f32,
    pub y_min: f32,
    pub x_max: f32,
    pub y_max: f32,
}

#[derive(Debug)]
pub struct Keypoints {
    pub right_eye: KeyPoint,
    pub left_eye: KeyPoint,
    pub nose: KeyPoint,
    pub mouth: KeyPoint,
    pub right_ear: KeyPoint,
    pub left_ear: KeyPoint,
}

#[derive(Debug)]
pub struct KeyPoint {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug)]
pub struct FaceDetection {
    pub bounding_box: BoundingBox,
    pub key_points: Keypoints,
    pub score: f32,
}

impl FaceDetection {
    pub fn from_tensor(tensor: &Tensor, // (17)
    ) -> Result<Self> {
        if tensor.dims() != [17] {
            return Result::Err(Error::ShapeMismatchBinaryOp {
                lhs: tensor.shape().clone(),
                rhs: Shape::from(&[17]),
                op: "from_tensor",
            });
        }

        let vector = tensor.to_vec1::<f32>()?;

        let bounding_box = BoundingBox {
            x_min: vector[1],
            y_min: vector[0],
            x_max: vector[3],
            y_max: vector[2],
        };

        let key_points = Keypoints {
            right_eye: KeyPoint {
                x: vector[5],
                y: vector[4],
            },
            left_eye: KeyPoint {
                x: vector[7],
                y: vector[6],
            },
            nose: KeyPoint {
                x: vector[9],
                y: vector[8],
            },
            mouth: KeyPoint {
                x: vector[11],
                y: vector[10],
            },
            right_ear: KeyPoint {
                x: vector[13],
                y: vector[12],
            },
            left_ear: KeyPoint {
                x: vector[15],
                y: vector[14],
            },
        };

        let score = vector[16];

        Ok(Self {
            bounding_box,
            key_points,
            score,
        })
    }

    pub fn from_tensors(
        tensors: Vec<Tensor>, // Vec<(17)>
    ) -> Result<Vec<Self>> {
        let mut face_detections = Vec::with_capacity(tensors.len());

        for tensor in tensors {
            face_detections.push(Self::from_tensor(&tensor)?);
        }

        Ok(face_detections)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn test_from_tensor() {
        let device = Device::Cpu;
        let dtype = DType::F32;

        let tensor = Tensor::from_slice(
            &[
                0.1, 0.11, 0.9, 0.91, // bounding box
                0.7, 0.6, // right eye
                0.3, 0.6, // left eye
                0.5, 0.5, // nose
                0.5, 0.3, // mouth
                0.8, 0.55, // right ear
                0.2, 0.55, // left ear
                0.9,  // score
            ],
            17,
            &device,
        )
        .unwrap()
        .to_dtype(dtype)
        .unwrap();

        let face_detection = FaceDetection::from_tensor(&tensor).unwrap();

        assert_eq!(
            face_detection
                .bounding_box
                .x_min,
            0.11
        );
        assert_eq!(
            face_detection
                .bounding_box
                .y_min,
            0.1
        );
        assert_eq!(
            face_detection
                .bounding_box
                .x_max,
            0.91
        );
        assert_eq!(
            face_detection
                .bounding_box
                .y_max,
            0.9
        );

        assert_eq!(
            face_detection
                .key_points
                .right_eye
                .x,
            0.7
        );
        assert_eq!(
            face_detection
                .key_points
                .right_eye
                .y,
            0.6
        );
        assert_eq!(
            face_detection
                .key_points
                .left_eye
                .x,
            0.3
        );
        assert_eq!(
            face_detection
                .key_points
                .left_eye
                .y,
            0.6
        );
        assert_eq!(
            face_detection
                .key_points
                .nose
                .x,
            0.5
        );
        assert_eq!(
            face_detection
                .key_points
                .nose
                .y,
            0.5
        );
        assert_eq!(
            face_detection
                .key_points
                .mouth
                .x,
            0.5
        );
        assert_eq!(
            face_detection
                .key_points
                .mouth
                .y,
            0.3
        );
        assert_eq!(
            face_detection
                .key_points
                .right_ear
                .x,
            0.8
        );
        assert_eq!(
            face_detection
                .key_points
                .right_ear
                .y,
            0.55
        );
        assert_eq!(
            face_detection
                .key_points
                .left_ear
                .x,
            0.2
        );
        assert_eq!(
            face_detection
                .key_points
                .left_ear
                .y,
            0.55
        );

        assert_eq!(face_detection.score, 0.9);
    }

    #[test]
    fn test_from_tensors() {
        let device = Device::Cpu;
        let dtype = DType::F32;

        let tensor_1 = Tensor::from_slice(
            &[
                0.1, 0.11, 0.9, 0.91, // bounding box
                0.7, 0.6, // right eye
                0.3, 0.6, // left eye
                0.5, 0.5, // nose
                0.5, 0.3, // mouth
                0.8, 0.55, // right ear
                0.2, 0.55, // left ear
                0.9,  // score
            ],
            17,
            &device,
        )
        .unwrap()
        .to_dtype(dtype)
        .unwrap();

        let tensor_2 = Tensor::from_slice(
            &[
                0.2, 0.21, 0.8, 0.81, // bounding box
                0.6, 0.5, // right eye
                0.4, 0.5, // left eye
                0.6, 0.4, // nose
                0.6, 0.2, // mouth
                0.7, 0.45, // right ear
                0.3, 0.45, // left ear
                0.8,  // score
            ],
            17,
            &device,
        )
        .unwrap()
        .to_dtype(dtype)
        .unwrap();

        let face_detections =
            FaceDetection::from_tensors(vec![tensor_1, tensor_2]).unwrap();

        assert_eq!(face_detections.len(), 2);
        assert_eq!(
            face_detections[0]
                .bounding_box
                .x_min,
            0.11
        );
        assert_eq!(
            face_detections[1]
                .bounding_box
                .x_min,
            0.21
        );
    }
}
