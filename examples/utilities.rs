use candle_core::{DType, Device, Result, Tensor};
use face_tracking_rs::blaze_face::{
    blaze_face::{BlazeFace, ModelType},
    face_detection::FaceDetection,
};
use image::{DynamicImage, Rgba, RgbaImage};
use imageproc::drawing::{draw_cross, draw_hollow_rect};
use imageproc::rect::Rect;

pub(crate) fn load_model(
    model_type: ModelType,
    min_score_threshold: f32,
    device: &Device,
    dtype: DType,
) -> Result<BlazeFace> {
    let pth_path = match model_type {
        | ModelType::Back => "src/blaze_face/data/blazefaceback.pth",
        | ModelType::Front => "src/blaze_face/data/blazeface.pth",
    };

    // Load the variables
    let variables = candle_nn::VarBuilder::from_pth(pth_path, dtype, device)?;

    let anchor_path = match model_type {
        | ModelType::Back => "src/blaze_face/data/anchorsback.npy",
        | ModelType::Front => "src/blaze_face/data/anchors.npy",
    };

    // Load the anchors
    let anchors = Tensor::read_npy(anchor_path)? // (896, 4)
        .to_dtype(dtype)?
        .to_device(device)?;

    // Load the model
    BlazeFace::load(
        model_type,
        &variables,
        anchors,
        100.,
        min_score_threshold,
        0.3,
    )
}

pub(crate) fn convert_image_to_tensor(
    image: &DynamicImage,
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

pub(crate) fn draw_face_detections(
    image: &RgbaImage,
    detections: Vec<FaceDetection>,
    x_scale: f32,
    y_scale: f32,
) -> Result<RgbaImage> {
    let mut image = image.clone();

    for detection in detections {
        // Draw the red bounding box
        let bounding_box = detection.bounding_box;
        let rect = Rect::at(
            (bounding_box.x_min * x_scale) as i32,
            (bounding_box.y_min * y_scale) as i32,
        )
        .of_size(
            ((bounding_box.x_max - bounding_box.x_min) * x_scale) as u32,
            ((bounding_box.y_max - bounding_box.y_min) * y_scale) as u32,
        );
        image = draw_hollow_rect(&image, rect, Rgba([255, 0, 0, 255]));

        // Draw the green key points
        let keypoints = detection.key_points;
        image = draw_cross(
            &image,
            Rgba([0, 255, 0, 255]),
            (keypoints.right_eye.x * x_scale) as i32,
            (keypoints.right_eye.y * y_scale) as i32,
        );
        image = draw_cross(
            &image,
            Rgba([0, 255, 0, 255]),
            (keypoints.left_eye.x * x_scale) as i32,
            (keypoints.left_eye.y * y_scale) as i32,
        );
        image = draw_cross(
            &image,
            Rgba([0, 255, 0, 255]),
            (keypoints.nose.x * x_scale) as i32,
            (keypoints.nose.y * y_scale) as i32,
        );
        image = draw_cross(
            &image,
            Rgba([0, 255, 0, 255]),
            (keypoints.mouth.x * x_scale) as i32,
            (keypoints.mouth.y * y_scale) as i32,
        );
        image = draw_cross(
            &image,
            Rgba([0, 255, 0, 255]),
            (keypoints.right_ear.x * x_scale) as i32,
            (keypoints.right_ear.y * y_scale) as i32,
        );
        image = draw_cross(
            &image,
            Rgba([0, 255, 0, 255]),
            (keypoints.left_ear.x * x_scale) as i32,
            (keypoints.left_ear.y * y_scale) as i32,
        );
    }

    Ok(image)
}

fn main() {}
