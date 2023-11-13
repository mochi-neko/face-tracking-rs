use candle_core::{DType, Device, Result, Tensor};
use face_tracking_rs::blaze_face::{
    blaze_face::{BlazeFace, ModelType},
    face_detection::FaceDetection,
};
use image::{DynamicImage, Rgba, RgbaImage};
use imageproc::drawing::{draw_cross, draw_hollow_rect};
use imageproc::rect::Rect;

pub fn load_model(
    model_type: ModelType,
    min_score_threshold: f32,
    min_suppression_threshold: f32,
    device: &Device,
) -> Result<BlazeFace> {
    let dtype = DType::F16;
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
        min_suppression_threshold,
    )
}

pub(crate) fn load_image(
    image_path: &str,
    model_type: ModelType,
) -> anyhow::Result<DynamicImage> {
    let image = image::open(image_path)?;
    let image = match model_type {
        | ModelType::Back => image.resize_to_fill(
            256,
            256,
            image::imageops::FilterType::Nearest,
        ),
        | ModelType::Front => image.resize_to_fill(
            128,
            128,
            image::imageops::FilterType::Nearest,
        ),
    };

    Ok(image)
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
    model_type: ModelType,
) -> Result<RgbaImage> {
    let scale = match model_type {
        | ModelType::Back => 256.,
        | ModelType::Front => 128.,
    };

    let mut image = image.clone();

    for detection in detections {
        // Draw the red bounding box
        let bounding_box = detection.bounding_box;
        let rect = Rect::at(
            (bounding_box.x_min * scale) as i32,
            (bounding_box.y_min * scale) as i32,
        )
        .of_size(
            ((bounding_box.x_max - bounding_box.x_min) * scale) as u32,
            ((bounding_box.y_max - bounding_box.y_min) * scale) as u32,
        );
        image = draw_hollow_rect(&image, rect, Rgba([255, 0, 0, 255]));

        // Draw the green key points
        let keypoints = detection.key_points;
        image = draw_cross(
            &image,
            Rgba([0, 255, 0, 255]),
            (keypoints.right_eye.x * scale) as i32,
            (keypoints.right_eye.y * scale) as i32,
        );
        image = draw_cross(
            &image,
            Rgba([0, 255, 0, 255]),
            (keypoints.left_eye.x * scale) as i32,
            (keypoints.left_eye.y * scale) as i32,
        );
        image = draw_cross(
            &image,
            Rgba([0, 255, 0, 255]),
            (keypoints.nose.x * scale) as i32,
            (keypoints.nose.y * scale) as i32,
        );
        image = draw_cross(
            &image,
            Rgba([0, 255, 0, 255]),
            (keypoints.mouth.x * scale) as i32,
            (keypoints.mouth.y * scale) as i32,
        );
        image = draw_cross(
            &image,
            Rgba([0, 255, 0, 255]),
            (keypoints.right_ear.x * scale) as i32,
            (keypoints.right_ear.y * scale) as i32,
        );
        image = draw_cross(
            &image,
            Rgba([0, 255, 0, 255]),
            (keypoints.left_ear.x * scale) as i32,
            (keypoints.left_ear.y * scale) as i32,
        );
    }

    Ok(image)
}

fn main() {}
