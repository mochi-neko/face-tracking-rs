mod utilities;

use candle_core::{DType, Device};
use face_tracking_rs::blaze_face::{
    blaze_face::ModelType, face_detection::FaceDetection,
};

use crate::utilities::draw_face_detections;

fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let dtype = DType::F32;

    let image = image::open("test_data/1face.png")?;
    let image_tensor = utilities::convert_image_to_tensor(&image, &device)?;

    let model = utilities::load_model(ModelType::Front, 0.75, &device, dtype)?;

    let detections = model.predict_on_image(&image_tensor)?;
    let detections = FaceDetection::from_tensors(
        detections
            .get(0)
            .unwrap()
            .clone(),
    )?;

    println!("{:?}", detections);

    let detected_image = draw_face_detections(
        &image.to_rgba8(),
        detections,
        128.,
        128.,
    )?;

    detected_image.save("output/1face_detected.png")?;

    Ok(())
}
