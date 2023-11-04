mod utilities;

use candle_core::Device;
use face_tracking_rs::blaze_face::{
    blaze_face::ModelType, face_detection::FaceDetection,
};

fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    println!("device: {:?}", device);

    let model_type = ModelType::Front;
    let image = utilities::load_image("test_data/3faces.png", model_type)?;
    let image_tensor = utilities::convert_image_to_tensor(&image, &device)?;

    let model = utilities::load_model(model_type, 0.6, &device)?;

    let detections = model.predict_on_image(&image_tensor)?;
    let detections = FaceDetection::from_tensors(
        detections.first()
            .unwrap()
            .clone(),
    )?;

    println!("{:?} faces detections: {:?}", detections.len(), detections);

    let detected_image = utilities::draw_face_detections(
        &image.to_rgba8(),
        detections,
        model_type,
    )?;

    detected_image.save("output/3faces_detected.png")?;

    Ok(())
}