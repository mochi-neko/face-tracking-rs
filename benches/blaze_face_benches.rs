use candle_core::{safetensors, DType, Device, Module, Result, Tensor};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use face_tracking_rs::blaze_face::{
    blaze_block::BlazeBlock,
    blaze_face::{BlazeFace, ModelType},
};

fn load_model(
    model_type: ModelType,
    device: &Device,
) -> Result<BlazeFace> {
    let dtype = DType::F16;

    // Load the variables
    let safetensors_path = match model_type {
        | ModelType::Back => "src/blaze_face/data/blazefaceback.safetensors",
        | ModelType::Front => "src/blaze_face/data/blazeface.safetensors",
    };
    let safetensors = safetensors::load(safetensors_path, device)?;
    let variables =
        candle_nn::VarBuilder::from_tensors(safetensors, dtype, device);

    // Load the anchors
    let anchor_path = match model_type {
        | ModelType::Back => "src/blaze_face/data/anchorsback.npy",
        | ModelType::Front => "src/blaze_face/data/anchors.npy",
    };
    let anchors = Tensor::read_npy(anchor_path)?
        .to_dtype(dtype)?
        .to_device(device)?;

    // Load the model
    BlazeFace::load(
        model_type, &variables, anchors, 100., 0.6, 0.3,
    )
}

fn load_image(
    path: &str,
    model_type: ModelType,
    device: &Device,
) -> anyhow::Result<Tensor> {
    let dtype = DType::F16;

    // Open the image
    let image = image::open(path)?;

    // Resize the image
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

    // Convert image pixels to tensor (3, 128, 128) or (3, 256, 256)
    let pixels = image.to_rgb32f().to_vec();
    let tensor = Tensor::from_vec(
        pixels,
        (
            image.width() as usize,
            image.height() as usize,
            3,
        ),
        device,
    )?
    .permute((2, 1, 0))?
    .contiguous()?
    .broadcast_mul(&Tensor::from_slice(
        &[2_f32],
        1,
        device,
    )?)?
    .broadcast_sub(&Tensor::from_slice(
        &[1_f32],
        1,
        device,
    )?)?
    .to_dtype(dtype)?;

    anyhow::Ok(tensor)
}

fn blaze_face_benchmark(c: &mut Criterion) {
    let device = Device::Cpu;

    // Load the models
    let front_model = load_model(ModelType::Front, &device).unwrap();
    let back_model = load_model(ModelType::Back, &device).unwrap();

    // Load the image
    let image_path = "test_data/1face.png";
    let image_front =
        load_image(image_path, ModelType::Front, &device).unwrap();
    let image_back = load_image(image_path, ModelType::Back, &device).unwrap();

    c.bench_function("1face_front", |b| {
        b.iter(|| {
            //black_box(|| {
            // Run the model
            let _detections = front_model
                .predict_on_image(&image_front)
                .unwrap();
            //})
        });
    });

    c.bench_function("1face_back", |b| {
        b.iter(|| {
            //black_box(|| {
            // Run the model
            let _detections = back_model
                .predict_on_image(&image_back)
                .unwrap();
            //})
        });
    });
}

fn blaze_face_forward_benchmark(c: &mut Criterion) {
    let device = Device::Cpu;

    // Load the models
    let front_model = load_model(ModelType::Front, &device).unwrap();
    let back_model = load_model(ModelType::Back, &device).unwrap();

    // Load the image
    let image_path = "test_data/1face.png";
    let image_front = load_image(image_path, ModelType::Front, &device)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    let image_back = load_image(image_path, ModelType::Back, &device)
        .unwrap()
        .unsqueeze(0)
        .unwrap();

    c.bench_function("1face_front_forward", |b| {
        b.iter(|| {
            //black_box(|| {
            // Run the model
            let _detections = front_model
                .forward(&image_front)
                .unwrap();
            //})
        });
    });

    c.bench_function("1face_back_forward", |b| {
        b.iter(|| {
            //black_box(|| {
            // Run the model
            let _detections = back_model
                .forward(&image_back)
                .unwrap();
            //})
        });
    });
}

fn blaze_block_benchmark(c: &mut Criterion) {
    let device = Device::Cpu;
    let dtype = DType::F16;

    let single_weight_0 =
        Tensor::zeros(&[24, 1, 3, 3], dtype, &device).unwrap();
    let single_bias_0 = Tensor::zeros(&[24], dtype, &device).unwrap();
    let single_weight_1 =
        Tensor::zeros(&[24, 24, 1, 1], dtype, &device).unwrap();
    let single_bias_1 = Tensor::zeros(&[24], dtype, &device).unwrap();

    let blaze_block_single = BlazeBlock::new(
        24,
        24,
        3,
        face_tracking_rs::blaze_face::blaze_block::StrideType::Single,
        single_weight_0,
        single_bias_0,
        single_weight_1,
        single_bias_1,
    )
    .unwrap();

    let single_input = Tensor::zeros(&[1, 24, 64, 64], dtype, &device).unwrap();

    let double_weight_0 =
        Tensor::zeros(&[28, 1, 3, 3], dtype, &device).unwrap();
    let double_bias_0 = Tensor::zeros(&[28], dtype, &device).unwrap();
    let double_weight_1 =
        Tensor::zeros(&[32, 28, 1, 1], dtype, &device).unwrap();
    let double_bias_1 = Tensor::zeros(&[32], dtype, &device).unwrap();

    let blaze_block_double = BlazeBlock::new(
        28,
        32,
        3,
        face_tracking_rs::blaze_face::blaze_block::StrideType::Double,
        double_weight_0,
        double_bias_0,
        double_weight_1,
        double_bias_1,
    )
    .unwrap();

    let double_intput =
        Tensor::zeros(&[1, 28, 64, 64], dtype, &device).unwrap();

    c.bench_function("blaze_block_single", |b| {
        b.iter(|| {
            let _output = blaze_block_single
                .forward(&single_input)
                .unwrap();
        });
    });

    c.bench_function("blaze_block_double", |b| {
        b.iter(|| {
            let _output = blaze_block_double
                .forward(&double_intput)
                .unwrap();
        });
    });
}

criterion_group!(
    benches,
    blaze_face_benchmark,
    blaze_face_forward_benchmark,
    blaze_block_benchmark,
);
criterion_main!(benches);
