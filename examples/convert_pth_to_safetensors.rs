use std::collections::HashMap;

use candle_core::{safetensors, DType, Device, Result, Tensor};
use candle_nn::VarBuilder;

fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let dtype = DType::F16;

    let front_model_path = "src/blaze_face/data/blazeface.pth";

    let front_model =
        candle_nn::VarBuilder::from_pth(front_model_path, dtype, &device)?;

    let front_hash_map = load_hash_map_front(&front_model)?;

    safetensors::save(
        &front_hash_map,
        "src/blaze_face/data/blazeface.safetensors",
    )?;

    let back_model_path = "src/blaze_face/data/blazefaceback.pth";

    let back_model =
        candle_nn::VarBuilder::from_pth(back_model_path, dtype, &device)?;

    let back_hash_map = load_hash_map_back(&back_model)?;

    safetensors::save(
        &back_hash_map,
        "src/blaze_face/data/blazefaceback.safetensors",
    )?;

    Ok(())
}

fn load_hash_map_front(
    variables: &VarBuilder
) -> Result<HashMap<String, Tensor>> {
    let mut hash_map = HashMap::new();

    hash_map.insert(
        "backbone1.0.weight".to_string(),
        variables.get_with_hints(
            (24, 3, 5, 5),
            "backbone1.0.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone1.0.bias".to_string(),
        variables.get_with_hints(
            24,
            "backbone1.0.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "backbone1.2.convs.0.weight".to_string(),
        variables.get_with_hints(
            (24, 1, 3, 3),
            "backbone1.2.convs.0.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone1.2.convs.0.bias".to_string(),
        variables.get_with_hints(
            24,
            "backbone1.2.convs.0.bias",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone1.2.convs.1.weight".to_string(),
        variables.get_with_hints(
            (24, 24, 1, 1),
            "backbone1.2.convs.1.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone1.2.convs.1.bias".to_string(),
        variables.get_with_hints(
            24,
            "backbone1.2.convs.1.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "backbone1.3.convs.0.weight".to_string(),
        variables.get_with_hints(
            (24, 1, 3, 3),
            "backbone1.3.convs.0.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone1.3.convs.0.bias".to_string(),
        variables.get_with_hints(
            24,
            "backbone1.3.convs.0.bias",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone1.3.convs.1.weight".to_string(),
        variables.get_with_hints(
            (28, 24, 1, 1),
            "backbone1.3.convs.1.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone1.3.convs.1.bias".to_string(),
        variables.get_with_hints(
            28,
            "backbone1.3.convs.1.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "backbone1.4.convs.0.weight".to_string(),
        variables.get_with_hints(
            (28, 1, 3, 3),
            "backbone1.4.convs.0.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone1.4.convs.0.bias".to_string(),
        variables.get_with_hints(
            28,
            "backbone1.4.convs.0.bias",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone1.4.convs.1.weight".to_string(),
        variables.get_with_hints(
            (32, 28, 1, 1),
            "backbone1.4.convs.1.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone1.4.convs.1.bias".to_string(),
        variables.get_with_hints(
            32,
            "backbone1.4.convs.1.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "backbone1.5.convs.0.weight".to_string(),
        variables.get_with_hints(
            (32, 1, 3, 3),
            "backbone1.5.convs.0.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone1.5.convs.0.bias".to_string(),
        variables.get_with_hints(
            32,
            "backbone1.5.convs.0.bias",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone1.5.convs.1.weight".to_string(),
        variables.get_with_hints(
            (36, 32, 1, 1),
            "backbone1.5.convs.1.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone1.5.convs.1.bias".to_string(),
        variables.get_with_hints(
            36,
            "backbone1.5.convs.1.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "backbone1.6.convs.0.weight".to_string(),
        variables.get_with_hints(
            (36, 1, 3, 3),
            "backbone1.6.convs.0.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone1.6.convs.0.bias".to_string(),
        variables.get_with_hints(
            36,
            "backbone1.6.convs.0.bias",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone1.6.convs.1.weight".to_string(),
        variables.get_with_hints(
            (42, 36, 1, 1),
            "backbone1.6.convs.1.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone1.6.convs.1.bias".to_string(),
        variables.get_with_hints(
            42,
            "backbone1.6.convs.1.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "backbone1.7.convs.0.weight".to_string(),
        variables.get_with_hints(
            (42, 1, 3, 3),
            "backbone1.7.convs.0.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone1.7.convs.0.bias".to_string(),
        variables.get_with_hints(
            42,
            "backbone1.7.convs.0.bias",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone1.7.convs.1.weight".to_string(),
        variables.get_with_hints(
            (48, 42, 1, 1),
            "backbone1.7.convs.1.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone1.7.convs.1.bias".to_string(),
        variables.get_with_hints(
            48,
            "backbone1.7.convs.1.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "backbone1.8.convs.0.weight".to_string(),
        variables.get_with_hints(
            (48, 1, 3, 3),
            "backbone1.8.convs.0.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone1.8.convs.0.bias".to_string(),
        variables.get_with_hints(
            48,
            "backbone1.8.convs.0.bias",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone1.8.convs.1.weight".to_string(),
        variables.get_with_hints(
            (56, 48, 1, 1),
            "backbone1.8.convs.1.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone1.8.convs.1.bias".to_string(),
        variables.get_with_hints(
            56,
            "backbone1.8.convs.1.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "backbone1.9.convs.0.weight".to_string(),
        variables.get_with_hints(
            (56, 1, 3, 3),
            "backbone1.9.convs.0.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone1.9.convs.0.bias".to_string(),
        variables.get_with_hints(
            56,
            "backbone1.9.convs.0.bias",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone1.9.convs.1.weight".to_string(),
        variables.get_with_hints(
            (64, 56, 1, 1),
            "backbone1.9.convs.1.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone1.9.convs.1.bias".to_string(),
        variables.get_with_hints(
            64,
            "backbone1.9.convs.1.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "backbone1.10.convs.0.weight".to_string(),
        variables.get_with_hints(
            (64, 1, 3, 3),
            "backbone1.10.convs.0.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone1.10.convs.0.bias".to_string(),
        variables.get_with_hints(
            64,
            "backbone1.10.convs.0.bias",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone1.10.convs.1.weight".to_string(),
        variables.get_with_hints(
            (72, 64, 1, 1),
            "backbone1.10.convs.1.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone1.10.convs.1.bias".to_string(),
        variables.get_with_hints(
            72,
            "backbone1.10.convs.1.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "backbone1.11.convs.0.weight".to_string(),
        variables.get_with_hints(
            (72, 1, 3, 3),
            "backbone1.11.convs.0.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone1.11.convs.0.bias".to_string(),
        variables.get_with_hints(
            72,
            "backbone1.11.convs.0.bias",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone1.11.convs.1.weight".to_string(),
        variables.get_with_hints(
            (80, 72, 1, 1),
            "backbone1.11.convs.1.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone1.11.convs.1.bias".to_string(),
        variables.get_with_hints(
            80,
            "backbone1.11.convs.1.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "backbone1.12.convs.0.weight".to_string(),
        variables.get_with_hints(
            (80, 1, 3, 3),
            "backbone1.12.convs.0.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone1.12.convs.0.bias".to_string(),
        variables.get_with_hints(
            80,
            "backbone1.12.convs.0.bias",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone1.12.convs.1.weight".to_string(),
        variables.get_with_hints(
            (88, 80, 1, 1),
            "backbone1.12.convs.1.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone1.12.convs.1.bias".to_string(),
        variables.get_with_hints(
            88,
            "backbone1.12.convs.1.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "backbone2.0.convs.0.weight".to_string(),
        variables.get_with_hints(
            (88, 1, 3, 3),
            "backbone2.0.convs.0.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone2.0.convs.0.bias".to_string(),
        variables.get_with_hints(
            88,
            "backbone2.0.convs.0.bias",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone2.0.convs.1.weight".to_string(),
        variables.get_with_hints(
            (96, 88, 1, 1),
            "backbone2.0.convs.1.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone2.0.convs.1.bias".to_string(),
        variables.get_with_hints(
            96,
            "backbone2.0.convs.1.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "backbone2.1.convs.0.weight".to_string(),
        variables.get_with_hints(
            (96, 1, 3, 3),
            "backbone2.1.convs.0.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone2.1.convs.0.bias".to_string(),
        variables.get_with_hints(
            96,
            "backbone2.1.convs.0.bias",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone2.1.convs.1.weight".to_string(),
        variables.get_with_hints(
            (96, 96, 1, 1),
            "backbone2.1.convs.1.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone2.1.convs.1.bias".to_string(),
        variables.get_with_hints(
            96,
            "backbone2.1.convs.1.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "backbone2.2.convs.0.weight".to_string(),
        variables.get_with_hints(
            (96, 1, 3, 3),
            "backbone2.2.convs.0.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone2.2.convs.0.bias".to_string(),
        variables.get_with_hints(
            96,
            "backbone2.2.convs.0.bias",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone2.2.convs.1.weight".to_string(),
        variables.get_with_hints(
            (96, 96, 1, 1),
            "backbone2.2.convs.1.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone2.2.convs.1.bias".to_string(),
        variables.get_with_hints(
            96,
            "backbone2.2.convs.1.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "backbone2.3.convs.0.weight".to_string(),
        variables.get_with_hints(
            (96, 1, 3, 3),
            "backbone2.3.convs.0.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone2.3.convs.0.bias".to_string(),
        variables.get_with_hints(
            96,
            "backbone2.3.convs.0.bias",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone2.3.convs.1.weight".to_string(),
        variables.get_with_hints(
            (96, 96, 1, 1),
            "backbone2.3.convs.1.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone2.3.convs.1.bias".to_string(),
        variables.get_with_hints(
            96,
            "backbone2.3.convs.1.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "backbone2.4.convs.0.weight".to_string(),
        variables.get_with_hints(
            (96, 1, 3, 3),
            "backbone2.4.convs.0.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone2.4.convs.0.bias".to_string(),
        variables.get_with_hints(
            96,
            "backbone2.4.convs.0.bias",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone2.4.convs.1.weight".to_string(),
        variables.get_with_hints(
            (96, 96, 1, 1),
            "backbone2.4.convs.1.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone2.4.convs.1.bias".to_string(),
        variables.get_with_hints(
            96,
            "backbone2.4.convs.1.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "classifier_8.weight".to_string(),
        variables.get_with_hints(
            (2, 88, 1, 1),
            "classifier_8.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "classifier_8.bias".to_string(),
        variables.get_with_hints(
            2,
            "classifier_8.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "classifier_16.weight".to_string(),
        variables.get_with_hints(
            (6, 96, 1, 1),
            "classifier_16.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "classifier_16.bias".to_string(),
        variables.get_with_hints(
            6,
            "classifier_16.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "regressor_8.weight".to_string(),
        variables.get_with_hints(
            (32, 88, 1, 1),
            "regressor_8.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "regressor_8.bias".to_string(),
        variables.get_with_hints(
            32,
            "regressor_8.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "regressor_16.weight".to_string(),
        variables.get_with_hints(
            (96, 96, 1, 1),
            "regressor_16.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "regressor_16.bias".to_string(),
        variables.get_with_hints(
            96,
            "regressor_16.bias",
            candle_nn::init::ZERO,
        )?,
    );

    Ok(hash_map)
}

fn load_hash_map_back(
    variables: &VarBuilder
) -> Result<HashMap<String, Tensor>> {
    let mut hash_map = HashMap::new();

    hash_map.insert(
        "backbone.0.weight".to_string(),
        variables.get_with_hints(
            (24, 3, 5, 5),
            "backbone.0.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.0.bias".to_string(),
        variables.get_with_hints(
            24,
            "backbone.0.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "backbone.2.convs.0.weight".to_string(),
        variables.get_with_hints(
            (24, 1, 3, 3),
            "backbone.2.convs.0.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.2.convs.0.bias".to_string(),
        variables.get_with_hints(
            24,
            "backbone.2.convs.0.bias",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.2.convs.1.weight".to_string(),
        variables.get_with_hints(
            (24, 24, 1, 1),
            "backbone.2.convs.1.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.2.convs.1.bias".to_string(),
        variables.get_with_hints(
            24,
            "backbone.2.convs.1.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "backbone.3.convs.0.weight".to_string(),
        variables.get_with_hints(
            (24, 1, 3, 3),
            "backbone.3.convs.0.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.3.convs.0.bias".to_string(),
        variables.get_with_hints(
            24,
            "backbone.3.convs.0.bias",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.3.convs.1.weight".to_string(),
        variables.get_with_hints(
            (24, 24, 1, 1),
            "backbone.3.convs.1.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.3.convs.1.bias".to_string(),
        variables.get_with_hints(
            24,
            "backbone.3.convs.1.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "backbone.4.convs.0.weight".to_string(),
        variables.get_with_hints(
            (24, 1, 3, 3),
            "backbone.4.convs.0.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.4.convs.0.bias".to_string(),
        variables.get_with_hints(
            24,
            "backbone.4.convs.0.bias",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.4.convs.1.weight".to_string(),
        variables.get_with_hints(
            (24, 24, 1, 1),
            "backbone.4.convs.1.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.4.convs.1.bias".to_string(),
        variables.get_with_hints(
            24,
            "backbone.4.convs.1.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "backbone.5.convs.0.weight".to_string(),
        variables.get_with_hints(
            (24, 1, 3, 3),
            "backbone.5.convs.0.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.5.convs.0.bias".to_string(),
        variables.get_with_hints(
            24,
            "backbone.5.convs.0.bias",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.5.convs.1.weight".to_string(),
        variables.get_with_hints(
            (24, 24, 1, 1),
            "backbone.5.convs.1.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.5.convs.1.bias".to_string(),
        variables.get_with_hints(
            24,
            "backbone.5.convs.1.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "backbone.6.convs.0.weight".to_string(),
        variables.get_with_hints(
            (24, 1, 3, 3),
            "backbone.6.convs.0.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.6.convs.0.bias".to_string(),
        variables.get_with_hints(
            24,
            "backbone.6.convs.0.bias",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.6.convs.1.weight".to_string(),
        variables.get_with_hints(
            (24, 24, 1, 1),
            "backbone.6.convs.1.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.6.convs.1.bias".to_string(),
        variables.get_with_hints(
            24,
            "backbone.6.convs.1.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "backbone.7.convs.0.weight".to_string(),
        variables.get_with_hints(
            (24, 1, 3, 3),
            "backbone.7.convs.0.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.7.convs.0.bias".to_string(),
        variables.get_with_hints(
            24,
            "backbone.7.convs.0.bias",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.7.convs.1.weight".to_string(),
        variables.get_with_hints(
            (24, 24, 1, 1),
            "backbone.7.convs.1.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.7.convs.1.bias".to_string(),
        variables.get_with_hints(
            24,
            "backbone.7.convs.1.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "backbone.8.convs.0.weight".to_string(),
        variables.get_with_hints(
            (24, 1, 3, 3),
            "backbone.8.convs.0.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.8.convs.0.bias".to_string(),
        variables.get_with_hints(
            24,
            "backbone.8.convs.0.bias",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.8.convs.1.weight".to_string(),
        variables.get_with_hints(
            (24, 24, 1, 1),
            "backbone.8.convs.1.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.8.convs.1.bias".to_string(),
        variables.get_with_hints(
            24,
            "backbone.8.convs.1.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "backbone.9.convs.0.weight".to_string(),
        variables.get_with_hints(
            (24, 1, 3, 3),
            "backbone.9.convs.0.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.9.convs.0.bias".to_string(),
        variables.get_with_hints(
            24,
            "backbone.9.convs.0.bias",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.9.convs.1.weight".to_string(),
        variables.get_with_hints(
            (24, 24, 1, 1),
            "backbone.9.convs.1.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.9.convs.1.bias".to_string(),
        variables.get_with_hints(
            24,
            "backbone.9.convs.1.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "backbone.10.convs.0.weight".to_string(),
        variables.get_with_hints(
            (24, 1, 3, 3),
            "backbone.10.convs.0.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.10.convs.0.bias".to_string(),
        variables.get_with_hints(
            24,
            "backbone.10.convs.0.bias",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.10.convs.1.weight".to_string(),
        variables.get_with_hints(
            (24, 24, 1, 1),
            "backbone.10.convs.1.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.10.convs.1.bias".to_string(),
        variables.get_with_hints(
            24,
            "backbone.10.convs.1.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "backbone.11.convs.0.weight".to_string(),
        variables.get_with_hints(
            (24, 1, 3, 3),
            "backbone.11.convs.0.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.11.convs.0.bias".to_string(),
        variables.get_with_hints(
            24,
            "backbone.11.convs.0.bias",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.11.convs.1.weight".to_string(),
        variables.get_with_hints(
            (24, 24, 1, 1),
            "backbone.11.convs.1.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.11.convs.1.bias".to_string(),
        variables.get_with_hints(
            24,
            "backbone.11.convs.1.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "backbone.12.convs.0.weight".to_string(),
        variables.get_with_hints(
            (24, 1, 3, 3),
            "backbone.12.convs.0.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.12.convs.0.bias".to_string(),
        variables.get_with_hints(
            24,
            "backbone.12.convs.0.bias",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.12.convs.1.weight".to_string(),
        variables.get_with_hints(
            (24, 24, 1, 1),
            "backbone.12.convs.1.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.12.convs.1.bias".to_string(),
        variables.get_with_hints(
            24,
            "backbone.12.convs.1.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "backbone.13.convs.0.weight".to_string(),
        variables.get_with_hints(
            (24, 1, 3, 3),
            "backbone.13.convs.0.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.13.convs.0.bias".to_string(),
        variables.get_with_hints(
            24,
            "backbone.13.convs.0.bias",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.13.convs.1.weight".to_string(),
        variables.get_with_hints(
            (24, 24, 1, 1),
            "backbone.13.convs.1.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.13.convs.1.bias".to_string(),
        variables.get_with_hints(
            24,
            "backbone.13.convs.1.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "backbone.14.convs.0.weight".to_string(),
        variables.get_with_hints(
            (24, 1, 3, 3),
            "backbone.14.convs.0.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.14.convs.0.bias".to_string(),
        variables.get_with_hints(
            24,
            "backbone.14.convs.0.bias",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.14.convs.1.weight".to_string(),
        variables.get_with_hints(
            (24, 24, 1, 1),
            "backbone.14.convs.1.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.14.convs.1.bias".to_string(),
        variables.get_with_hints(
            24,
            "backbone.14.convs.1.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "backbone.15.convs.0.weight".to_string(),
        variables.get_with_hints(
            (24, 1, 3, 3),
            "backbone.15.convs.0.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.15.convs.0.bias".to_string(),
        variables.get_with_hints(
            24,
            "backbone.15.convs.0.bias",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.15.convs.1.weight".to_string(),
        variables.get_with_hints(
            (24, 24, 1, 1),
            "backbone.15.convs.1.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.15.convs.1.bias".to_string(),
        variables.get_with_hints(
            24,
            "backbone.15.convs.1.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "backbone.16.convs.0.weight".to_string(),
        variables.get_with_hints(
            (24, 1, 3, 3),
            "backbone.16.convs.0.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.16.convs.0.bias".to_string(),
        variables.get_with_hints(
            24,
            "backbone.16.convs.0.bias",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.16.convs.1.weight".to_string(),
        variables.get_with_hints(
            (24, 24, 1, 1),
            "backbone.16.convs.1.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.16.convs.1.bias".to_string(),
        variables.get_with_hints(
            24,
            "backbone.16.convs.1.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "backbone.17.convs.0.weight".to_string(),
        variables.get_with_hints(
            (24, 1, 3, 3),
            "backbone.17.convs.0.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.17.convs.0.bias".to_string(),
        variables.get_with_hints(
            24,
            "backbone.17.convs.0.bias",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.17.convs.1.weight".to_string(),
        variables.get_with_hints(
            (48, 24, 1, 1),
            "backbone.17.convs.1.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.17.convs.1.bias".to_string(),
        variables.get_with_hints(
            48,
            "backbone.17.convs.1.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "backbone.18.convs.0.weight".to_string(),
        variables.get_with_hints(
            (48, 1, 3, 3),
            "backbone.18.convs.0.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.18.convs.0.bias".to_string(),
        variables.get_with_hints(
            48,
            "backbone.18.convs.0.bias",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.18.convs.1.weight".to_string(),
        variables.get_with_hints(
            (48, 48, 1, 1),
            "backbone.18.convs.1.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.18.convs.1.bias".to_string(),
        variables.get_with_hints(
            48,
            "backbone.18.convs.1.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "backbone.19.convs.0.weight".to_string(),
        variables.get_with_hints(
            (48, 1, 3, 3),
            "backbone.19.convs.0.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.19.convs.0.bias".to_string(),
        variables.get_with_hints(
            48,
            "backbone.19.convs.0.bias",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.19.convs.1.weight".to_string(),
        variables.get_with_hints(
            (48, 48, 1, 1),
            "backbone.19.convs.1.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.19.convs.1.bias".to_string(),
        variables.get_with_hints(
            48,
            "backbone.19.convs.1.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "backbone.20.convs.0.weight".to_string(),
        variables.get_with_hints(
            (48, 1, 3, 3),
            "backbone.20.convs.0.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.20.convs.0.bias".to_string(),
        variables.get_with_hints(
            48,
            "backbone.20.convs.0.bias",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.20.convs.1.weight".to_string(),
        variables.get_with_hints(
            (48, 48, 1, 1),
            "backbone.20.convs.1.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.20.convs.1.bias".to_string(),
        variables.get_with_hints(
            48,
            "backbone.20.convs.1.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "backbone.21.convs.0.weight".to_string(),
        variables.get_with_hints(
            (48, 1, 3, 3),
            "backbone.21.convs.0.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.21.convs.0.bias".to_string(),
        variables.get_with_hints(
            48,
            "backbone.21.convs.0.bias",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.21.convs.1.weight".to_string(),
        variables.get_with_hints(
            (48, 48, 1, 1),
            "backbone.21.convs.1.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.21.convs.1.bias".to_string(),
        variables.get_with_hints(
            48,
            "backbone.21.convs.1.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "backbone.22.convs.0.weight".to_string(),
        variables.get_with_hints(
            (48, 1, 3, 3),
            "backbone.22.convs.0.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.22.convs.0.bias".to_string(),
        variables.get_with_hints(
            48,
            "backbone.22.convs.0.bias",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.22.convs.1.weight".to_string(),
        variables.get_with_hints(
            (48, 48, 1, 1),
            "backbone.22.convs.1.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.22.convs.1.bias".to_string(),
        variables.get_with_hints(
            48,
            "backbone.22.convs.1.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "backbone.23.convs.0.weight".to_string(),
        variables.get_with_hints(
            (48, 1, 3, 3),
            "backbone.23.convs.0.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.23.convs.0.bias".to_string(),
        variables.get_with_hints(
            48,
            "backbone.23.convs.0.bias",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.23.convs.1.weight".to_string(),
        variables.get_with_hints(
            (48, 48, 1, 1),
            "backbone.23.convs.1.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.23.convs.1.bias".to_string(),
        variables.get_with_hints(
            48,
            "backbone.23.convs.1.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "backbone.24.convs.0.weight".to_string(),
        variables.get_with_hints(
            (48, 1, 3, 3),
            "backbone.24.convs.0.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.24.convs.0.bias".to_string(),
        variables.get_with_hints(
            48,
            "backbone.24.convs.0.bias",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.24.convs.1.weight".to_string(),
        variables.get_with_hints(
            (48, 48, 1, 1),
            "backbone.24.convs.1.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.24.convs.1.bias".to_string(),
        variables.get_with_hints(
            48,
            "backbone.24.convs.1.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "backbone.25.convs.0.weight".to_string(),
        variables.get_with_hints(
            (48, 1, 3, 3),
            "backbone.25.convs.0.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.25.convs.0.bias".to_string(),
        variables.get_with_hints(
            48,
            "backbone.25.convs.0.bias",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.25.convs.1.weight".to_string(),
        variables.get_with_hints(
            (96, 48, 1, 1),
            "backbone.25.convs.1.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.25.convs.1.bias".to_string(),
        variables.get_with_hints(
            96,
            "backbone.25.convs.1.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "backbone.26.convs.0.weight".to_string(),
        variables.get_with_hints(
            (96, 1, 3, 3),
            "backbone.26.convs.0.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.26.convs.0.bias".to_string(),
        variables.get_with_hints(
            96,
            "backbone.26.convs.0.bias",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.26.convs.1.weight".to_string(),
        variables.get_with_hints(
            (96, 96, 1, 1),
            "backbone.26.convs.1.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.26.convs.1.bias".to_string(),
        variables.get_with_hints(
            96,
            "backbone.26.convs.1.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "backbone.27.convs.0.weight".to_string(),
        variables.get_with_hints(
            (96, 1, 3, 3),
            "backbone.27.convs.0.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.27.convs.0.bias".to_string(),
        variables.get_with_hints(
            96,
            "backbone.27.convs.0.bias",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.27.convs.1.weight".to_string(),
        variables.get_with_hints(
            (96, 96, 1, 1),
            "backbone.27.convs.1.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.27.convs.1.bias".to_string(),
        variables.get_with_hints(
            96,
            "backbone.27.convs.1.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "backbone.28.convs.0.weight".to_string(),
        variables.get_with_hints(
            (96, 1, 3, 3),
            "backbone.28.convs.0.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.28.convs.0.bias".to_string(),
        variables.get_with_hints(
            96,
            "backbone.28.convs.0.bias",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.28.convs.1.weight".to_string(),
        variables.get_with_hints(
            (96, 96, 1, 1),
            "backbone.28.convs.1.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.28.convs.1.bias".to_string(),
        variables.get_with_hints(
            96,
            "backbone.28.convs.1.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "backbone.29.convs.0.weight".to_string(),
        variables.get_with_hints(
            (96, 1, 3, 3),
            "backbone.29.convs.0.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.29.convs.0.bias".to_string(),
        variables.get_with_hints(
            96,
            "backbone.29.convs.0.bias",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.29.convs.1.weight".to_string(),
        variables.get_with_hints(
            (96, 96, 1, 1),
            "backbone.29.convs.1.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.29.convs.1.bias".to_string(),
        variables.get_with_hints(
            96,
            "backbone.29.convs.1.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "backbone.30.convs.0.weight".to_string(),
        variables.get_with_hints(
            (96, 1, 3, 3),
            "backbone.30.convs.0.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.30.convs.0.bias".to_string(),
        variables.get_with_hints(
            96,
            "backbone.30.convs.0.bias",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.30.convs.1.weight".to_string(),
        variables.get_with_hints(
            (96, 96, 1, 1),
            "backbone.30.convs.1.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.30.convs.1.bias".to_string(),
        variables.get_with_hints(
            96,
            "backbone.30.convs.1.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "backbone.31.convs.0.weight".to_string(),
        variables.get_with_hints(
            (96, 1, 3, 3),
            "backbone.31.convs.0.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.31.convs.0.bias".to_string(),
        variables.get_with_hints(
            96,
            "backbone.31.convs.0.bias",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.31.convs.1.weight".to_string(),
        variables.get_with_hints(
            (96, 96, 1, 1),
            "backbone.31.convs.1.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.31.convs.1.bias".to_string(),
        variables.get_with_hints(
            96,
            "backbone.31.convs.1.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "backbone.32.convs.0.weight".to_string(),
        variables.get_with_hints(
            (96, 1, 3, 3),
            "backbone.32.convs.0.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.32.convs.0.bias".to_string(),
        variables.get_with_hints(
            96,
            "backbone.32.convs.0.bias",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.32.convs.1.weight".to_string(),
        variables.get_with_hints(
            (96, 96, 1, 1),
            "backbone.32.convs.1.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "backbone.32.convs.1.bias".to_string(),
        variables.get_with_hints(
            96,
            "backbone.32.convs.1.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "final.convs.0.weight".to_string(),
        variables.get_with_hints(
            (96, 1, 3, 3),
            "final.convs.0.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "final.convs.0.bias".to_string(),
        variables.get_with_hints(
            96,
            "final.convs.0.bias",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "final.convs.1.weight".to_string(),
        variables.get_with_hints(
            (96, 96, 1, 1),
            "final.convs.1.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "final.convs.1.bias".to_string(),
        variables.get_with_hints(
            96,
            "final.convs.1.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "classifier_8.weight".to_string(),
        variables.get_with_hints(
            (2, 96, 1, 1),
            "classifier_8.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "classifier_8.bias".to_string(),
        variables.get_with_hints(
            2,
            "classifier_8.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "classifier_16.weight".to_string(),
        variables.get_with_hints(
            (6, 96, 1, 1),
            "classifier_16.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "classifier_16.bias".to_string(),
        variables.get_with_hints(
            6,
            "classifier_16.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "regressor_8.weight".to_string(),
        variables.get_with_hints(
            (32, 96, 1, 1),
            "regressor_8.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "regressor_8.bias".to_string(),
        variables.get_with_hints(
            32,
            "regressor_8.bias",
            candle_nn::init::ZERO,
        )?,
    );

    hash_map.insert(
        "regressor_16.weight".to_string(),
        variables.get_with_hints(
            (96, 96, 1, 1),
            "regressor_16.weight",
            candle_nn::init::ZERO,
        )?,
    );
    hash_map.insert(
        "regressor_16.bias".to_string(),
        variables.get_with_hints(
            96,
            "regressor_16.bias",
            candle_nn::init::ZERO,
        )?,
    );

    Ok(hash_map)
}
