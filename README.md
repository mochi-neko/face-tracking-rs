# face-tracking-rs

Aims to implement face tracking system by [candle](https://github.com/huggingface/candle) in Rust.

## Features

- BlaceFace
    - A lightweight face detection model.
    - [Paper](https://arxiv.org/abs/1907.05047) 

## Test images credits

Images for testing ([/test_data](./test_data/)) are following:

- 1face.png: Fei Fei Li by [ITU Pictures](https://www.flickr.com/photos/itupictures/35011409612/), CC BY 2.0
- 3faces.png: Geoffrey Hinton, Yoshua Bengio, Yann Lecun. Found at [AIBuilders](https://aibuilders.ai/le-prix-turing-recompense-trois-pionniers-de-lintelligence-artificielle-yann-lecun-yoshua-bengio-et-geoffrey-hinton/)
- 4faces.png: from Andrew Ng’s Facebook page / [KDnuggets](https://www.kdnuggets.com/2015/03/talking-machine-deep-learning-gurus-p1.html)

These images were scaled down to 128x128 pixels as that is the expected input size of the model.

## Acknowledgements

- [MediaPipe](https://github.com/google/mediapipe)
- [BlazeFace-PyTorch](https://github.com/hollance/BlazeFace-PyTorch)
- [candle](https://github.com/huggingface/candle)

## License

Licensed under the [MIT](./LICENSE) license.