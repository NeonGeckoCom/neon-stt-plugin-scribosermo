

# install

- pip install ds-ctcdecoder
    - [ds_ctcdecoder-0.10.0a3-cp37-cp37m-linux_armv7l.whl](https://gitlab.com/Jaco-Assistant/Scribosermo/-/raw/master/extras/misc/ds_ctcdecoder-0.10.0a3-cp37-cp37m-linux_armv7l.whl)
- get deepspeech scorer (deepspeech-0.9.3-models.scorer) from [here](https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer)
- get models from [here](https://gitlab.com/Jaco-Assistant/Scribosermo#pretrained-checkpoints-and-language-models)

# config

```json

{
  "stt": {
     "module": "neon-stt-plugin-scribosermo",
     "neon-stt-plugin-scribosermo": {
        "model": "qnetp15/model_quantized.tflite",
        "ds_scorer": "deepspeech-0.9.3-models.scorer"
     }
  }
}
```