learn about used models [here](https://gitlab.com/Jaco-Assistant/Scribosermo#pretrained-checkpoints-and-language-models)


# install

- pip install ds-ctcdecoder
    - [ds_ctcdecoder-0.10.0a3-cp37-cp37m-linux_armv7l.whl](https://gitlab.com/Jaco-Assistant/Scribosermo/-/raw/master/extras/misc/ds_ctcdecoder-0.10.0a3-cp37-cp37m-linux_armv7l.whl)

# config

```json

{
  "stt": {
     "module": "neon-stt-plugin-scribosermo"
  }
}
```

## Docker

This plugin can be used together with [ovos-stt-http-server](https://github.com/OpenVoiceOS/ovos-stt-http-server) 

```bash
docker run -p 8080:8080 ghcr.io/neongeckocom/scribosermo-stt:master
```