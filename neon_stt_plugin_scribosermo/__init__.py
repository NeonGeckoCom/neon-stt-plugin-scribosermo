import json
import multiprocessing as mp
from os import makedirs
from os.path import dirname, join, isfile

import numpy as np
import requests
import tflite_runtime.interpreter as tflite
from ds_ctcdecoder import Alphabet, Scorer, ctc_beam_search_decoder
from mediafiredl import MediafireDL as MF
from ovos_plugin_manager.templates.stt import STT
from ovos_utils.log import LOG
from ovos_utils.xdg_utils import xdg_data_home



class ModelContainer:
    DEFAULT_MODELS = {
        "en": ("en-qnet15-quantized", "en-deepspeech"),
        "de": ("de-d37cv-wer0066-quantized", "kenlm_de_all"),
        "fr": ("fr-d7cv-wer0110-quantized", "fr_pocolm_d7cv"),
        #  "es": ("es-cv-d8cv-wer0100-quantized", "kenlm_es_n12"),
        #  [ctc_beam_search_decoder.cpp:279] FATAL: "(alphabet.GetSize()+1) == (class_dim)" check failed.
        #  Number of output classes in acoustic model does not match number of labels in the alphabet file.
        #  Alphabet file must be the same one that was used to train the acoustic model.
        "it": ("it-d5cv-wer0115-quantized", "it_pocolm_d5cv")
    }
    URLS = {
        "de-cv-wer0077-full": "https://www.mediafire.com/file/27a6epdh7wxkzyx/model_full.tflite/file",
        "de-cv-wer0077-quantized": "https://www.mediafire.com/file/faehnst71byn7pc/model_quantized.tflite/file",
        "de-d37cv-wer0066-full": "https://www.mediafire.com/file/1zzdrqhvbo8xduh/model_full.tflite/file",
        "de-d37cv-wer0066-quantized": "https://www.mediafire.com/file/4jsao7xw4n6uwx5/model_quantized.tflite/file",
        "kenlm_de_tcv": "https://www.mediafire.com/file/xb2dq2roh8ckawf/kenlm_de_tcv.scorer/file",
        "kenlm_de_all": "https://www.mediafire.com/file/pzj8prgv2h0c8ue/kenlm_de_all.scorer/file",
        "de_pocolm_large": "https://www.mediafire.com/file/b64k0uqv69ehe9p/de_pocolm_large.scorer/file",

        "en-qnet5-full": "https://www.mediafire.com/file/i92e56x71oikhgq/model_full.tflite/file",
        "en-qnet5-quantized": "https://www.mediafire.com/file/77w8hr9ff073v7c/model_quantized.tflite/file",
        "en-qnet15-full": "https://www.mediafire.com/file/7q17n5ornb80ygo/model_full.tflite/file",
        "en-qnet15-quantized": "https://www.mediafire.com/file/c16vt3cbfnv2d3j/model_quantized.tflite/file",
        "en-deepspeech": "https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer",

        "es-cv-wer0105-full": "https://www.mediafire.com/file/d6sgnqfqen4s85k/model_full.tflite/file",
        "es-cv-wer0105-quantized": "https://www.mediafire.com/file/2nt2yso1w1fckzt/model_quantized.tflite/file",
        "es-cv-d8cv-wer0100-full": "https://www.mediafire.com/file/evv5z0gphw6cz1i/model_full.tflite/file",
        "es-cv-d8cv-wer0100-quantized": "https://www.mediafire.com/file/cb8r9skhx9s9205/model_quantized.tflite/file",
        "kenlm_es_n12": "https://www.mediafire.com/file/h38hmax7wnkxqfd/kenlm_es_n12.scorer/file",
        "es_pocolm_d8cv": "https://www.mediafire.com/file/pwt95u2wik8gr5s/es_pocolm_d8cv.scorer/file",

        "fr-cv-wer0121-full": "https://www.mediafire.com/file/jq5ql5z7pm9i9m2/model_full.tflite/file",
        "fr-cv-wer0121-quantized": "https://www.mediafire.com/file/1teb0i6qe9hpmm5/model_quantized.tflite/file",
        "fr-d7cv-wer0110-full": "https://www.mediafire.com/file/j9kgnmuo2m43mqm/model_full.tflite/file",
        "fr-d7cv-wer0110-quantized": "https://www.mediafire.com/file/lvscyozy45cd5ko/model_quantized.tflite/file",
        "kenlm_fr_n12": "https://www.mediafire.com/file/pcj322gp5ddpfhd/kenlm_fr_n12.scorer/file",
        "fr_pocolm_d7cv": "https://www.mediafire.com/file/55qv3bpu6z0m1p9/fr_pocolm_d7cv.scorer/file",

        "it-d5cv-wer0115-full": "https://www.mediafire.com/file/4i1a85m84idcxfm/model_full.tflite/file",
        "it-d5cv-wer0115-quantized": "https://www.mediafire.com/file/1x94edbb1892ikf/model_quantized.tflite/file",
        "it_pocolm_d5cv": "https://www.mediafire.com/file/cuf9adxqqxbqlbu/it_pocolm_d5cv.scorer/file"
    }

    def __init__(self, beam_size=256, alpha=0.931289039105002, beta=1.1834137581510284):
        self.beam_size = beam_size
        self.alpha = alpha
        self.beta = beta
        self.models = {}

    def load_lang(self, lang, checkpoint_file=None, scorer_file=None):
        lang = lang.split("-")[0]

        if not checkpoint_file:
            # default lang model
            checkpoint_file, scorer_file = download_default(lang)
        else:
            # auto download
            if checkpoint_file.startswith("http"):
                checkpoint_file = download(checkpoint_file)
            # pre defined aliases
            elif checkpoint_file in self.URLS:
                checkpoint_file = download(self.URLS[checkpoint_file])

            if score_file.startswith("http"):
                scorer_file = download(scorer_file)
            elif scorer_file in self.URLS:
                scorer_file = download(self.URLS[scorer_file])

        res_path = join(dirname(__file__), "res", lang)
        alphabet_json = join(res_path, "alphabet.json")
        alphabet_txt = join(res_path, "alphabet.txt")

        with open(alphabet_json, "r", encoding="utf-8") as file:
            alphabet_json = json.load(file)

        alphabet = Alphabet(alphabet_txt)

        scorer = Scorer(
            alpha=self.alpha,
            beta=self.beta,
            scorer_path=scorer_file,
            alphabet=alphabet,
        )
        #  Setup tflite environment
        interpreter = tflite.Interpreter(
            model_path=checkpoint_file, num_threads=mp.cpu_count()
        )

        self.models[lang] = (interpreter, scorer, alphabet, alphabet_json)

    def unload_lang(self, lang):
        if lang in self.models:
            del self.models[lang]

    def predict(self, audio, lang):
        """Feed an audio signal with shape [1, len_signal]
        into the network and get the predictions"""
        lang = lang.split("-")[0]
        if lang not in self.models:
            self.load_lang(lang)
        interpreter, scorer, alphabet, alphabet_json = self.models[lang]

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Enable dynamic shape inputs
        interpreter.resize_tensor_input(input_details[0]["index"], audio.shape)
        interpreter.allocate_tensors()

        # Feed audio
        interpreter.set_tensor(input_details[0]["index"], audio)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]["index"])
        return output_data

    def lm_predict(self, prediction, lang):
        """Decode the network's prediction with an additional language engine"""
        lang = lang.split("-")[0]
        if lang not in self.models:
            self.load_lang(lang)
        interpreter, scorer, alphabet, alphabet_json = self.models[lang]

        ldecoded = ctc_beam_search_decoder(
            prediction.tolist(),
            alphabet=alphabet,
            beam_size=self.beam_size,
            cutoff_prob=1.0,
            cutoff_top_n=512,
            scorer=scorer,
            hot_words=dict(),
            num_results=1,
        )
        lm_text = ldecoded[0][1]
        return lm_text

    def predict_audio(self, audio, lang):
        lang = lang.split("-")[0]
        if lang not in self.models:
            self.load_lang(lang)
        audio = np.frombuffer(audio.get_wav_data(), np.int16)
        audio = audio / np.iinfo(np.int16).max
        audio = np.expand_dims(audio, axis=0)
        audio = audio.astype(np.float32)
        prediction = self.predict(audio, lang)
        lm_text = self.lm_predict(prediction[0], lang)
        return lm_text


class ScriboSermoSTT(STT):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lang = self.lang.split("-")[0]
        self.beam_size = self.config.get("beam_size") or 256
        self.alpha = self.config.get("alpha") or 0.931289039105002
        self.beta = self.config.get("beta") or 1.1834137581510284
        self.engine = ModelContainer(self.beam_size, self.alpha, self.beta)

        local_model = self.config.get("local_model")
        if local_model:
            self.engine.load_lang(local_model.get("lang") or self.lang,
                                  local_model["checkpoint"],
                                  local_model["scorer"])

        if self.lang not in self.engine.models:
            self.engine.load_lang(self.lang)

        for lang in self.config.get("preloaded_langs") or []:
            if lang not in self.engine.models:
                self.engine.load_lang(lang)

    def execute(self, audio, language=None):
        lang = language or self.lang
        return self.engine.predict_audio(audio, lang)


def download(url):
    base_path = join(xdg_data_home(), "scribosermo")
    makedirs(base_path, exist_ok=True)

    if "mediafire.com" in url:
        file_name = MF.GetName(url)
    else:
        file_name = url.split("/")[-1]

    model_path = join(base_path, file_name)
    if isfile(model_path):
        LOG.debug("Model exists, skipping download")
        return model_path

    LOG.info(f"Downloading {url}")

    if "mediafire.com" in url:
        return MF.Download(url, output=base_path)
    else:
        data = requests.get(url).content
        with open(model_path, "wb") as f:
            f.write(data)
    return model_path


def download_default(lang="en"):
    lang = lang.split("-")[0].lower()
    # TODO custom exception for invalid lang
    model_url, scorer_url = ModelContainer.DEFAULT_MODELS[lang]
    model = download(ModelContainer.URLS[model_url])
    scorer = download(ModelContainer.URLS[scorer_url])
    return model, scorer


def download_all():
    print(download_default("en"))
    print(download_default("es"))
    print(download_default("de"))
    print(download_default("fr"))
    print(download_default("it"))


if __name__ == "__main__":
    download_all()
