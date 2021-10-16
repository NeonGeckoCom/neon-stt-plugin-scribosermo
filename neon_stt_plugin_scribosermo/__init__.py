import json
from os.path import dirname
from os.path import join, isfile

import numpy as np
import tflit
from ds_ctcdecoder import Alphabet, Scorer, ctc_beam_search_decoder
from ovos_plugin_manager.templates.stt import STT
from ovos_utils.log import LOG


class ScriboSermoSTT(STT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        lang = self.config.get("lang") or "en"
        self.lang = lang.split("-")[0]
        self.checkpoint_file = self.config.get("model")
        if not self.checkpoint_file or not isfile(self.checkpoint_file):
            LOG.error("You need to provide a tflite model")
            LOG.info(
                "download model from https://gitlab.com/Jaco-Assistant/Scribosermo#pretrained-checkpoints-and-language-models")
            raise FileNotFoundError

        self.ds_scorer_path = self.config.get("ds_scorer")
        if not self.ds_scorer_path or not isfile(self.ds_scorer_path):
            LOG.error("You need to provide a language model")
            LOG.info(
                "download model from https://github.com/mozilla/DeepSpeech/releases/tag/v0.9.3")
            raise FileNotFoundError

        res_path = join(dirname(__file__), "res", lang)
        self.alphabet_path = join(res_path, "alphabet.json")
        self.ds_alphabet_path = join(res_path, "alphabet.txt")

        self.beam_size = self.config.get("beam_size") or 256
        self.alpha = self.config.get("alpha") or 0.931289039105002
        self.beta = self.config.get("beta") or 1.1834137581510284

        with open(self.alphabet_path, "r", encoding="utf-8") as file:
            self.alphabet = json.load(file)
        self.ds_alphabet = Alphabet(self.ds_alphabet_path)
        self.ds_scorer = Scorer(
            alpha=self.alpha,
            beta=self.beta,
            scorer_path=self.ds_scorer_path,
            alphabet=self.ds_alphabet,
        )
        #  Setup tflite environment
        self.model = tflit.Model(self.checkpoint_file)
        self.interpreter = self.model.interpreter

    def predict(self, audio):
        """Feed an audio signal with shape [1, len_signal]
        into the network and get the predictions"""
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        # Enable dynamic shape inputs
        self.interpreter.resize_tensor_input(input_details[0]["index"],
                                             audio.shape)
        self.interpreter.allocate_tensors()

        # Feed audio
        self.interpreter.set_tensor(input_details[0]["index"], audio)
        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(output_details[0]["index"])
        return output_data

    def lm_predict(self, prediction):
        """Decode the network's prediction with an additional language model"""
        ldecoded = ctc_beam_search_decoder(
            prediction.tolist(),
            alphabet=self.ds_alphabet,
            beam_size=self.beam_size,
            cutoff_prob=1.0,
            cutoff_top_n=512,
            scorer=self.ds_scorer,
            hot_words=dict(),
            num_results=1,
        )
        lm_text = ldecoded[0][1]
        return lm_text

    def execute(self, audio, language=None):
        audio = np.frombuffer(audio.get_wav_data(), np.int16)
        audio = audio / np.iinfo(np.int16).max
        audio = np.expand_dims(audio, axis=0)
        audio = audio.astype(np.float32)
        prediction = self.predict(audio)
        lm_text = self.lm_predict(prediction[0])
        return lm_text
