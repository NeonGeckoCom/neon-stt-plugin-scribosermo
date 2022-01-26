import json
from os.path import join, exists
from os import listdir
import os
import multiprocessing as mp
import wave
import numpy as np
import soundfile as sf
import tflite_runtime.interpreter as tflite
from ovos_plugin_manager.templates.stt import STT
from timeit import default_timer as timer
from neon_utils.logger import LOG
from xdg import BaseDirectory as XDG
from neon_sftp.connector import NeonSFTPConnector
# If you want to improve the transcriptions with an additional language model, without using the
# training container, you can find a prebuilt pip-package in the published assets here:
# https://github.com/mozilla/DeepSpeech/releases/tag/v0.9.3
# or for use on a Raspberry Pi you can use the one from extras/misc directory
from ds_ctcdecoder import Alphabet, Scorer, swigwrapper

# ==================================================================================================
class ScriboSermoSTT(STT):

    def __init__(self, lang, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lang = lang or "en"
        print(lang)
        model, scorer = self.download_model()
        self.checkpoint_file = model
        self.ds_scorer_path = scorer
        ROOT_DIR = os.path.dirname(os.path.realpath(__file__)) + '/res/' + lang
        self.alphabet_path = join(ROOT_DIR, [f for f in listdir(ROOT_DIR) if f.endswith('.json')][0])
        self.ds_alphabet_path = join(ROOT_DIR, [f for f in listdir(ROOT_DIR) if f.endswith('.txt')][0])
        self.beam_size = 1024
        self.sample_rate = 16000
        self.ds_alphabet = Alphabet(self.ds_alphabet_path)

    def download_model(self):
        '''
        Downloading model and scorer for the specific language
        from server using NeonSFTPConnector plugin.
        Creating a folder  'polyglot_models' in xdg_data_home
        Creating a language folder in 'polyglot_models' folder
        '''
        folder = join(XDG.xdg_data_home, 'scribosermo_models/')+self.lang
        graph = folder + '/model_quantized.tflite'
        scorer = folder + '/models.scorer'
        if not exists(folder):
            if exists(join(XDG.xdg_data_home, 'scribosermo_models')):
                os.mkdir(folder)
            else:
                os.mkdir(join(XDG.xdg_data_home, 'scribosermo_models'))
                os.mkdir(folder)
            LOG.info(f"Downloading model for scribosermo ...")
            LOG.info("this might take a while")
            with open(os.environ.get('SFTP_CREDS_PATH', 'sftp_config.json')) as f:
                sftp_creds = json.load(f)
                NeonSFTPConnector.connector = NeonSFTPConnector(**sftp_creds)
            get_graph = '/scribosermo/'+self.lang+'/model_quantized.tflite'
            get_scorer = '/scribosermo/'+self.lang+'/models.scorer'
            NeonSFTPConnector.connector.get_file(get_from=get_graph, save_to=graph)
            LOG.info(f"Model downloaded to {folder}")
            NeonSFTPConnector.connector.get_file(get_from=get_scorer, save_to=scorer)
            LOG.info(f"Scorer downloaded to {folder}")
            model_path = graph
            scorer_file_path = scorer
        else:
            LOG.info(f"Model exists {folder}")
            model_path = graph
            scorer_file_path = scorer
        return model_path, scorer_file_path

    def load_audio(self, wav_path):
        """Load wav file with the required format"""

        audio, _ = sf.read(wav_path, dtype="int16")
        audio = audio / np.iinfo(np.int16).max
        audio = np.expand_dims(audio, axis=0)
        audio = audio.astype(np.float32)
        return audio

    def predict(self, interpreter, audio):
        """Feed an audio signal with shape [1, len_signal] into the network and get the predictions"""

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

    def feed_chunk(self,
        chunk: np.array, overlap: int, offset: int, interpreter, decoder
    ) -> None:
        """Feed an audio chunk with shape [1, len_chunk] into the decoding process"""

        # Get network prediction for chunk
        prediction = self.predict(interpreter, chunk)
        prediction = prediction[0]

        # Extract the interesting part in the middle of the prediction
        timesteps_overlap = int(len(prediction) / (chunk.shape[1] / overlap)) - 2
        prediction = prediction[timesteps_overlap:-timesteps_overlap]

        # Apply some offset for improved results
        prediction = prediction[: len(prediction) - offset]

        # Feed into decoder
        decoder.next(prediction.tolist())

    def decode(self, decoder):
        """Get decoded prediction and convert to text"""
        results = decoder.decode(num_results=1)
        results = [(res.confidence, self.ds_alphabet.Decode(res.tokens)) for res in results]

        lm_text = results[0][1]
        return lm_text

    def execute(self, wav_path, language=None):
        inference_start = timer()
        chunk_size = int(1.0 * self.sample_rate)
        frame_overlap = int(2.0 * self.sample_rate)
        char_offset = 4

        with open(self.alphabet_path, "r", encoding="utf-8") as file:
            alphabet = json.load(file)

        ds_scorer = Scorer(
            alpha=0.931289039105002,
            beta=1.1834137581510284,
            scorer_path=self.ds_scorer_path,
            alphabet=self.ds_alphabet,
        )
        ds_decoder = swigwrapper.DecoderState()
        ds_decoder.init(
            alphabet=self.ds_alphabet,
            beam_size=self.beam_size,
            cutoff_prob=1.0,
            cutoff_top_n=512,
            ext_scorer=ds_scorer,
            hot_words=dict(),
        )
        LOG.info("Loading model ...")
        interpreter = tflite.Interpreter(
            model_path=self.checkpoint_file, num_threads=mp.cpu_count()
        )
        """Transcribe an audio file chunk by chunk"""

        # For reasons of simplicity, a wav-file is used instead of a microphone stream
        audio = self.load_audio(wav_path)
        audio = audio[0]

        # Add some empty padding that the last words are not cut from the transcription
        audio = np.concatenate([audio, np.zeros(shape=frame_overlap, dtype=np.float32)])

        start = 0
        buffer = np.zeros(shape=2 * frame_overlap + chunk_size, dtype=np.float32)
        while start < len(audio):

            # Cut a chunk from the complete audio signal
            stop = min(len(audio), start + chunk_size)
            chunk = audio[start:stop]
            start = stop

            # Add new frames to the end of the buffer
            buffer = buffer[chunk_size:]
            buffer = np.concatenate([buffer, chunk])

            # Now feed this frame into the decoding process
            ibuffer = np.expand_dims(buffer, axis=0)
            self.feed_chunk(ibuffer, frame_overlap, char_offset, interpreter, ds_decoder)

        # Get the text after the stream is finished
        text = self.decode(ds_decoder)
        LOG.info("Prediction scorer: {}".format(text))

        # estimated time
        fin = wave.open(wav_path, 'rb')
        fs_orig = fin.getframerate()
        audio_length = fin.getnframes() * (1 / fs_orig)
        inference_end = timer() - inference_start
        LOG.info('Inference took %0.3fs for %0.3fs audio file.' % (inference_end, audio_length))
        return text


if __name__ == "__main__":

    stt = ScriboSermoSTT('de')
    # print("Running transcription ...\n")
    # test_wav_path = '/home/mary/PycharmProjects/neon-stt-plugin-scribosermo/neon_stt_plugin_scribosermo/tests/test_audio/de/guten_tag_female.wav'
    # stt.execute(test_wav_path)
    # print("FINISHED")
