import json
import argparse
import logging
import time
import os
import io

import torch
from flask import Flask, request, abort, Response, jsonify, send_file
from flask_cors import CORS
import numpy as np

from utils.utils import load_config, ConfigDict, setup_model
from utils.audio import AudioProcessor
from utils.synthesis import synthesis
from utils.text.symbols import symbols, phonemes, symbols_FR

#app = Flask(__name__)

class EndpointAction:
    """Wrapper to use class method as endpoint for flask"""
    def __init__(self, action):
        self.action = action

    def __call__(self, *args):
        # Perform the action
        answer = self.action(*args)
        # Create the answer (bundle it in a correctly formatted HTTP answer)
        if isinstance(answer, str):
            self.response = Response(answer, status=200, headers={})
        elif isinstance(answer, bytes):
            self.response = Response()
            self.response.mimetype = "audio/wav"
            self.response.data = answer
        else:
            self.response = jsonify(answer)
        # Send it
        return self.response

class TTS_Worker:
    """Running class for TTS. """
    def __init__(self, config_path: str, model_path:str, service_port: int, cuda: bool):
        """ Create a TTS worker with a REST API.

        Supported model format are :
        - Tacotron / Griffin-Lim
        Keyword arguments:
        ==================
        config_path (str) -- Path to the config file containing audio and model parameters.

        model_path (str) -- Path to the model file.

        service_port (int) -- Port on which the API is served.
        """
        self.service_port = service_port
        try:
            self.config = load_config(config_path)
        except Exception as e:
            logging.error("Could not read config file at {}: {}".format(config_path, e.args))
            exit(-1)
        self.config.forward_attn_mask = True
        
        # Audio processor
        self.audio_processor = AudioProcessor(**self.config.audio)
        
        # Model
        self.model_name = os.path.basename(model_path)
        try:
            self.model = self._load_model(model_path, cuda=cuda)
        except Exception as e:
            logging.error("Could not load TTS model at {}: {}".format(model_path, e.args))
            self.error = "Could not load TTS model"
            self.ready = False
        else:
            self.ready = True
        self.busy = False

        # Server
        self.app = Flask('LinTTS')
        CORS(self.app)
        self.create_endpoints()

    def _load_model(self, model_path: str, cuda: bool = True):
        """ Loads the model based on model path and configuration"""
        if self.config['phoneme_language'] == 'en-us':
            num_chars = len(phonemes) if self.config.use_phonemes else len(symbols)
        elif self.config['phoneme_language'] == 'fr-fr':
            num_chars = len(symbols_FR)
        else:
            raise NotImplementedError("{} language not supported".format(self.config['phoneme_language']))
            exit(-1)
        
        model = setup_model(num_chars, self.config)
        cp = torch.load(model_path)
        model.load_state_dict(cp['model'])
        model.eval()
        if cuda:
            model.cuda()
        model.decoder.set_r(cp['r'])
        return model

    def _add_endpoint(self, endpoint=None, endpoint_name=None, handler=None, methods=['GET']):
            self.app.add_url_rule(endpoint, endpoint_name, EndpointAction(handler), methods=methods)

    def create_endpoints(self):
        """ Add API endpoints 
        TODO: Set endpoints in separate file
        """
        self._add_endpoint(endpoint='/synthesize', endpoint_name='/synthesize', handler=self._synthesize, methods=['POST'])
        self._add_endpoint(endpoint='/check', endpoint_name='/check', handler=self._check)
        self._add_endpoint(endpoint='/stop', endpoint_name='/stop', handler=self._stop)
        
    def _synthesize(self) -> bytes:
        """ /synthesize endpoint.
        Retrieves the input 'text' field of a POST request and synthesize a wav audio file from it.
        """
        self.busy = True
        body = request.form
        if not 'text' in body.keys():
            self.busy = False
            abort(400)
        text = body['text']

        logging.debug("Synthetizing sentence: {}".format(text))
        t_start = time.time()
        waveform = synthesis(self.model, text, self.config, True, self.audio_processor, None, False, self.config.enable_eos_bos_chars)
        wav_file = self.audio_processor.create_wav(waveform)
        t_stop = time.time()
        logging.debug("Sentence synthetize in {:.4f}s".format(t_stop - t_start)) 
        self.busy = False
        return wav_file

    def run(self):
        """ Run the flask server """
        self.app.run(host='0.0.0.0', port=self.service_port)

    def _check(self):
        """/check endpoint.
        Returns 200 - TTS <version> <model>. or 200 Service down.
        """
        try:
            with open('manifest.json', 'r') as f:
                version = json.load(f)['version']
        except:
            version = 'unknown'

        if self.ready:
            return "TTS v{} running {}".format(version, self.model_name) if self.ready else "Service down:{}".format(self.error)

    def _stop(self):
        """/stop endpoint.
        Stops the service.
        """
        self.ready = False
        logging.info("Server shutting down on request.")
        func = request.environ.get('werkzeug.server.shutdown')
        func()
        return "Service stopping"


def main(args):
    # Environment variable override arguments
    if 'SERVICE_PORT' in os.environ.keys():
        args.service_port = int(os.environ['SERVICE_PORT'])

    worker = TTS_Worker(args.config_path, args.model_path, args.service_port, args.cuda)
    worker.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help="Server config path (.json)")
    parser.add_argument('model_path', type=str, help="Model Path (.pth.tar)")
    parser.add_argument('--cuda', action="store_true", help="Use Cuda")
    parser.add_argument('--service_port', type=int, default=5000, help="Service port.")
    parser.add_argument('--debug', action="store_true", help="Prompt debug")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="TTS_Worker %(levelname)8s %(asctime)s %(message)s ")
    main(args)