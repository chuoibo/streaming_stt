import torch
import logging
import numpy as np

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


class Wav2vec2Inference:
    def __init__(self):
        self.model_name = ''
        self.model_cache = ''


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = Wav2Vec2ForCTC.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            cache_dir=self.model_cache
        ).to(self.device)

        self.processor = Wav2Vec2Processor.from_pretrained(
            pretrained_model_name_or_path=self.model_name, 
            cache_dir=self.model_cache)

        logging.info('Loading pretrained processor ...')


    def speech_recognition(self, audio_buffer):
        if len(audio_buffer) == 0:
            return ""

        inputs = self.processor(torch.tensor(audio_buffer), 
                                sampling_rate=16000, 
                                return_tensors='pt', 
                                padding=True).input_values

        with torch.no_grad():
            logits = self.model(inputs.to(self.device)).logits            

        predicted_ids = torch.argmax(logits, dim=-1)
        generated_text = self.processor.batch_decode(predicted_ids)[0]

        return generated_text