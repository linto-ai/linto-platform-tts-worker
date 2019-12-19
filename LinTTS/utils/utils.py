import re
import json

from models.tacotron import Tacotron

class ConfigDict(dict):
    def __init__(self, *args, **kwargs):
        super(ConfigDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def load_config(config_path) -> ConfigDict:
    config = ConfigDict()
    with open(config_path, "r") as f:
        input_str = f.read()
    input_str = re.sub(r'\\\n', '', input_str)
    input_str = re.sub(r'//.*\n', '\n', input_str)
    data = json.loads(input_str)
    config.update(data)
    return config

def setup_model(num_chars, c):
    model = Tacotron(num_chars=num_chars,
                        num_speakers=0,
                        r=c.r,
                        linear_dim=1025,
                        mel_dim=80,
                        gst=c.use_gst,
                        memory_size=c.memory_size,
                        attn_win=c.windowing,
                        attn_norm=c.attention_norm,
                        prenet_type=c.prenet_type,
                        prenet_dropout=c.prenet_dropout,
                        forward_attn=c.use_forward_attn,
                        trans_agent=c.transition_agent,
                        forward_attn_mask=c.forward_attn_mask,
                        location_attn=c.location_attn,
                        separate_stopnet=c.separate_stopnet)
    return model