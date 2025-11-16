from functools import partial
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from att import att_full, att_sink, att_dsa_static, att_dsa_dynamic

def register_moba():
    ALL_ATTENTION_FUNCTIONS["att_full"] = partial(att_full)
    ALL_ATTENTION_FUNCTIONS["att_sink"] = partial(att_sink)
    ALL_ATTENTION_FUNCTIONS["att_dsa_static"] = partial(att_dsa_static)
    ALL_ATTENTION_FUNCTIONS["att_dsa_dynamic"] = partial(att_dsa_dynamic)