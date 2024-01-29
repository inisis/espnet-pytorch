import time
import librosa
import torch
import numpy as np

from espnet_onnx.utils.config import get_config
from espnet_onnx.asr.frontend.frontend import Frontend
from espnet_onnx.asr.frontend.normalize.global_mvn import GlobalMVN
from espnet_onnx.asr.frontend.normalize.utterance_mvn import UtteranceMVN
from espnet_onnx.asr.scorer.ctc_prefix_scorer import CTCPrefixScorer
from espnet_onnx.asr.scorer.length_bonus import LengthBonus
from espnet_onnx.asr.model.lm import get_lm
from espnet_onnx.asr.beam_search.batch_beam_search import BatchBeamSearch
from espnet_onnx.asr.beam_search.beam_search import BeamSearch
from espnet_onnx.asr.postprocess.token_id_converter import TokenIDConverter
from espnet_onnx.asr.postprocess.build_tokenizer import build_tokenizer
from espnet_onnx.asr.model.decoders.xformer import XformerDecoder


import onnxruntime

y, sr = librosa.load('sample.wav', sr=16000)

config = get_config('asr_tmp/20240126_145803/config.yaml')
frontend = Frontend(config.encoder.frontend, ['CPUExecutionProvider'])
if config.encoder.do_normalize:
    if config.encoder.normalize.type == "gmvn":
        normalize = GlobalMVN(config.encoder.normalize)
    elif config.encoder.normalize.type == "utterance_mvn":
        normalize = UtteranceMVN(config.encoder.normalize)

encoder = onnxruntime.InferenceSession(config.encoder.model_path)
decoder = XformerDecoder(config.decoder, ['CPUExecutionProvider'])

scorers = {"decoder": decoder}
weights = {}
ctc = CTCPrefixScorer(
    config.ctc, config.token.eos, ['CPUExecutionProvider']
)
scorers.update(
    ctc=ctc, length_bonus=LengthBonus(len(config.token.list))
)
weights.update(
    decoder=config.weights.decoder,
    ctc=config.weights.ctc,
    length_bonus=config.weights.length_bonus,
)

lm = get_lm(config, ['CPUExecutionProvider'])
if lm is not None:
    scorers.update(lm=lm)
    weights.update(lm=config.weights.lm)

beam_search = BeamSearch(
    config.beam_search,
    config.token,
    scorers=scorers,
    weights=weights,
)
beam_search.__class__ = BatchBeamSearch

converter = TokenIDConverter(token_list=config.token.list)
tokenizer = build_tokenizer(**config.tokenizer)

speech = y
start_time = time.time()

speech = speech[np.newaxis, :]
# lengths: (1,)
speech_length = np.array([speech.shape[1]]).astype(np.int64)

feats, feat_length = frontend(speech, speech_length)

# 2. normalize with global MVN
if config.encoder.do_normalize:
    feats, feat_length = normalize(feats, feat_length)

# 3. forward encoder
encoder_out, encoder_out_lens = encoder.run(["encoder_out", "encoder_out_lens"], {"feats": feats})

nbest_hyps = beam_search(encoder_out[0])[:1]

results = []
for hyp in nbest_hyps:
    # remove sos/eos and get results
    token_int = list(hyp.yseq[1 : -1])

    # remove blank symbol id, which is assumed to be 0
    token_int = list([int(i) for i in filter(lambda x: x != 0, token_int)])

    # Change integer-ids to tokens
    token = converter.ids2tokens(token_int)

    text = tokenizer.tokens2text(token)

    results.append((text, token, token_int, hyp))

nbest = results

end_time = time.time()
running_time = end_time - start_time

print(f"Function took {running_time} seconds to run.")

print(len(nbest))
print(nbest[0][0])
