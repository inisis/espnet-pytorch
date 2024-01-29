import time
import librosa
import torch
import numpy as np

from espnet_onnx.utils.config import Config
from espnet_onnx.utils.config import get_config
from espnet_onnx.asr.frontend.frontend import Frontend
from espnet_onnx.asr.frontend.normalize.global_mvn import GlobalMVN
from espnet_onnx.asr.frontend.normalize.utterance_mvn import UtteranceMVN
from espnet_onnx.asr.scorer.length_bonus import LengthBonus
from espnet_onnx.asr.beam_search.batch_beam_search import BatchBeamSearch
from espnet_onnx.asr.beam_search.beam_search import BeamSearch
from espnet_onnx.asr.postprocess.token_id_converter import TokenIDConverter
from espnet_onnx.asr.postprocess.build_tokenizer import build_tokenizer
from espnet_onnx.asr.scorer.interface import BatchScorerInterface, BatchPartialScorerInterface

import six
from typing import Any, List, Tuple
from scipy.special import log_softmax, logsumexp


class XformerDecoder(BatchScorerInterface):
    def __init__(
        self,
        config: Config,
    ):
        """Onnx support for espnet2.asr.decoder.transformer_decoder

        Args:
            config (Config):
            use_quantized (bool): Flag to use quantized model
        """
        self.decoder = torch.load('asr_tmp/20240126_145803/full/xformer_decoder.gm')
        self.config = config
        self.n_layers = config.n_layers
        self.odim = config.odim

    def batch_score(
        self, ys: np.ndarray, states: List[Any], xs: np.ndarray
    ) -> Tuple[np.ndarray, List[Any]]:
        if len(ys.shape) == 1:
            ys = ys[None, :]

        n_batch = len(ys)
        if states[0] is None:
            batch_state = [
                np.zeros((1, 1, self.odim), dtype=np.float32)
                for _ in range(self.n_layers)
            ]
        else:
            # transpose state of [batch, layer] into [layer, batch]
            batch_state = [
                np.concatenate([states[b][i][None, :] for b in range(n_batch)])
                for i in range(self.n_layers)
            ]

        # batch decoding
        ys = torch.from_numpy(ys).to(torch.int64)
        xs = torch.from_numpy(xs)
        batch_state = [torch.from_numpy(s) for s in batch_state]
        logp, *states = self.decoder(ys, xs, *batch_state)
        logp = logp.detach().numpy()
        states = [s.detach().numpy() for s in states]
        
        if type(self.n_layers) == 1:
            states = [states]

        # transpose state of [layer, batch] into [batch, layer]
        state_list = [
            [states[i][b] for i in range(self.n_layers)] for b in range(n_batch)
        ]

        return logp, state_list

class CTCPrefixScorer(BatchPartialScorerInterface):
    """Decoder interface wrapper for CTCPrefixScore."""

    def __init__(
        self, ctc: Config, eos: int, providers: List[str], use_quantized: bool = False
    ):
        """Initialize class.
        Args:
            ctc (np.ndarray): The CTC implementation.
                For example, :class:`espnet.nets.pytorch_backend.ctc.CTC`
            eos (int): The end-of-sequence id.
        """
        self.ctc = torch.load('asr_tmp/20240126_145803/full/ctc.gm')
        self.eos = eos
        self.impl = None

    def init_state(self, x: np.ndarray):
        """Get an initial state for decoding.
        Args:
            x (np.ndarray): The encoded feature tensor
        Returns: initial state
        """
        x = self.ctc.run(["ctc_out"], {"x": x[None, :]})[0]
        logp = np.squeeze(x, axis=0)
        # TODO(karita): use CTCPrefixScoreTH
        self.impl = CTCPrefixScore(logp, 0, self.eos, np)
        return 0, self.impl.initial_state()

    def select_state(self, state, i, new_id=None):
        """Select state with relative ids in the main beam search.
        Args:
            state: Decoder state for prefix tokens
            i (int): Index to select a state in the main beam search
            new_id (int): New label id to select a state if necessary
        Returns:
            state: pruned state
        """
        if type(state) == tuple:
            if len(state) == 2:  # for CTCPrefixScore
                sc, st = state
                return sc[i], st[i]
            else:  # for CTCPrefixScoreTH (need new_id > 0)
                r, log_psi, f_min, f_max, scoring_idmap = state
                s = log_psi[i, new_id].repeat(log_psi.shape[1])
                if scoring_idmap is not None:
                    return r[:, :, i, scoring_idmap[i, new_id]], s, f_min, f_max
                else:
                    return r[:, :, i, new_id], s, f_min, f_max
        return None if state is None else state[i]

    def score_partial(self, y, ids, state, x):
        """Score new token.
        Args:
            y (np.ndarray): 1D prefix token
            next_tokens (np.ndarray): torch.int64 next token to score
            state: decoder state for prefix tokens
            x (np.ndarray): 2D encoder feature that generates ys
        Returns:
            tuple[np.ndarray, Any]:
                Tuple of a score tensor for y that has a shape `(len(next_tokens),)`
                and next state for ys
        """
        prev_score, state = state
        presub_score, new_st = self.impl(y, ids, state)
        tscore = np.array(presub_score - prev_score, dtype=x.dtype)
        return tscore, (presub_score, new_st)

    def batch_init_state(self, x: np.ndarray):
        """Get an initial state for decoding.
        Args:
            x (np.ndarray): The encoded feature tensor
        Returns: initial state
        """
        x = torch.from_numpy(x).unsqueeze(0)
        logp = self.ctc(x)
        logp = logp.detach().numpy()
        xlen = np.array([logp.shape[1]])
        self.impl = CTCPrefixScoreTH(logp, xlen, 0, self.eos)
        return None

    def batch_score_partial(self, y, ids, state, x):
        """Score new token.
        Args:
            y (np.ndarray): 1D prefix token
            ids (np.ndarray): torch.int64 next token to score
            state: decoder state for prefix tokens
            x (np.ndarray): 2D encoder feature that generates ys
        Returns:
            tuple[np.ndarray, Any]:
                Tuple of a score tensor for y that has a shape `(len(next_tokens),)`
                and next state for ys
        """
        if state[0] is not None:
            batch_state = (
                np.concatenate([s[0][..., None] for s in state], axis=2),
                np.concatenate([s[1][None, :] for s in state]),
                state[0][2],
                state[0][3],
            )
        else:
            batch_state = None
        return self.impl(y, batch_state, ids)

    def extend_prob(self, x: np.ndarray):
        """Extend probs for decoding.

        This extension is for streaming decoding
        as in Eq (14) in https://arxiv.org/abs/2006.14941

        Args:
            x (np.ndarray): The encoded feature tensor

        """
        x = self.ctc.run(["ctc_out"], {"x": x[None, :]})[0]
        logp = log_softmax(x, axis=-1)
        self.impl.extend_prob(logp)

    def extend_state(self, state):
        """Extend state for decoding.

        This extension is for streaming decoding
        as in Eq (14) in https://arxiv.org/abs/2006.14941

        Args:
            state: The states of hyps

        Returns: exteded state

        """
        new_state = []
        for s in state:
            new_state.append(self.impl.extend_state(s))

        return new_state

class CTCPrefixScoreTH:
    def __init__(
        self, x: np.ndarray, xlens: np.ndarray, blank: int, eos: int, margin: int = 0
    ):
        """Construct CTC prefix scorer
        :param np.ndarray x: input label posterior sequences (B, T, O)
        :param np.ndarray xlens: input lengths (B,)
        :param int blank: blank label id
        :param int eos: end-of-sequence id
        :param int margin: margin parameter for windowing (0 means no windowing)
        """
        # In the comment lines,
        # we assume T: input_length, B: batch size, W: beam width, O: output dim.
        self.logzero = -10000000000.0
        self.blank = blank
        self.eos = eos
        self.batch = x.shape[0]
        self.input_length = x.shape[1]
        self.odim = x.shape[2]
        self.dtype = x.dtype

        # Pad the rest of posteriors in the batch
        # TODO(takaaki-hori): need a better way without for-loops
        for i, l in enumerate(xlens):
            if l < self.input_length:
                x[i, l:, :] = self.logzero
                x[i, l:, blank] = 0
        # Reshape input x
        xn = x.transpose(1, 0, 2)  # (B, T, O) -> (T, B, O)
        xb = xn[:, :, None, self.blank].repeat(self.odim, axis=2)
        # operation is faster than np.stack
        self.x = np.concatenate([xn[None, :], xb[None, :]])
        self.end_frames = xlens - 1

        # Setup CTC windowing
        self.margin = margin
        if margin > 0:
            self.frame_ids = np.arange(self.input_length, dtype=self.dtype)
        # Base indices for index conversion
        self.idx_bh = None
        self.idx_b = np.arange(self.batch)
        self.idx_bo = (self.idx_b * self.odim)[:, None]

    def __call__(self, y, state, scoring_ids=None, att_w=None):
        """Compute CTC prefix scores for next labels
        :param list y: prefix label sequences
        :param tuple state: previous CTC state
        :param np.ndarray pre_scores: scores for pre-selection of hypotheses (BW, O)
        :param np.ndarray att_w: attention weights to decide CTC window
        :return new_state, ctc_local_scores (BW, O)
        """
        output_length = len(y[0]) - 1  # ignore blank and sos
        last_ids = [yi[-1] for yi in y]  # last output label ids
        n_bh = len(last_ids)  # batch * hyps
        n_hyps = n_bh // self.batch  # assuming each utterance has the same # of hyps
        self.scoring_num = scoring_ids.shape[-1] if scoring_ids is not None else 0
        # prepare state info
        if state is None:
            r_prev = np.full(
                (self.input_length, 2, self.batch, n_hyps),
                self.logzero,
                dtype=self.dtype,
            )
            r_prev[:, 1] = np.cumsum(self.x[0, :, :, self.blank], 0)[:, :, None]
            r_prev = r_prev.reshape(-1, 2, n_bh)
            s_prev = 0.0
            f_min_prev = 0
            f_max_prev = 1
        else:
            r_prev, s_prev, f_min_prev, f_max_prev = state

        # select input dimensions for scoring
        if self.scoring_num > 0:
            scoring_idmap = np.full((n_bh, self.odim), -1, dtype=np.int64)
            snum = self.scoring_num
            if self.idx_bh is None or n_bh > len(self.idx_bh):
                self.idx_bh = np.arange(n_bh).reshape(-1, 1)
            scoring_idmap[self.idx_bh[:n_bh], scoring_ids] = np.arange(snum)
            scoring_idx = (
                scoring_ids + self.idx_bo.repeat(n_hyps, axis=1).reshape(-1, 1)
            ).reshape(-1)
            x_ = np.take(
                self.x.reshape(2, -1, self.batch * self.odim), scoring_idx, axis=2
            ).reshape(2, -1, n_bh, snum)
        else:
            scoring_ids = None
            scoring_idmap = None
            snum = self.odim
            x_ = self.x[:, :, :, None].repeat(n_hyps, axis=3).reshape(2, -1, n_bh, snum)

        # new CTC forward probs are prepared as a (T x 2 x BW x S) tensor
        # that corresponds to r_t^n(h) and r_t^b(h) in a batch.
        r = np.full((self.input_length, 2, n_bh, snum), self.logzero, dtype=self.dtype)
        if output_length == 0:
            r[0, 0] = x_[0, 0]

        r_sum = logsumexp(r_prev, axis=1)
        log_phi = r_sum[:, :, None].repeat(snum, 2)
        if scoring_ids is not None:
            for idx in range(n_bh):
                pos = scoring_idmap[idx, int(last_ids[idx])]
                if pos >= 0:
                    log_phi[:, idx, pos] = r_prev[:, 1, idx]
        else:
            for idx in range(n_bh):
                log_phi[:, idx, last_ids[idx]] = r_prev[:, 1, idx]

        # decide start and end frames based on attention weights
        if att_w is not None and self.margin > 0:
            f_arg = np.matmul(att_w, self.frame_ids)
            f_min = max(int(f_arg.min()), f_min_prev)
            f_max = max(int(f_arg.max()), f_max_prev)
            start = min(f_max_prev, max(f_min - self.margin, output_length, 1))
            end = min(f_max + self.margin, self.input_length)
        else:
            f_min = f_max = 0
            start = max(output_length, 1)
            end = self.input_length

        # compute forward probabilities log(r_t^n(h)) and log(r_t^b(h))
        for t in range(start, end):
            rp = r[t - 1]
            rr = np.concatenate(
                [rp[0:1], log_phi[t - 1 : t], rp[0:1], rp[1:2]]
            ).reshape(2, 2, n_bh, snum)
            r[t] = logsumexp(rr, 1) + x_[:, t]

        # compute log prefix probabilities log(psi)
        log_phi_x = np.concatenate((log_phi[0][None, :], log_phi[:-1]), axis=0) + x_[0]
        if scoring_ids is not None:
            log_psi = np.full((n_bh, self.odim), self.logzero, dtype=self.dtype)
            log_psi_ = logsumexp(
                np.concatenate(
                    (log_phi_x[start:end], r[start - 1, 0][None, :]), axis=0
                ),
                axis=0,
            )
            for si in range(n_bh):
                log_psi[si, scoring_ids[si].astype(np.int64)] = log_psi_[si]
        else:
            log_psi = logsumexp(
                np.concatenate(
                    (log_phi_x[start:end], r[start - 1, 0][None, :]), axis=0
                ),
                axis=0,
            )

        for si in range(n_bh):
            log_psi[si, self.eos] = r_sum[self.end_frames[si // n_hyps], si]

        # exclude blank probs
        log_psi[:, self.blank] = self.logzero

        return (log_psi - s_prev), (r, log_psi, f_min, f_max, scoring_idmap)

    def index_select_state(self, state, best_ids):
        """Select CTC states according to best ids
        :param state    : CTC state
        :param best_ids : index numbers selected by beam pruning (B, W)
        :return selected_state
        """
        r, s, f_min, f_max, scoring_idmap = state
        # convert ids to BHO space
        n_bh = len(s)
        n_hyps = n_bh // self.batch
        vidx = (best_ids + (self.idx_b * (n_hyps * self.odim)).reshape(-1, 1)).reshape(
            -1
        )
        # select hypothesis scores
        s_new = np.take(s.reshape(-1), vidx, axis=0)
        s_new = s_new.reshape(-1, 1).repeat(self.odim, axis=1).reshape(n_bh, self.odim)
        # convert ids to BHS space (S: scoring_num)
        if scoring_idmap is not None:
            snum = self.scoring_num
            hyp_idx = (
                best_ids // self.odim + (self.idx_b * n_hyps).reshape(-1, 1)
            ).reshape(-1)
            label_ids = np.fmod(best_ids, self.odim).reshape(-1)
            score_idx = scoring_idmap[hyp_idx, label_ids]
            score_idx[score_idx == -1] = 0
            vidx = score_idx + hyp_idx * snum
        else:
            snum = self.odim
        # select forward probabilities
        r_new = np.take(r.reshape(-1, 2, n_bh * snum), vidx, axis=2).reshape(
            -1, 2, n_bh
        )
        return r_new, s_new, f_min, f_max

    def extend_prob(self, x):
        """Extend CTC prob.
        :param np.ndarray x: input label posterior sequences (B, T, O)
        """
        if self.x.shape[1] < x.shape[1]:  # self.x (2,T,B,O); x (B,T,O)
            # Pad the rest of posteriors in the batch
            # TODO(takaaki-hori): need a better way without for-loops
            xlens = np.array([x.shape[1]])
            for i, l in enumerate(xlens):
                if l < self.input_length:
                    x[i, l:, :] = self.logzero
                    x[i, l:, self.blank] = 0
            tmp_x = self.x
            xn = x.transpose(1, 0, 2)  # (B, T, O) -> (T, B, O)
            xb = xn[:, :, None, self.blank].repeat(self.odim, axis=2)
            self.x = np.concatenate([xn[None, :], xb[None, :]])  # (2, T, B, O)
            self.x[:, : tmp_x.shape[1], :, :] = tmp_x
            self.input_length = x.shape[1]
            self.end_frames = xlens - 1

    def extend_state(self, state):
        """Compute CTC prefix state.
        :param state    : CTC state
        :return ctc_state
        """

        if state is None:
            # nothing to do
            return state
        else:
            r_prev, s_prev, f_min_prev, f_max_prev = state

            r_prev_new = np.full(
                (self.input_length, 2),
                self.logzero,
                dtype=self.dtype,
            )
            start = max(r_prev.shape[0], 1)
            r_prev_new[0:start] = r_prev
            for t in six.moves.range(start, self.input_length):
                r_prev_new[t, 1] = r_prev_new[t - 1, 1] + self.x[0, t, :, self.blank]

            return (r_prev_new, s_prev, f_min_prev, f_max_prev)

class TransformerLM(BatchScorerInterface):
    def __init__(
        self,
        config,
        providers: List[str],
        use_quantized: bool = False,
    ):
        self.lm_session = torch.load('asr_tmp/20240126_145803/full/transformer_lm.gm')
        self.nlayers = config.nlayers
        self.odim = config.odim

    def score(self, y: np.ndarray, state: Any, x: np.ndarray) -> Tuple[np.ndarray, Any]:
        """Score new token.

        Args:
            y (np.ndarray): 1D torch.int64 prefix tokens.
            state: Scorer state for prefix tokens
            x (np.ndarray): encoder feature that generates ys.

        Returns:
            tuple[np.ndarray, Any]: Tuple of
                torch.float32 scores for next token (vocab_size)
                and next state for ys

        """
        y = y[None, :]
        input_dic = {"tgt": y}

        if state is None:
            state = [
                np.zeros((1, 1, self.odim), dtype=np.float32)
                for _ in range(self.nlayers)
            ]

        input_dic.update({k: v for k, v in zip(self.enc_in_cache_names, state)})
        decoded, *new_state = self.lm_session.run(self.enc_output_names, input_dic)

        if self.nlayers == 1:
            new_state = [new_state]

        logp = log_softmax(decoded, axis=-1).squeeze(0)
        return logp, new_state

    def batch_score(
        self, ys: np.ndarray, states: List[Any], xs: np.ndarray
    ) -> Tuple[np.ndarray, List[Any]]:
        """Score new token batch.

        Args:
            ys (np.ndarray): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (np.ndarray):
                The encoder feature that generates ys (n_batch, xlen, n_feat).

        Returns:
            tuple[np.ndarray, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, vocab_size)`
                and next state list for ys.

        """
        # merge states
        ys = ys.astype(np.int64)
        n_batch = len(ys)
        is_first_iteration = False
        if states[0] is None:
            batch_state = [
                np.zeros((1, 1, self.odim), dtype=np.float32)
                for _ in range(self.nlayers)
            ]
            is_first_iteration = True
        else:
            # transpose state of [batch, layer] into [layer, batch]
            batch_state = [
                np.concatenate([states[b][i][None, :] for b in range(n_batch)])
                for i in range(self.nlayers)
            ]

        ys = torch.from_numpy(ys)
        batch_state = [torch.from_numpy(s) for s in batch_state]
        decoded, *new_state = self.lm_session(ys, *batch_state)
        decoded = decoded.detach().numpy()
        new_state = [s.detach().numpy() for s in new_state]

        logp = log_softmax(decoded, axis=-1)

        # if first iteration, remove the first row
        if is_first_iteration:
            new_state = [new_state[i][:, -1:] for i in range(len(new_state))]

        # transpose state of [layer, batch] into [batch, layer]
        state_list = [
            [new_state[i][b] for i in range(self.nlayers)] for b in range(n_batch)
        ]
        return logp, state_list


y, sr = librosa.load('sample2.wav', sr=16000)

config = get_config('asr_tmp/20240126_145803/config.yaml')
frontend = Frontend(config.encoder.frontend, ['CPUExecutionProvider'])
if config.encoder.do_normalize:
    if config.encoder.normalize.type == "gmvn":
        normalize = GlobalMVN(config.encoder.normalize)
    elif config.encoder.normalize.type == "utterance_mvn":
        normalize = UtteranceMVN(config.encoder.normalize)

encoder = torch.load('asr_tmp/20240126_145803/full/default_encoder.gm')
decoder = XformerDecoder(config.decoder)

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

lm = TransformerLM(config.lm, ['CPUExecutionProvider'])
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
encoder_out, encoder_out_lens = encoder(torch.from_numpy(feats))
encoder_out = encoder_out.detach().numpy()

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
