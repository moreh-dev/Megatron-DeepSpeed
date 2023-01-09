# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""T5 model."""

import torch
import functools

from deepspeed import comm as dist
from megatron import (
    get_args,
    mpu
)
from megatron.model.enums import AttnMaskType, LayerType
from megatron.model.language_model import parallel_lm_logits, get_language_model
from megatron.model.transformer import LayerNorm
from megatron.model.utils import (
    openai_gelu,
    get_linear_layer,
    init_method_normal,
    scaled_init_method_normal
)
from .module import MegatronModule, fp32_to_float16
from .language_model import EmbeddingPipe, EmbeddingPipeEncDec
from .transformer import ParallelTransformerLayerPipe, ParallelTransformerLayerPipeEnc, ParallelTransformerLayerPipeDec

from deepspeed.pipe import PipelineModule, LayerSpec, TiedLayerSpec

def t5_extended_attention_mask(attention_mask_list):

    def attn_mask_postprocess(attn_mask):
        # [b, 1, s, s]
        extended_attention_mask = attn_mask.unsqueeze(1)
        return extended_attention_mask

    return [attn_mask_postprocess(attn_mask) for attn_mask in attention_mask_list]


def t5_position_ids(token_ids):
    # Create position ids
    seq_length = token_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=token_ids.device)
    position_ids = position_ids.unsqueeze(0).expand_as(token_ids)

    return position_ids

class T5LMHeadPipe(MegatronModule):
    """Masked LM head for T5

    Arguments:
        mpu_vocab_size: model parallel size of vocabulary.
        hidden_size: hidden size
        init_method: init method for weight initialization
        layernorm_epsilon: tolerance for layer norm divisions
        parallel_output: wether output logits being distributed or not.
    """


    def __init__(self, mpu_vocab_size, parallel_output, word_embeddings_weight_func):
        super(T5LMHeadPipe, self).__init__()

        args = get_args()

        # create it's own weight and sync in every step
        self.weight_func = word_embeddings_weight_func
        self.bias = torch.nn.Parameter(torch.zeros(mpu_vocab_size))
        self.bias.model_parallel = True
        self.bias.partition_dim = 0
        self.bias.stride = 1
        self.parallel_output = parallel_output

    def forward(self, inputs):
        # hidden, enc_output, attention_mask, enc_dec_mask
        hidden_states, _, _, _ = inputs
        output = parallel_lm_logits(hidden_states,
                                    self.weight_func(),
                                    self.parallel_output,
                                    bias=self.bias)
        return output


class T5LMHead(MegatronModule):
    """Masked LM head for T5

    Arguments:
        mpu_vocab_size: model parallel size of vocabulary.
        hidden_size: hidden size
        init_method: init method for weight initialization
        layernorm_epsilon: tolerance for layer norm divisions
        parallel_output: wether output logits being distributed or not.
    """

    def __init__(self, mpu_vocab_size, parallel_output):
        super(T5LMHead, self).__init__()

        args = get_args()

        self.bias = torch.nn.Parameter(torch.zeros(mpu_vocab_size))
        self.bias.model_parallel = True
        self.bias.partition_dim = 0
        self.bias.stride = 1
        self.parallel_output = parallel_output

    def forward(self, hidden_states, word_embeddings_weight):
        output = parallel_lm_logits(hidden_states,
                                    word_embeddings_weight,
                                    self.parallel_output,
                                    bias=self.bias)
        return output


class T5Model(MegatronModule):
    """T5 Language model."""

    def __init__(self, num_tokentypes=0, parallel_output=True):
        super(T5Model, self).__init__()
        args = get_args()

        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        init_method = init_method_normal(args.init_method_std)
        scaled_init_method = scaled_init_method_normal(args.init_method_std,
                                                       args.num_layers)

        self.language_model, self._language_model_key = get_language_model(
            num_tokentypes=num_tokentypes,
            add_pooler=False,
            add_decoder=True,
            encoder_attn_mask_type=AttnMaskType.padding,
            init_method=init_method,
            scaled_init_method=scaled_init_method)

        self.lm_head = T5LMHead(
            self.language_model.embedding.word_embeddings.weight.size(0),
            parallel_output)
        self._lm_head_key = 'lm_head'

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.language_model.set_input_tensor(input_tensor)

    def forward(self, encoder_input_ids, decoder_input_ids, encoder_attn_mask,
                decoder_attn_mask, encoder_decoder_attn_mask,
                tokentype_ids=None, lm_labels=None, enc_hidden_states=None):

        # Converting the attention masks to proper parameter settings
        encoder_attn_mask, decoder_attn_mask, encoder_decoder_attn_mask = t5_extended_attention_mask(
            [encoder_attn_mask, decoder_attn_mask, encoder_decoder_attn_mask])

        encoder_position_ids = t5_position_ids(encoder_input_ids)
        decoder_position_ids = t5_position_ids(decoder_input_ids)

        lm_output = self.language_model(encoder_input_ids,
                                        encoder_position_ids,
                                        encoder_attn_mask,
                                        decoder_input_ids,
                                        decoder_position_ids,
                                        decoder_attn_mask,
                                        encoder_decoder_attn_mask,
                                        tokentype_ids=tokentype_ids,
                                        enc_hidden_states=enc_hidden_states,
                                        output_enc_hidden=False)

        decoder_output, encoder_output, *moe_losses = lm_output

        # Output.
        lm_logits = self.lm_head(decoder_output,
                                 self.language_model.embedding.word_embeddings.weight)

        if lm_labels is None:
            return lm_logits, encoder_output
        else:
            if self.fp16_lm_cross_entropy:
                assert lm_logits.dtype == torch.half
                lm_loss = mpu.vocab_parallel_cross_entropy(lm_logits, lm_labels)
            else:
                lm_loss = mpu.vocab_parallel_cross_entropy(lm_logits.float(),
                                                           lm_labels)
            return lm_loss, encoder_output

    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        """For easy load when model is combined with other heads,
        add an extra key."""

        state_dict_ = {}
        state_dict_[self._language_model_key] \
            = self.language_model.state_dict_for_save_checkpoint(
            destination, prefix, keep_vars)
        state_dict_[self._lm_head_key] \
            = self.lm_head.state_dict_for_save_checkpoint(
            destination, prefix, keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        self.language_model.load_state_dict(
            state_dict[self._language_model_key], strict=strict)
        self.lm_head.load_state_dict(state_dict[self._lm_head_key],
                                     strict=strict)

def CrossEntropy(output, labels):
    labels = labels[0]
    # TODO(jeesoo): loss mask?

    args = get_args()

    loss = mpu.vocab_parallel_cross_entropy(output.float(), labels)
    return loss.sum()

class DummyLanguageModel():
    def __init__(self):
        self.embedding = None

class T5ModelPipe(PipelineModule,MegatronModule):
    """T5 Language model."""

    def initialize_word_embeddings(self, init_method_normal):
        args = get_args()
        if not self.share_word_embeddings:
            raise Exception('initialize_word_embeddings() was called but '
                            'share_word_embeddings is false')

        # This function just initializes the word embeddings in the final stage
        # when we are using pipeline parallelism. If we aren't using pipeline
        # parallelism there is nothing to do.
        if args.pipeline_model_parallel_size == 1:
            return

        # Parameters are shared between the word embeddings layer, and the
        # heads at the end of the model. In a pipelined setup with more than
        # one stage, the initial embedding layer and the head are on different
        # workers, so we do the following:
        # 1. Create a second copy of word_embeddings on the last stage, with
        #    initial parameters of 0.0.
        # 2. Do an all-reduce between the first and last stage to ensure that
        #    the two copies of word_embeddings start off with the same
        #    parameter values.
        # 3. In the training loop, before an all-reduce between the grads of
        #    the two word_embeddings layers to ensure that every applied weight
        #    update is the same on both stages.
        if mpu.is_pipeline_last_stage():
            assert not mpu.is_pipeline_first_stage()
            self._word_embeddings_for_head_key = 'word_embeddings_for_head'
            # set word_embeddings weights to 0 here, then copy first
            # stage's weights using all_reduce below.
            self.word_embeddings = mpu.VocabParallelEmbedding(
                args.padded_vocab_size, args.hidden_size,
                init_method=init_method_normal(args.init_method_std))
            self.word_embeddings.weight.data.fill_(0)
            self.word_embeddings.weight.shared = True

        # Ensure that first and last stages have the same initial parameter
        # values.
        if torch.distributed.is_initialized():
            if mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage():
                torch.distributed.all_reduce(self.word_embeddings_weight(None).data,
                                             group=mpu.get_embedding_group())
        else:
            print("WARNING! Distributed processes aren't initialized, so "
                  "word embeddings in the last layer are not initialized. "
                  "If you are just manipulating a model this is fine, but "
                  "this needs to be handled manually. If you are training "
                  "something is definitely wrong.")

    def word_embeddings_weight(self, self2):
        if mpu.is_pipeline_first_stage(ignore_virtual=True):
            return self.tied_modules['embedding'].word_embeddings.weight
        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            if not self.share_word_embeddings:
                raise Exception('word_embeddings_weight() called for last '
                                'stage, but share_word_embeddings is false')
            return self.word_embeddings.weight
        raise Exception('word_embeddings_weight() should be '
                        'called for first and last stage only')

    def __init__(self, num_tokentypes=0, parallel_output=True):
        args = get_args()
        self.parallel_output = parallel_output

        init_method = init_method_normal(args.init_method_std)
        scaled_init_method = scaled_init_method_normal(args.init_method_std,
                                                       args.num_layers)

        self.specs = []

        self.embedding = TiedLayerSpec('embedding',
                                       EmbeddingPipeEncDec,
                                       args.hidden_size,
                                       args.padded_vocab_size,
                                       args.max_position_embeddings,
                                       args.hidden_dropout,
                                       init_method=init_method,
                                       num_tokentypes=num_tokentypes,
                                       tied_weight_attr='word_embeddings_weight')
        '''
        def _to_float16(inputs):
            if args.fp16:
                return fp32_to_float16(inputs, lambda v: v.half())
            elif args.bf16:
                return fp32_to_float16(inputs, lambda v: v.bfloat16())
            else:
                return inputs
        self.specs.append(_to_float16)
        '''

        # encoder-decoder embedding layer
        self.specs.append(self.embedding)
        self.share_word_embeddings = True

        def transpose_hidden(x_i, i, fp32_residual_connection):
            if i == 0:
                if fp32_residual_connection:
                    return x_i.transpose(0, 1).contiguous().float()
                else:
                    return x_i.transpose(0, 1).contiguous()
            else:
                return x_i

        # encoder
        for layer_idx in range(args.num_layers):
            self.specs.append(
                LayerSpec(ParallelTransformerLayerPipeEnc,
                    init_method=init_method,
                    output_layer_init_method=scaled_init_method,
                    layer_number=layer_idx,
                    self_attn_mask_type=AttnMaskType.padding))

        '''
        encoder_output, enc_mask, dec_embeddings, dec_mask, enc_dec_mask
        ---->
        dec_embeddings, encoder_output, dec_mask, enc_dec_mask
        '''
        self.specs.append(lambda x: (x[2], x[0], x[3], x[4]))

        # decoder
        for layer_idx in range(args.num_layers):
            self.specs.append(
                LayerSpec(ParallelTransformerLayerPipeDec,
                    init_method=init_method,
                    output_layer_init_method=scaled_init_method,
                    layer_number=layer_idx,
                    layer_type=LayerType.decoder,
                    self_attn_mask_type=AttnMaskType.padding))

        self.specs.append(lambda x: [transpose_hidden(x_i, i, args.fp32_residual_connection) for i, x_i in enumerate(x)])

        # lm head
        self.specs.append(
                LayerSpec(T5LMHeadPipe,
                          args.padded_vocab_size,
                          parallel_output,
                          functools.partial(self.word_embeddings_weight, self)))

        from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology
        topo = PipeModelDataParallelTopology(num_pp=mpu.get_pipeline_model_parallel_world_size(),
                                             num_mp=mpu.get_tensor_model_parallel_world_size(),
                                             num_dp=mpu.get_data_parallel_world_size())

        super().__init__(layers=self.specs,
                         loss_fn=CrossEntropy,
                         topology=topo,
                         activation_checkpoint_interval=0,
                         partition_method='type:transformer')
