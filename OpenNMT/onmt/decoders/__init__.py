"""Module defining decoders."""
from onmt.decoders.decoder import DecoderBase, RNNFusionDecoderBase, InputFeedRNNDecoder, \
    StdRNNDecoder, DynamicFusionDecoder, ColdFusionDecoder, RNNDynamicFusionDecoderBase # necessary to import the Fusion Decoder
from onmt.decoders.transformer import TransformerDecoder
from onmt.decoders.cnn_decoder import CNNDecoder


str2dec = {"rnn": StdRNNDecoder, "ifrnn": InputFeedRNNDecoder,
           "cnn": CNNDecoder, "transformer": TransformerDecoder, "dynamic_fusion": DynamicFusionDecoder, "cold_fusion": ColdFusionDecoder}

__all__ = ["DecoderBase", "TransformerDecoder", "StdRNNDecoder", "CNNDecoder",
           "InputFeedRNNDecoder", "str2dec", "DynamicFusionDecoder", "RNNFusionDecoderBase", "RNNDynamicFusionDecoderBase"]
