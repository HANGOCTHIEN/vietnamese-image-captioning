"""Kiểm thử LSTM E0 bằng tensor nhỏ, không training hoặc tải pretrained weights."""

from pathlib import Path

import pytest
import torch
from torch import nn

from image_captioning.models.lstm_decoder import LSTMDecoder


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    """Giới hạn threads cho phép toán nhỏ, khôi phục cấu hình khi xong module."""
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(autouse=True)
def random_state():
    """Seed do test quản lý, không yêu cầu model tự thay đổi RNG toàn cục."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        yield


@pytest.fixture
def decoder() -> LSTMDecoder:
    """Cấu hình CPU nhỏ theo hợp đồng encoder, vocabulary giả gồm 100 ID."""
    return LSTMDecoder(
        vocab_size=100, encoder_dim=2048, embedding_dim=32,
        hidden_dim=64, num_layers=1, dropout=.1, pad_token_id=0,
    )


def test_import_constructor_and_defaults(decoder: LSTMDecoder) -> None:
    assert isinstance(decoder, nn.Module)
    assert decoder.embedding.num_embeddings == 100
    assert decoder.lstm.batch_first
    assert not decoder.lstm.bidirectional
    assert decoder.lstm.dropout == 0
    assert decoder.embedding_dropout.p == decoder.output_dropout.p == .1
    defaults = LSTMDecoder(vocab_size=17)
    assert (defaults.encoder_dim, defaults.embedding_dim, defaults.hidden_dim, defaults.num_layers) == (2048, 256, 512, 1)
    assert defaults.embedding_dropout.p == defaults.output_dropout.p == .3


@pytest.mark.parametrize("batch_size,time", [(2, 10), (1, 6), (1, 1)])
def test_forward_shapes_and_finite_logits(decoder: LSTMDecoder, batch_size: int, time: int) -> None:
    logits = decoder(torch.randn(batch_size, 2048), torch.randint(0, 100, (batch_size, time)))
    assert logits.shape == (batch_size, time, 100)
    assert logits.device.type == "cpu"
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("num_layers,batch_size", [(1, 1), (2, 1), (3, 2)])
def test_state_and_multilayer_shapes(num_layers: int, batch_size: int) -> None:
    model = LSTMDecoder(19, encoder_dim=7, embedding_dim=8, hidden_dim=11, num_layers=num_layers)
    features = torch.randn(batch_size, 7)
    hidden, cell = model.init_state(features)
    assert hidden.shape == cell.shape == (num_layers, batch_size, 11)
    assert hidden.is_contiguous() and cell.is_contiguous()
    assert hidden.abs().max() <= 1 and cell.abs().max() <= 1
    assert model(features, torch.ones(batch_size, 4, dtype=torch.long)).shape == (batch_size, 4, 19)
    assert model.lstm.dropout == (.3 if num_layers > 1 else 0)


def test_image_conditioning_uses_independent_hidden_and_cell() -> None:
    model = LSTMDecoder(9, encoder_dim=4, embedding_dim=3, hidden_dim=5, dropout=0).eval()
    assert model.image_to_hidden.weight.data_ptr() != model.image_to_cell.weight.data_ptr()
    with torch.no_grad():
        model.image_to_hidden.weight.zero_()
        model.image_to_hidden.bias.zero_()
        model.image_to_hidden.weight[:, 0] = .5
        model.image_to_cell.weight.zero_()
        model.image_to_cell.bias.zero_()
        model.image_to_cell.weight[:, 1] = .25
        model.embedding.weight.fill_(.1)
        model.embedding.weight[0].zero_()
        for name, parameter in model.lstm.named_parameters():
            parameter.fill_(.1 if "weight" in name else 0)
        model.output_projection.weight.fill_(.25)
        model.output_projection.bias.zero_()
    features = torch.tensor([[0., 0., 0., 0.], [1., 0., 0., 0.], [0., 1., 0., 0.]])
    hidden, cell = model.init_state(features)
    assert hidden[:, 0].eq(0).all() and cell[:, 0].eq(0).all()
    assert hidden[:, 1].gt(0).all() and cell[:, 1].eq(0).all()
    assert hidden[:, 2].eq(0).all() and cell[:, 2].gt(0).all()
    logits = model(features, torch.ones(3, 2, dtype=torch.long))
    # Các weights dương cố định làm thay đổi đầu ra theo từng loại initial state.
    assert logits[1, 0].gt(logits[0, 0]).all()
    assert logits[2, 0].gt(logits[0, 0]).all()


@pytest.mark.parametrize("features_trainable", [False, True])
def test_gradient_flow(decoder: LSTMDecoder, features_trainable: bool) -> None:
    features = torch.randn(2, 2048, requires_grad=features_trainable)
    logits = decoder(features, torch.tensor([[1, 4, 5, 0], [1, 6, 2, 0]]))
    logits.square().mean().backward()
    for name, parameter in decoder.named_parameters():
        assert parameter.requires_grad, name
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    if features_trainable:
        assert features.grad is not None
        assert torch.isfinite(features.grad).all() and features.grad.abs().sum() > 0
    else:
        assert features.grad is None


def test_padding_embedding_and_unmodified_raw_logits() -> None:
    model = LSTMDecoder(13, encoder_dim=5, embedding_dim=4, hidden_dim=6, pad_token_id=7, dropout=0)
    assert model.embedding.padding_idx == 7
    assert model.embedding.weight[7].eq(0).all()
    model(torch.randn(2, 5), torch.tensor([[1, 7, 2], [1, 3, 7]])).square().mean().backward()
    assert model.embedding.weight.grad[7].eq(0).all()
    assert model.embedding.weight.grad[1].abs().sum() > 0
    with torch.no_grad():
        model.output_projection.weight.zero_()
        model.output_projection.bias.copy_(torch.linspace(-3, 3, 13))
    logits = model(torch.randn(2, 5), torch.full((2, 3), 7, dtype=torch.long))
    expected = torch.linspace(-3, 3, 13).expand(2, 3, 13)
    torch.testing.assert_close(logits, expected)
    assert not logits.eq(0).all()
    assert not any(isinstance(module, (nn.Softmax, nn.LogSoftmax)) for module in model.modules())


def test_dropout_modes_and_no_persistent_state(decoder: LSTMDecoder) -> None:
    features, tokens = torch.randn(2, 2048), torch.ones(2, 5, dtype=torch.long)
    decoder.eval()
    assert not decoder.embedding_dropout.training and not decoder.output_dropout.training
    first = decoder(features, tokens)
    decoder.step(tokens[:, 0], decoder.init_state(features))
    torch.testing.assert_close(decoder(features, tokens), first)
    assert not hasattr(decoder, "state")
    decoder.train()
    assert decoder.embedding_dropout.training and decoder.output_dropout.training


@pytest.mark.parametrize("layers,batch_size", [(1, 1), (2, 2)])
def test_step_and_forward_equivalence(layers: int, batch_size: int) -> None:
    model = LSTMDecoder(23, encoder_dim=9, embedding_dim=7, hidden_dim=11,
                        num_layers=layers, dropout=.4).eval()
    features = torch.randn(batch_size, 9)
    tokens = torch.tensor([[1, 4, 5, 2, 0], [1, 6, 2, 0, 0]])[:batch_size]
    with torch.no_grad():
        expected = model(features, tokens)
        state = model.init_state(features)
        results = []
        for current in tokens.unbind(1):
            logits, state = model.step(current, state)
            assert logits.shape == (batch_size, 23)
            assert state[0].shape == state[1].shape == (layers, batch_size, 11)
            results.append(logits)
    torch.testing.assert_close(torch.stack(results, dim=1), expected, rtol=1e-5, atol=1e-6)


def test_step_keeps_gradients(decoder: LSTMDecoder) -> None:
    features = torch.randn(2, 2048, requires_grad=True)
    logits, state = decoder.step(torch.tensor([1, 2]), decoder.init_state(features))
    logits.square().mean().backward()
    assert features.grad is not None and features.grad.abs().sum() > 0
    assert state[0].grad_fn is not None and state[1].grad_fn is not None
    assert decoder.image_to_hidden.weight.grad.abs().sum() > 0
    assert decoder.image_to_cell.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("argument,value", [
    ("vocab_size", 0), ("vocab_size", -1), ("vocab_size", 3.5), ("vocab_size", True),
    ("encoder_dim", 0), ("encoder_dim", -2), ("encoder_dim", 3.5),
    ("embedding_dim", 0), ("embedding_dim", -1), ("embedding_dim", True),
    ("hidden_dim", 0), ("hidden_dim", -1), ("hidden_dim", 4.5),
    ("num_layers", 0), ("num_layers", -1), ("num_layers", True),
    ("dropout", -.1), ("dropout", 1), ("dropout", float("nan")),
    ("dropout", float("inf")), ("dropout", "0.3"), ("dropout", True),
    ("pad_token_id", -1), ("pad_token_id", 10), ("pad_token_id", 1.5), ("pad_token_id", False),
])
def test_invalid_constructor(argument: str, value: object) -> None:
    config = {"vocab_size": 10, argument: value}
    with pytest.raises((TypeError, ValueError), match=argument):
        LSTMDecoder(**config)


@pytest.mark.parametrize("kind,error", [
    ("not_tensor", TypeError), ("spatial", ValueError), ("width", ValueError),
    ("rank_one", ValueError), ("empty_batch", ValueError), ("integer", TypeError),
    ("double", TypeError), ("device", ValueError),
])
def test_invalid_global_features(decoder: LSTMDecoder, kind: str, error: type[Exception]) -> None:
    features = {
        "not_tensor": [[0.]], "spatial": torch.zeros(2, 49, 2048),
        "width": torch.zeros(2, 7), "rank_one": torch.zeros(2048),
        "empty_batch": torch.zeros(0, 2048), "integer": torch.zeros(2, 2048, dtype=torch.long),
        "double": torch.zeros(2, 2048, dtype=torch.float64),
        "device": torch.empty(2, 2048, device="meta"),
    }[kind]
    with pytest.raises(error, match="global_features"):
        decoder.init_state(features)
    with pytest.raises(error, match="global_features"):
        decoder(features, torch.ones(2, 3, dtype=torch.long))


@pytest.mark.parametrize("kind,error", [
    ("not_tensor", TypeError), ("rank_one", ValueError), ("rank_three", ValueError),
    ("empty", ValueError), ("float", TypeError), ("int32", TypeError),
    ("negative", ValueError), ("out_of_range", ValueError), ("device", ValueError),
])
def test_invalid_input_ids(decoder: LSTMDecoder, kind: str, error: type[Exception]) -> None:
    tokens = {
        "not_tensor": [1, 2], "rank_one": torch.ones(3, dtype=torch.long),
        "rank_three": torch.ones(2, 3, 1, dtype=torch.long), "empty": torch.empty(2, 0, dtype=torch.long),
        "float": torch.ones(2, 3), "int32": torch.ones(2, 3, dtype=torch.int32),
        "negative": torch.tensor([[1, -1], [1, 2]]),
        "out_of_range": torch.tensor([[1, 100], [1, 2]]),
        "device": torch.empty(2, 3, dtype=torch.long, device="meta"),
    }[kind]
    with pytest.raises(error, match="input_ids"):
        decoder(torch.randn(2, 2048), tokens)


def test_batch_mismatch(decoder: LSTMDecoder) -> None:
    with pytest.raises(ValueError, match="Batch size"):
        decoder(torch.randn(4, 2048), torch.ones(2, 3, dtype=torch.long))


@pytest.mark.parametrize("kind,error", [
    ("container", TypeError), ("length", TypeError), ("not_tensor", TypeError),
    ("layers", ValueError), ("batch", ValueError), ("hidden", ValueError),
    ("cell_shape", ValueError), ("dtype", TypeError), ("cell_dtype", TypeError),
    ("device", ValueError),
])
def test_invalid_step_state(decoder: LSTMDecoder, kind: str, error: type[Exception]) -> None:
    hidden, cell = decoder.init_state(torch.randn(2, 2048))
    bad_state = {
        "container": [hidden, cell], "length": (hidden,), "not_tensor": (None, cell),
        "layers": (torch.zeros(2, 2, 64), cell), "batch": (torch.zeros(1, 3, 64), cell),
        "hidden": (torch.zeros(1, 2, 63), cell), "cell_shape": (hidden, torch.zeros(2, 64)),
        "dtype": (hidden.double(), cell), "cell_dtype": (hidden, cell.long()),
        "device": (torch.empty(1, 2, 64, device="meta"), cell),
    }[kind]
    with pytest.raises(error, match="state"):
        decoder.step(torch.tensor([1, 2]), bad_state)


@pytest.mark.parametrize("token_ids,error", [
    (torch.tensor([[1], [2]]), ValueError), (torch.tensor([1., 2.]), TypeError),
    (torch.tensor([-1, 2]), ValueError), (torch.tensor([1, 100]), ValueError),
    (torch.empty(0, dtype=torch.long), ValueError),
])
def test_invalid_step_tokens(decoder: LSTMDecoder, token_ids: torch.Tensor, error: type[Exception]) -> None:
    with pytest.raises(error, match="token_ids"):
        decoder.step(token_ids, decoder.init_state(torch.randn(2, 2048)))


def test_float64_device_management_and_no_attention() -> None:
    model = LSTMDecoder(13, encoder_dim=7, embedding_dim=5, hidden_dim=9).double().eval()
    features, tokens = torch.randn(2, 7, dtype=torch.double), torch.ones(2, 3, dtype=torch.long)
    assert model(features, tokens).dtype == torch.double
    logits, state = model.step(tokens[:, 0], model.init_state(features))
    assert logits.dtype == state[0].dtype == state[1].dtype == torch.double
    assert all(isinstance(module, (LSTMDecoder, nn.Embedding, nn.Linear, nn.LSTM, nn.Dropout))
               for module in model.modules())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA không khả dụng; môi trường đang dùng CPU wheel.")
def test_cuda_forward_and_step() -> None:
    model = LSTMDecoder(17, encoder_dim=9, embedding_dim=8, hidden_dim=11).to("cuda").eval()
    features = torch.randn(2, 9, device="cuda")
    tokens = torch.ones(2, 3, dtype=torch.long, device="cuda")
    logits = model(features, tokens)
    single, state = model.step(tokens[:, 0], model.init_state(features))
    assert logits.shape == (2, 3, 17) and single.shape == (2, 17)
    assert logits.is_cuda and single.is_cuda and state[0].is_cuda and state[1].is_cuda
    torch.testing.assert_close(single, logits[:, 0], rtol=1e-4, atol=1e-5)


def test_actual_vocabulary_integration() -> None:
    from image_captioning.data.tokenizer_vi import VietnameseTokenizer
    path = Path(__file__).resolve().parents[1] / "data/processed/vocab.json"
    if not path.is_file():
        pytest.skip("Không có vocabulary KTVIC thực tế tại data/processed/vocab.json.")
    vocab = VietnameseTokenizer.load_vocab(path)
    assert [vocab.token_to_id[token] for token in ("<pad>", "<bos>", "<eos>", "<unk>")] == [0, 1, 2, 3]
    model = LSTMDecoder(len(vocab), embedding_dim=16, hidden_dim=24, pad_token_id=vocab.pad_id)
    tokens = torch.tensor([vocab.encode("ba chiếc thuyền đang di chuyển")[:-1]], dtype=torch.long)
    logits = model(torch.randn(1, 2048), tokens)
    assert logits.shape == (1, tokens.shape[1], len(vocab))
    assert torch.isfinite(logits).all()


def test_existing_collate_contract(decoder: LSTMDecoder) -> None:
    from image_captioning.data.dataset import collate_fn
    batch = collate_fn([
        {"image": torch.zeros(3, 224, 224), "image_id": 10, "annotation_id": 1,
         "caption_ids": torch.tensor([1, 4, 2])},
        {"image": torch.zeros(3, 224, 224), "image_id": 11, "annotation_id": 2,
         "caption_ids": torch.tensor([1, 5, 6, 2])},
    ])
    logits = decoder(torch.randn(2, 2048), batch["input_ids"])
    assert logits.shape[:2] == batch["target_ids"].shape == batch["attention_mask"].shape
    assert logits.shape[2] == decoder.vocab_size
    assert batch["input_ids"].tolist() == [[1, 4, 0], [1, 5, 6]]
    assert batch["target_ids"].tolist() == [[4, 2, 0], [5, 6, 2]]


def test_cnn_encoder_offline_compatibility(decoder: LSTMDecoder) -> None:
    from image_captioning.models.cnn_encoder import CNNEncoder
    encoder = CNNEncoder(pretrained=False, freeze_backbone=True)
    decoder.eval()
    with torch.no_grad():
        global_features, _ = encoder(torch.randn(1, 3, 224, 224))
        logits = decoder(global_features, torch.tensor([[1, 4, 5, 2]]))
    assert global_features.shape == (1, encoder.encoder_dim)
    assert logits.shape == (1, 4, decoder.vocab_size)
    assert torch.isfinite(logits).all()
