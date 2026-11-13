"""Docling options read from the configuration: models folder, time limit, status of the conversion."""

from types import SimpleNamespace

from docling.datamodel.base_models import ConversionStatus
from docling_core.types.doc import DoclingDocument

from hgrag import config, parsing


def test_parse_options_keep_the_same_identity_without_a_time_limit(monkeypatch):
    monkeypatch.setattr(config, "DOCLING_TIMEOUT", 0.0)
    assert "document_timeout" not in parsing.parse_options()
    monkeypatch.setattr(config, "DOCLING_TIMEOUT", 900.0)
    assert parsing.parse_options()["document_timeout"] == 900.0


def test_converter_uses_the_models_folder_and_the_time_limit(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DOCLING_ARTIFACTS", str(tmp_path))
    monkeypatch.setattr(config, "DOCLING_TIMEOUT", 900.0)
    captured = {}
    monkeypatch.setattr(parsing, "PdfFormatOption", lambda pipeline_options: captured.update(opts=pipeline_options))
    monkeypatch.setattr(parsing, "DocumentConverter", lambda format_options: None)
    parsing.build_converter()
    assert str(captured["opts"].artifacts_path) == str(tmp_path) and captured["opts"].document_timeout == 900.0


def test_converter_keeps_docling_defaults_without_these_options(monkeypatch):
    monkeypatch.setattr(config, "DOCLING_ARTIFACTS", "")
    monkeypatch.setattr(config, "DOCLING_TIMEOUT", 0.0)
    captured = {}
    monkeypatch.setattr(parsing, "PdfFormatOption", lambda pipeline_options: captured.update(opts=pipeline_options))
    monkeypatch.setattr(parsing, "DocumentConverter", lambda format_options: None)
    parsing.build_converter()
    assert captured["opts"].artifacts_path is None and captured["opts"].document_timeout is None


def test_manifest_keeps_the_status_of_the_conversion(tmp_path):
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.7")
    result = SimpleNamespace(document=DoclingDocument(name="a"), status=ConversionStatus.PARTIAL_SUCCESS)
    converter = SimpleNamespace(convert=lambda _path: result)
    manifest = parsing.parse_pdf(pdf, output_root=tmp_path / "out", converter=converter, describe_pictures=False)
    assert manifest["conversion_status"] == "partial_success"
    assert "docling" in manifest["runtime"]["packages"]  # a fake converter: the layout part is left out, no failure


def test_runtime_names_the_layout_model_its_engine_and_its_weights(tmp_path, monkeypatch):
    """What a local parsing and a Databricks one are compared on when their items differ."""
    weights = tmp_path / "org--layout"
    weights.mkdir()
    (weights / "model.safetensors").write_bytes(b"weights")
    layout = SimpleNamespace(kind="layout_object_detection", engine_options=SimpleNamespace(engine_type="transformers"),
                             model_spec=SimpleNamespace(name="layout", repo_id="org/layout", revision="main"))
    opts = SimpleNamespace(layout_options=layout, artifacts_path=tmp_path,
                           accelerator_options=SimpleNamespace(device="cpu", num_threads=2))
    converter = SimpleNamespace(format_to_options={parsing.InputFormat.PDF: SimpleNamespace(pipeline_options=opts)})
    monkeypatch.setattr(parsing, "_runtime", None)
    info = parsing.runtime_info(converter)
    assert info["layout"]["engine"] == "transformers" and info["device"] == "cpu" and "error" not in info
    import hashlib

    assert info["layout"]["weights"] == {"folder": "org--layout", "file": "model.safetensors", "bytes": 7,
                                         "sha256": hashlib.sha256(b"weights").hexdigest()[:16]}
    monkeypatch.setattr(parsing, "_runtime", None)
