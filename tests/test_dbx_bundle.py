"""The bundle only ships the job and the notebooks, holds no secret and follows the version of the package."""

import re
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "databricks"
CONFIG = yaml.safe_load((BUNDLE / "databricks.yml").read_text(encoding="utf-8"))
JOB_FILE = BUNDLE / "resources" / "hgrag_ingest.job.yml"
JOB = yaml.safe_load(JOB_FILE.read_text(encoding="utf-8"))["resources"]["jobs"]["hgrag_ingest"]
STORAGE_FILE = BUNDLE / "resources" / "hgrag_storage.yml"
STOP_FILE = BUNDLE / "resources" / "hgrag_stop_search.job.yml"


def test_wheel_version_follows_the_package():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert CONFIG["variables"]["package_version"]["default"] == project["version"]


def test_bundle_root_holds_no_data_of_the_repository():
    assert not (BUNDLE / "data").exists()
    assert not list(BUNDLE.rglob("*.pdf"))
    assert "sync" not in CONFIG, "the bundle synchronizes its own folder only"


def test_workspace_names_are_variables_without_default():
    for name in ("catalog", "schema", "volume", "search_endpoint"):
        assert "default" not in CONFIG["variables"][name]


def test_schema_and_volume_keep_the_names_of_the_job_and_survive_a_destroy():
    resources = yaml.safe_load(STORAGE_FILE.read_text(encoding="utf-8"))["resources"]
    schema = resources["schemas"]["hgrag_schema"]
    volume = resources["volumes"]["hgrag_volume"]
    assert CONFIG["experimental"]["skip_name_prefix_for_schema"] is True
    assert (schema["catalog_name"], schema["name"]) == ("${var.catalog}", "${var.schema}")
    assert (volume["catalog_name"], volume["name"]) == ("${var.catalog}", "${var.volume}")
    assert schema["lifecycle"]["prevent_destroy"] is True
    assert volume["lifecycle"]["prevent_destroy"] is True
    assert volume["volume_type"] == "MANAGED"


def test_stop_search_job_is_paused_by_default_and_declares_what_its_notebook_reads():
    job = yaml.safe_load(STOP_FILE.read_text(encoding="utf-8"))["resources"]["jobs"]["hgrag_stop_search"]
    assert job["schedule"]["pause_status"] == "${var.stop_search_schedule}"
    assert CONFIG["variables"]["stop_search_schedule"]["default"] == "PAUSED"
    notebook = (STOP_FILE.parent / job["tasks"][0]["notebook_task"]["notebook_path"]).resolve()
    read = set(re.findall(r'dbutils\.widgets\.get\("(\w+)"\)', notebook.read_text(encoding="utf-8")))
    assert read == {p["name"] for p in job["parameters"]}
    wheel = next(p["default"] for p in job["parameters"] if p["name"] == "wheel")
    assert wheel == next(p["default"] for p in JOB["parameters"] if p["name"] == "wheel")


def test_job_installs_the_wheel_the_bundle_deploys():
    artifact = CONFIG["artifacts"]["hgrag"]
    assert (BUNDLE / artifact["path"]).resolve() == ROOT
    wheel = "hgrag-${var.package_version}-py3-none-any.whl"
    # The exact name: a glob would also upload the old wheels of dist/.
    assert [f["source"] for f in artifact["files"]] == [f"../dist/{wheel}"]
    defaults = {p["name"]: p["default"] for p in JOB["parameters"]}
    assert defaults["wheel"] == f"${{workspace.artifact_path}}/.internal/{wheel}"


def test_no_secret_nor_host_in_the_bundle():
    for path in (BUNDLE / "databricks.yml", JOB_FILE, STORAGE_FILE, STOP_FILE):
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"dapi[0-9a-f]{20,}|hf_[A-Za-z0-9]{25,}|sk-ant-|host:|\.cloud\.databricks\.com", text)


def test_indexing_runs_after_parsing_and_runs_do_not_overlap():
    tasks = {t["task_key"]: t for t in JOB["tasks"]}
    assert tasks["index_chunks"]["depends_on"] == [{"task_key": "parse_documents"}]
    assert JOB["max_concurrent_runs"] == 1


def test_task_notebooks_exist_and_declare_every_job_parameter_they_read():
    declared = {p["name"] for p in JOB["parameters"]}
    for task in JOB["tasks"]:
        notebook = (JOB_FILE.parent / task["notebook_task"]["notebook_path"]).resolve()
        assert notebook.is_file(), notebook
        read = set(re.findall(r'dbutils\.widgets\.get\("(\w+)"\)', notebook.read_text(encoding="utf-8")))
        missing = read - declared - {"checkpoint", "document_timeout", "test_query"}  # keep their notebook default
        assert not missing, (notebook.name, missing)
