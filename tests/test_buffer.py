import pytest

from unittest.mock import patch
from pybert.models.buffer import Buffer

@pytest.fixture()
def buffer():
    """Return a basic Buffer Model."""
    yield Buffer()

@patch("pybert.models.buffer.IBISModel.from_ibis_file")
def test_buffer_ibis_filepath_changed_happy_path(mocked_ibis_model, buffer:Buffer):
    buffer._ibis_filepath_changed("")
    assert buffer.use_ibis == True

@patch("pybert.models.buffer.IBISModel.from_ibis_file", side_effect=ValueError("Fake Exception Thrown when Reading IBIS Model."))
def test_buffer_ibis_filepath_changed_throws_exception(mocked_ibis_model, buffer:Buffer):
    buffer._ibis_filepath_changed("")
    assert buffer.use_ibis == False

@patch("pybert.models.buffer.AMIParamConfigurator")
def test_buffer_ami_filepath_changed_happy_path(mocked_ami_configurator, buffer:Buffer):
    buffer._ami_filepath_changed("")
    assert buffer.ami_config

@patch("pybert.models.buffer.AMIParamConfigurator", side_effect=ValueError("Fake Exception Thrown when Reading AMI Config."))
def test_buffer_ami_filepath_changed_throws_exception(mocked_ami_configurator, buffer:Buffer):
    buffer._ami_filepath_changed("")
    assert buffer.ami_config is None

@patch("pybert.models.buffer.AMIModel")
def test_buffer_ami_filepath_changed_happy_path(mocked_ami_model, buffer:Buffer):
    buffer._dll_filepath_changed("")
    assert buffer.ami_model

@patch("pybert.models.buffer.AMIModel", side_effect=ValueError("Fake Exception Thrown when binding AMI Model."))
def test_buffer_dll_filepath_changed_throws_exception(mocked_ami_model, buffer:Buffer):
    buffer._dll_filepath_changed("")
    assert buffer.ami_model is None
