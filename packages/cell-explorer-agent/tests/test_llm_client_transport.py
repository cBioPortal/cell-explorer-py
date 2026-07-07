"""AnthropicLLMClient transport selection + Bedrock runtime deps."""

import anthropic

from cell_explorer_agent import AnthropicLLMClient


def test_bedrock_runtime_dep_importable():
    # anthropic[bedrock] provides boto3, required for Bedrock SigV4/cred resolution
    # at request time (construction alone does not import it).
    import boto3  # noqa: F401


def test_bedrock_transport_uses_region():
    client = AnthropicLLMClient(transport="bedrock", bedrock_region="us-east-1")
    assert isinstance(client._client, anthropic.AsyncAnthropicBedrock)
    assert client._client.aws_region == "us-east-1"


def test_bedrock_transport_region_override():
    client = AnthropicLLMClient(transport="bedrock", bedrock_region="eu-west-1")
    assert client._client.aws_region == "eu-west-1"


def test_anthropic_transport_is_default(monkeypatch):
    # AsyncAnthropic() needs an api key present to construct
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    client = AnthropicLLMClient()
    assert isinstance(client._client, anthropic.AsyncAnthropic)
