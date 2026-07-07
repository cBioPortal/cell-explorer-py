"""AnthropicLLMClient transport selection + Bedrock runtime deps."""


def test_bedrock_runtime_dep_importable():
    # anthropic[bedrock] provides boto3, required for Bedrock SigV4/cred resolution
    # at request time (construction alone does not import it).
    import boto3  # noqa: F401
