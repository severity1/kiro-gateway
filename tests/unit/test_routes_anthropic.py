
# -*- coding: utf-8 -*-

"""
Unit tests for Anthropic API endpoints (routes_anthropic.py).

Tests the following endpoint:
- POST /v1/messages - Anthropic Messages API

For OpenAI API tests, see test_routes_openai.py.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from datetime import datetime, timezone
import json

from fastapi import HTTPException
from fastapi.testclient import TestClient

from kiro.routes_anthropic import verify_anthropic_api_key, router
from kiro.config import PROXY_API_KEY


# =============================================================================
# Tests for verify_anthropic_api_key function
# =============================================================================

class TestVerifyAnthropicApiKey:
    """Tests for the verify_anthropic_api_key authentication function."""
    
    @pytest.mark.asyncio
    async def test_valid_x_api_key_returns_true(self):
        """
        What it does: Verifies that a valid x-api-key header passes authentication.
        Purpose: Ensure Anthropic native authentication works.
        """
        print("Setup: Creating valid x-api-key...")
        
        print("Action: Calling verify_anthropic_api_key...")
        result = await verify_anthropic_api_key(x_api_key=PROXY_API_KEY, authorization=None)
        
        print(f"Comparing result: Expected True, Got {result}")
        assert result is True
    
    @pytest.mark.asyncio
    async def test_valid_bearer_token_returns_true(self):
        """
        What it does: Verifies that a valid Bearer token passes authentication.
        Purpose: Ensure OpenAI-style authentication also works.
        """
        print("Setup: Creating valid Bearer token...")
        valid_auth = f"Bearer {PROXY_API_KEY}"
        
        print("Action: Calling verify_anthropic_api_key...")
        result = await verify_anthropic_api_key(x_api_key=None, authorization=valid_auth)
        
        print(f"Comparing result: Expected True, Got {result}")
        assert result is True
    
    @pytest.mark.asyncio
    async def test_x_api_key_takes_precedence(self):
        """
        What it does: Verifies x-api-key is checked before Authorization header.
        Purpose: Ensure Anthropic native auth has priority.
        """
        print("Setup: Both headers provided...")
        
        print("Action: Calling verify_anthropic_api_key with both headers...")
        result = await verify_anthropic_api_key(
            x_api_key=PROXY_API_KEY,
            authorization="Bearer wrong_key"
        )
        
        print(f"Comparing result: Expected True, Got {result}")
        assert result is True
    
    @pytest.mark.asyncio
    async def test_invalid_x_api_key_raises_401(self):
        """
        What it does: Verifies that an invalid x-api-key is rejected.
        Purpose: Ensure unauthorized access is blocked.
        """
        print("Setup: Creating invalid x-api-key...")
        
        print("Action: Calling verify_anthropic_api_key with invalid key...")
        with pytest.raises(HTTPException) as exc_info:
            await verify_anthropic_api_key(x_api_key="wrong_key", authorization=None)
        
        print(f"Checking: HTTPException with status 401...")
        assert exc_info.value.status_code == 401
    
    @pytest.mark.asyncio
    async def test_invalid_bearer_token_raises_401(self):
        """
        What it does: Verifies that an invalid Bearer token is rejected.
        Purpose: Ensure unauthorized access is blocked.
        """
        print("Setup: Creating invalid Bearer token...")
        
        print("Action: Calling verify_anthropic_api_key with invalid token...")
        with pytest.raises(HTTPException) as exc_info:
            await verify_anthropic_api_key(x_api_key=None, authorization="Bearer wrong_key")
        
        print(f"Checking: HTTPException with status 401...")
        assert exc_info.value.status_code == 401
    
    @pytest.mark.asyncio
    async def test_missing_both_headers_raises_401(self):
        """
        What it does: Verifies that missing both headers is rejected.
        Purpose: Ensure authentication is required.
        """
        print("Setup: No authentication headers...")
        
        print("Action: Calling verify_anthropic_api_key with no headers...")
        with pytest.raises(HTTPException) as exc_info:
            await verify_anthropic_api_key(x_api_key=None, authorization=None)
        
        print(f"Checking: HTTPException with status 401...")
        assert exc_info.value.status_code == 401
    
    @pytest.mark.asyncio
    async def test_empty_x_api_key_raises_401(self):
        """
        What it does: Verifies that empty x-api-key is rejected.
        Purpose: Ensure empty credentials are blocked.
        """
        print("Setup: Empty x-api-key...")
        
        print("Action: Calling verify_anthropic_api_key with empty key...")
        with pytest.raises(HTTPException) as exc_info:
            await verify_anthropic_api_key(x_api_key="", authorization=None)
        
        print(f"Checking: HTTPException with status 401...")
        assert exc_info.value.status_code == 401
    
    @pytest.mark.asyncio
    async def test_error_response_format_is_anthropic_style(self):
        """
        What it does: Verifies error response follows Anthropic format.
        Purpose: Ensure error format matches Anthropic API.
        """
        print("Setup: Invalid credentials...")
        
        print("Action: Calling verify_anthropic_api_key...")
        with pytest.raises(HTTPException) as exc_info:
            await verify_anthropic_api_key(x_api_key="wrong", authorization=None)
        
        print(f"Checking: Error format...")
        detail = exc_info.value.detail
        assert "type" in detail
        assert "error" in detail
        assert detail["error"]["type"] == "authentication_error"


# =============================================================================
# Tests for /v1/messages endpoint authentication
# =============================================================================

class TestMessagesAuthentication:
    """Tests for authentication on /v1/messages endpoint."""
    
    def test_messages_requires_authentication(self, test_client):
        """
        What it does: Verifies messages endpoint requires authentication.
        Purpose: Ensure protected endpoint is secured.
        """
        print("Action: POST /v1/messages without auth...")
        response = test_client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )
        
        print(f"Status: {response.status_code}")
        assert response.status_code == 401
    
    def test_messages_accepts_x_api_key(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies messages endpoint accepts x-api-key header.
        Purpose: Ensure Anthropic native authentication works.
        """
        print("Action: POST /v1/messages with x-api-key...")
        response = test_client.post(
            "/v1/messages",
            headers={"x-api-key": valid_proxy_api_key},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )
        
        print(f"Status: {response.status_code}")
        # Should pass auth (not 401)
        assert response.status_code != 401
    
    def test_messages_accepts_bearer_token(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies messages endpoint accepts Bearer token.
        Purpose: Ensure OpenAI-style authentication also works.
        """
        print("Action: POST /v1/messages with Bearer token...")
        response = test_client.post(
            "/v1/messages",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )
        
        print(f"Status: {response.status_code}")
        # Should pass auth (not 401)
        assert response.status_code != 401
    
    def test_messages_rejects_invalid_x_api_key(self, test_client, invalid_proxy_api_key):
        """
        What it does: Verifies messages endpoint rejects invalid x-api-key.
        Purpose: Ensure authentication is enforced.
        """
        print("Action: POST /v1/messages with invalid x-api-key...")
        response = test_client.post(
            "/v1/messages",
            headers={"x-api-key": invalid_proxy_api_key},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )
        
        print(f"Status: {response.status_code}")
        assert response.status_code == 401


# =============================================================================
# Tests for /v1/messages endpoint validation
# =============================================================================

class TestMessagesValidation:
    """Tests for request validation on /v1/messages endpoint."""
    
    def test_validates_missing_model(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies missing model field is rejected.
        Purpose: Ensure model is required.
        """
        print("Action: POST /v1/messages without model...")
        response = test_client.post(
            "/v1/messages",
            headers={"x-api-key": valid_proxy_api_key},
            json={
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )
        
        print(f"Status: {response.status_code}")
        assert response.status_code == 422
    
    def test_validates_missing_max_tokens(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies missing max_tokens field is rejected.
        Purpose: Ensure max_tokens is required (Anthropic API requirement).
        """
        print("Action: POST /v1/messages without max_tokens...")
        response = test_client.post(
            "/v1/messages",
            headers={"x-api-key": valid_proxy_api_key},
            json={
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )
        
        print(f"Status: {response.status_code}")
        assert response.status_code == 422
    
    def test_validates_missing_messages(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies missing messages field is rejected.
        Purpose: Ensure messages are required.
        """
        print("Action: POST /v1/messages without messages...")
        response = test_client.post(
            "/v1/messages",
            headers={"x-api-key": valid_proxy_api_key},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024
            }
        )
        
        print(f"Status: {response.status_code}")
        assert response.status_code == 422
    
    def test_validates_empty_messages_array(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies empty messages array is rejected.
        Purpose: Ensure at least one message is required.
        """
        print("Action: POST /v1/messages with empty messages...")
        response = test_client.post(
            "/v1/messages",
            headers={"x-api-key": valid_proxy_api_key},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "messages": []
            }
        )
        
        print(f"Status: {response.status_code}")
        assert response.status_code == 422
    
    def test_validates_invalid_json(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies invalid JSON is rejected.
        Purpose: Ensure proper JSON parsing.
        """
        print("Action: POST /v1/messages with invalid JSON...")
        response = test_client.post(
            "/v1/messages",
            headers={
                "x-api-key": valid_proxy_api_key,
                "Content-Type": "application/json"
            },
            content=b"not valid json {{{}"
        )
        
        print(f"Status: {response.status_code}")
        assert response.status_code == 422
    
    def test_validates_invalid_role(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies invalid message role is rejected.
        Purpose: Anthropic model strictly validates role (only 'user' or 'assistant').
        """
        print("Action: POST /v1/messages with invalid role...")
        response = test_client.post(
            "/v1/messages",
            headers={"x-api-key": valid_proxy_api_key},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "messages": [{"role": "invalid_role", "content": "Hello"}]
            }
        )
        
        print(f"Status: {response.status_code}")
        # Anthropic model strictly validates role - only 'user' or 'assistant' allowed
        assert response.status_code == 422
    
    def test_accepts_valid_request_format(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies valid request format passes validation.
        Purpose: Ensure Pydantic validation works correctly.
        """
        print("Action: POST /v1/messages with valid format...")
        response = test_client.post(
            "/v1/messages",
            headers={"x-api-key": valid_proxy_api_key},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )
        
        print(f"Status: {response.status_code}")
        # Should pass validation (not 422)
        assert response.status_code != 422


# =============================================================================
# Tests for /v1/messages system prompt
# =============================================================================

class TestMessagesSystemPrompt:
    """Tests for system prompt handling on /v1/messages endpoint."""
    
    def test_accepts_system_as_separate_field(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies system prompt as separate field is accepted.
        Purpose: Ensure Anthropic-style system prompt works.
        """
        print("Action: POST /v1/messages with system field...")
        response = test_client.post(
            "/v1/messages",
            headers={"x-api-key": valid_proxy_api_key},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "system": "You are a helpful assistant.",
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )
        
        print(f"Status: {response.status_code}")
        # Should pass validation
        assert response.status_code != 422
    
    def test_accepts_empty_system_prompt(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies empty system prompt is accepted.
        Purpose: Ensure system prompt is optional.
        """
        print("Action: POST /v1/messages with empty system...")
        response = test_client.post(
            "/v1/messages",
            headers={"x-api-key": valid_proxy_api_key},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "system": "",
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )
        
        print(f"Status: {response.status_code}")
        # Should pass validation
        assert response.status_code != 422
    
    def test_accepts_no_system_prompt(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies request without system prompt is accepted.
        Purpose: Ensure system prompt is optional.
        """
        print("Action: POST /v1/messages without system field...")
        response = test_client.post(
            "/v1/messages",
            headers={"x-api-key": valid_proxy_api_key},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )
        
        print(f"Status: {response.status_code}")
        # Should pass validation
        assert response.status_code != 422


# =============================================================================
# Tests for /v1/messages content blocks
# =============================================================================

class TestMessagesContentBlocks:
    """Tests for content block handling on /v1/messages endpoint."""
    
    def test_accepts_string_content(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies string content is accepted.
        Purpose: Ensure simple string content works.
        """
        print("Action: POST /v1/messages with string content...")
        response = test_client.post(
            "/v1/messages",
            headers={"x-api-key": valid_proxy_api_key},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )
        
        print(f"Status: {response.status_code}")
        assert response.status_code != 422
    
    def test_accepts_content_block_array(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies content block array is accepted.
        Purpose: Ensure Anthropic content block format works.
        """
        print("Action: POST /v1/messages with content blocks...")
        response = test_client.post(
            "/v1/messages",
            headers={"x-api-key": valid_proxy_api_key},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Hello"}
                        ]
                    }
                ]
            }
        )
        
        print(f"Status: {response.status_code}")
        assert response.status_code != 422
    
    def test_accepts_multiple_content_blocks(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies multiple content blocks are accepted.
        Purpose: Ensure complex content works.
        """
        print("Action: POST /v1/messages with multiple content blocks...")
        response = test_client.post(
            "/v1/messages",
            headers={"x-api-key": valid_proxy_api_key},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "First part"},
                            {"type": "text", "text": "Second part"}
                        ]
                    }
                ]
            }
        )
        
        print(f"Status: {response.status_code}")
        assert response.status_code != 422


# =============================================================================
# Tests for /v1/messages tool use
# =============================================================================

class TestMessagesToolUse:
    """Tests for tool use on /v1/messages endpoint."""
    
    def test_accepts_tool_definition(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies tool definition is accepted.
        Purpose: Ensure Anthropic tool format works.
        """
        print("Action: POST /v1/messages with tools...")
        response = test_client.post(
            "/v1/messages",
            headers={"x-api-key": valid_proxy_api_key},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "What's the weather?"}],
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Get weather for a location",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"}
                            },
                            "required": ["location"]
                        }
                    }
                ]
            }
        )
        
        print(f"Status: {response.status_code}")
        assert response.status_code != 422
    
    def test_accepts_multiple_tools(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies multiple tools are accepted.
        Purpose: Ensure multiple tool definitions work.
        """
        print("Action: POST /v1/messages with multiple tools...")
        response = test_client.post(
            "/v1/messages",
            headers={"x-api-key": valid_proxy_api_key},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}],
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Get weather",
                        "input_schema": {"type": "object", "properties": {}}
                    },
                    {
                        "name": "get_time",
                        "description": "Get time",
                        "input_schema": {"type": "object", "properties": {}}
                    }
                ]
            }
        )
        
        print(f"Status: {response.status_code}")
        assert response.status_code != 422
    
    def test_accepts_tool_result_message(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies tool result message is accepted.
        Purpose: Ensure tool result handling works.
        """
        print("Action: POST /v1/messages with tool result...")
        response = test_client.post(
            "/v1/messages",
            headers={"x-api-key": valid_proxy_api_key},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "messages": [
                    {"role": "user", "content": "What's the weather?"},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "call_123",
                                "name": "get_weather",
                                "input": {"location": "Moscow"}
                            }
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "call_123",
                                "content": "Sunny, 25°C"
                            }
                        ]
                    }
                ]
            }
        )
        
        print(f"Status: {response.status_code}")
        assert response.status_code != 422


# =============================================================================
# Tests for /v1/messages optional parameters
# =============================================================================

class TestMessagesOptionalParams:
    """Tests for optional parameters on /v1/messages endpoint."""
    
    def test_accepts_temperature_parameter(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies temperature parameter is accepted.
        Purpose: Ensure temperature control works.
        """
        print("Action: POST /v1/messages with temperature...")
        response = test_client.post(
            "/v1/messages",
            headers={"x-api-key": valid_proxy_api_key},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0.7
            }
        )
        
        print(f"Status: {response.status_code}")
        assert response.status_code != 422
    
    def test_accepts_top_p_parameter(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies top_p parameter is accepted.
        Purpose: Ensure nucleus sampling control works.
        """
        print("Action: POST /v1/messages with top_p...")
        response = test_client.post(
            "/v1/messages",
            headers={"x-api-key": valid_proxy_api_key},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}],
                "top_p": 0.9
            }
        )
        
        print(f"Status: {response.status_code}")
        assert response.status_code != 422
    
    def test_accepts_top_k_parameter(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies top_k parameter is accepted.
        Purpose: Ensure top-k sampling control works.
        """
        print("Action: POST /v1/messages with top_k...")
        response = test_client.post(
            "/v1/messages",
            headers={"x-api-key": valid_proxy_api_key},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}],
                "top_k": 40
            }
        )
        
        print(f"Status: {response.status_code}")
        assert response.status_code != 422
    
    def test_accepts_stream_true(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies stream=true is accepted.
        Purpose: Ensure streaming mode is supported.
        """
        print("Action: POST /v1/messages with stream=true...")
        
        # Mock the streaming function to avoid real HTTP requests
        async def mock_stream(*args, **kwargs):
            yield 'event: message_start\ndata: {"type":"message_start"}\n\n'
            yield 'event: message_stop\ndata: {"type":"message_stop"}\n\n'
        
        # Create mock response for HTTP client
        mock_response = MagicMock()
        mock_response.status_code = 200
        
        with patch('kiro.routes_anthropic.stream_kiro_to_anthropic', mock_stream), \
             patch('kiro.http_client.KiroHttpClient.request_with_retry', return_value=mock_response):
            response = test_client.post(
                "/v1/messages",
                headers={"x-api-key": valid_proxy_api_key},
                json={
                    "model": "claude-sonnet-4-5",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": True
                }
            )
        
        print(f"Status: {response.status_code}")
        assert response.status_code != 422
    
    def test_accepts_stop_sequences(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies stop_sequences parameter is accepted.
        Purpose: Ensure stop sequence control works.
        """
        print("Action: POST /v1/messages with stop_sequences...")
        response = test_client.post(
            "/v1/messages",
            headers={"x-api-key": valid_proxy_api_key},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}],
                "stop_sequences": ["END", "STOP"]
            }
        )
        
        print(f"Status: {response.status_code}")
        assert response.status_code != 422
    
    def test_accepts_metadata(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies metadata parameter is accepted.
        Purpose: Ensure metadata passing works.
        """
        print("Action: POST /v1/messages with metadata...")
        response = test_client.post(
            "/v1/messages",
            headers={"x-api-key": valid_proxy_api_key},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}],
                "metadata": {"user_id": "test_user"}
            }
        )
        
        print(f"Status: {response.status_code}")
        assert response.status_code != 422


# =============================================================================
# Tests for /v1/messages anthropic-version header
# =============================================================================

class TestMessagesAnthropicVersion:
    """Tests for anthropic-version header handling."""
    
    def test_accepts_anthropic_version_header(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies anthropic-version header is accepted.
        Purpose: Ensure Anthropic SDK compatibility.
        """
        print("Action: POST /v1/messages with anthropic-version header...")
        response = test_client.post(
            "/v1/messages",
            headers={
                "x-api-key": valid_proxy_api_key,
                "anthropic-version": "2023-06-01"
            },
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )
        
        print(f"Status: {response.status_code}")
        # Should pass validation
        assert response.status_code != 422
    
    def test_works_without_anthropic_version_header(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies request works without anthropic-version header.
        Purpose: Ensure header is optional.
        """
        print("Action: POST /v1/messages without anthropic-version header...")
        response = test_client.post(
            "/v1/messages",
            headers={"x-api-key": valid_proxy_api_key},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )
        
        print(f"Status: {response.status_code}")
        # Should pass validation
        assert response.status_code != 422


# =============================================================================
# Tests for router integration
# =============================================================================

class TestAnthropicRouterIntegration:
    """Tests for Anthropic router configuration and integration."""
    
    def test_router_has_messages_endpoint(self):
        """
        What it does: Verifies messages endpoint is registered.
        Purpose: Ensure endpoint is available.
        """
        print("Checking: Router endpoints...")
        routes = [route.path for route in router.routes]
        
        print(f"Found routes: {routes}")
        assert "/v1/messages" in routes
    
    def test_messages_endpoint_uses_post_method(self):
        """
        What it does: Verifies messages endpoint uses POST method.
        Purpose: Ensure correct HTTP method.
        """
        print("Checking: HTTP methods...")
        for route in router.routes:
            if route.path == "/v1/messages":
                print(f"Route /v1/messages methods: {route.methods}")
                assert "POST" in route.methods
                return
        pytest.fail("Messages endpoint not found")
    
    def test_router_has_anthropic_tag(self):
        """
        What it does: Verifies router has Anthropic API tag.
        Purpose: Ensure proper API documentation grouping.
        """
        print("Checking: Router tags...")
        print(f"Router tags: {router.tags}")
        assert "Anthropic API" in router.tags


# =============================================================================
# Tests for conversation history
# =============================================================================

class TestMessagesConversationHistory:
    """Tests for conversation history handling on /v1/messages endpoint."""
    
    def test_accepts_multi_turn_conversation(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies multi-turn conversation is accepted.
        Purpose: Ensure conversation history works.
        """
        print("Action: POST /v1/messages with conversation history...")
        response = test_client.post(
            "/v1/messages",
            headers={"x-api-key": valid_proxy_api_key},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                    {"role": "user", "content": "How are you?"}
                ]
            }
        )
        
        print(f"Status: {response.status_code}")
        assert response.status_code != 422
    
    def test_accepts_long_conversation(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies long conversation is accepted.
        Purpose: Ensure many messages work.
        """
        print("Action: POST /v1/messages with long conversation...")
        messages = []
        for i in range(10):
            messages.append({"role": "user", "content": f"Message {i}"})
            messages.append({"role": "assistant", "content": f"Response {i}"})
        messages.append({"role": "user", "content": "Final question"})
        
        response = test_client.post(
            "/v1/messages",
            headers={"x-api-key": valid_proxy_api_key},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "messages": messages
            }
        )
        
        print(f"Status: {response.status_code}")
        assert response.status_code != 422


# =============================================================================
# Tests for error response format
# =============================================================================

class TestMessagesErrorFormat:
    """Tests for error response format on /v1/messages endpoint."""
    
    def test_validation_error_format(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies validation error response format.
        Purpose: Ensure errors follow expected format.
        """
        print("Action: POST /v1/messages with invalid request...")
        response = test_client.post(
            "/v1/messages",
            headers={"x-api-key": valid_proxy_api_key},
            json={
                "model": "claude-sonnet-4-5"
                # Missing required fields
            }
        )
        
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        assert response.status_code == 422
    
    def test_auth_error_format_is_anthropic_style(self, test_client):
        """
        What it does: Verifies auth error follows Anthropic format.
        Purpose: Ensure error format matches Anthropic API.
        """
        print("Action: POST /v1/messages without auth...")
        response = test_client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )
        
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        assert response.status_code == 401
        
        # Check Anthropic error format
        data = response.json()
        assert "detail" in data
        detail = data["detail"]
        assert "type" in detail
        assert "error" in detail


# =============================================================================
# Tests for HTTP client selection (issue #54)
# =============================================================================

class TestAnthropicHTTPClientSelection:
    """
    Tests for HTTP client selection in Anthropic routes (issue #54).
    
    Verifies that streaming requests use per-request clients to avoid CLOSE_WAIT leak
    when network interface changes (VPN disconnect/reconnect), while non-streaming
    requests use shared client for connection pooling.
    """
    
    @patch('kiro.routes_anthropic.KiroHttpClient')
    def test_streaming_uses_per_request_client(
        self,
        mock_kiro_http_client_class,
        test_client,
        valid_proxy_api_key
    ):
        """
        What it does: Verifies streaming requests create per-request HTTP client.
        Purpose: Prevent CLOSE_WAIT leak on VPN disconnect (issue #54).
        """
        print("\n--- Test: Anthropic streaming uses per-request client ---")
        
        # Setup mock
        mock_client_instance = AsyncMock()
        mock_client_instance.request_with_retry = AsyncMock(
            side_effect=Exception("Network blocked")
        )
        mock_client_instance.close = AsyncMock()
        mock_kiro_http_client_class.return_value = mock_client_instance
        
        print("Action: POST /v1/messages with stream=true...")
        try:
            test_client.post(
                "/v1/messages",
                headers={"x-api-key": valid_proxy_api_key},
                json={
                    "model": "claude-sonnet-4-5",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": True
                }
            )
        except Exception:
            pass
        
        print("Checking: KiroHttpClient(shared_client=None)...")
        assert mock_kiro_http_client_class.called
        call_args = mock_kiro_http_client_class.call_args
        print(f"Call args: {call_args}")
        assert call_args[1]['shared_client'] is None, \
            "Streaming should use per-request client"
        print("✅ Anthropic streaming correctly uses per-request client")
    
    @patch('kiro.routes_anthropic.KiroHttpClient')
    def test_non_streaming_uses_shared_client(
        self,
        mock_kiro_http_client_class,
        test_client,
        valid_proxy_api_key
    ):
        """
        What it does: Verifies non-streaming requests use shared HTTP client.
        Purpose: Ensure connection pooling for non-streaming requests.
        """
        print("\n--- Test: Anthropic non-streaming uses shared client ---")
        
        # Setup mock
        mock_client_instance = AsyncMock()
        mock_client_instance.request_with_retry = AsyncMock(
            side_effect=Exception("Network blocked")
        )
        mock_client_instance.close = AsyncMock()
        mock_kiro_http_client_class.return_value = mock_client_instance
        
        print("Action: POST /v1/messages with stream=false...")
        try:
            test_client.post(
                "/v1/messages",
                headers={"x-api-key": valid_proxy_api_key},
                json={
                    "model": "claude-sonnet-4-5",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": False
                }
            )
        except Exception:
            pass
        
        print("Checking: KiroHttpClient(shared_client=app.state.http_client)...")
        assert mock_kiro_http_client_class.called
        call_args = mock_kiro_http_client_class.call_args
        print(f"Call args: {call_args}")
        assert call_args[1]['shared_client'] is not None, \
            "Non-streaming should use shared client"
        print("✅ Anthropic non-streaming correctly uses shared client")


# =============================================================================
# Tests for Truncation Recovery message modification (Issue #56)
# =============================================================================

class TestTruncationRecoveryMessageModification:
    """
    Tests for Truncation Recovery System message modification in routes_anthropic.
    
    Verifies that tool_result blocks are modified when truncation info exists in cache.
    Part of Truncation Recovery System (Issue #56).
    """
    
    @staticmethod
    def _get_block_value(block, key, default=""):
        """Helper to get value from dict or Pydantic object."""
        if isinstance(block, dict):
            return block.get(key, default)
        else:
            return getattr(block, key, default)
    
    def test_modifies_tool_result_dict_with_truncation_notice(self):
        """
        What it does: Verifies tool_result content block is modified when truncation info exists.
        Purpose: Ensure truncation notice is prepended to tool_result.
        """
        print("Setup: Saving truncation info to cache...")
        from kiro.truncation_state import save_tool_truncation
        from kiro.models_anthropic import AnthropicMessage
        
        tool_use_id = "tooluse_test_dict"
        save_tool_truncation(tool_use_id, "write_to_file", {"size_bytes": 5000, "reason": "test"})
        
        print("Setup: Creating request with tool_result...")
        messages = [
            AnthropicMessage(
                role="user",
                content=[
                    {"type": "tool_result", "tool_use_id": tool_use_id, "content": "Missing parameter error"}
                ]
            )
        ]
        
        print("Action: Processing messages through truncation recovery logic...")
        from kiro.truncation_recovery import should_inject_recovery, generate_truncation_tool_result
        from kiro.truncation_state import get_tool_truncation
        
        modified_messages = []
        for msg in messages:
            if msg.role == "user" and msg.content and isinstance(msg.content, list):
                modified_content_blocks = []
                has_modifications = False
                
                for block in msg.content:
                    block_type = self._get_block_value(block, "type")
                    block_tool_use_id = self._get_block_value(block, "tool_use_id")
                    original_content = self._get_block_value(block, "content", "")
                    
                    if block_type == "tool_result" and block_tool_use_id and should_inject_recovery():
                        truncation_info = get_tool_truncation(block_tool_use_id)
                        if truncation_info:
                            print(f"Found truncation info for {block_tool_use_id}")
                            synthetic = generate_truncation_tool_result(
                                truncation_info.tool_name,
                                truncation_info.tool_call_id,
                                truncation_info.truncation_info
                            )
                            modified_content = f"{synthetic['content']}\n\n---\n\nOriginal tool result:\n{original_content}"
                            
                            if isinstance(block, dict):
                                modified_block = block.copy()
                                modified_block["content"] = modified_content
                            else:
                                modified_block = block.model_copy(update={"content": modified_content})
                            
                            modified_content_blocks.append(modified_block)
                            has_modifications = True
                            continue
                    
                    modified_content_blocks.append(block)
                
                if has_modifications:
                    modified_msg = msg.model_copy(update={"content": modified_content_blocks})
                    modified_messages.append(modified_msg)
                    continue
            
            modified_messages.append(msg)
        
        print("Checking: Modified message content...")
        modified_msg = modified_messages[0]
        modified_block = modified_msg.content[0]
        content = self._get_block_value(modified_block, "content")
        print(f"Content: {content[:100]}...")
        
        assert "[API Limitation]" in content
        assert "Missing parameter error" in content
        assert "---" in content
    
    def test_modifies_tool_result_pydantic_with_truncation_notice(self):
        """
        What it does: Verifies tool_result content block (Pydantic) is modified when truncation info exists.
        Purpose: Ensure truncation notice works with Pydantic ToolResultContentBlock.
        """
        print("Setup: Saving truncation info to cache...")
        from kiro.truncation_state import save_tool_truncation
        from kiro.models_anthropic import AnthropicMessage, ToolResultContentBlock
        
        tool_use_id = "tooluse_test_pydantic"
        save_tool_truncation(tool_use_id, "write_to_file", {"size_bytes": 5000, "reason": "test"})
        
        print("Setup: Creating request with tool_result (Pydantic format)...")
        tool_result_block = ToolResultContentBlock(
            type="tool_result",
            tool_use_id=tool_use_id,
            content="Missing parameter error"
        )
        
        messages = [
            AnthropicMessage(role="user", content=[tool_result_block])
        ]
        
        print("Action: Processing messages through truncation recovery logic...")
        from kiro.truncation_recovery import should_inject_recovery, generate_truncation_tool_result
        from kiro.truncation_state import get_tool_truncation
        
        modified_messages = []
        for msg in messages:
            if msg.role == "user" and msg.content and isinstance(msg.content, list):
                modified_content_blocks = []
                has_modifications = False
                
                for block in msg.content:
                    block_type = self._get_block_value(block, "type")
                    block_tool_use_id = self._get_block_value(block, "tool_use_id")
                    original_content = self._get_block_value(block, "content", "")
                    
                    if block_type == "tool_result" and block_tool_use_id and should_inject_recovery():
                        truncation_info = get_tool_truncation(block_tool_use_id)
                        if truncation_info:
                            print(f"Found truncation info for {block_tool_use_id}")
                            synthetic = generate_truncation_tool_result(
                                truncation_info.tool_name,
                                truncation_info.tool_call_id,
                                truncation_info.truncation_info
                            )
                            modified_content = f"{synthetic['content']}\n\n---\n\nOriginal tool result:\n{original_content}"
                            
                            if isinstance(block, dict):
                                modified_block = block.copy()
                                modified_block["content"] = modified_content
                            else:
                                modified_block = block.model_copy(update={"content": modified_content})
                            
                            modified_content_blocks.append(modified_block)
                            has_modifications = True
                            continue
                    
                    modified_content_blocks.append(block)
                
                if has_modifications:
                    modified_msg = msg.model_copy(update={"content": modified_content_blocks})
                    modified_messages.append(modified_msg)
                    continue
            
            modified_messages.append(msg)
        
        print("Checking: Modified message content...")
        modified_msg = modified_messages[0]
        modified_block = modified_msg.content[0]
        content = self._get_block_value(modified_block, "content")
        print(f"Content: {content[:100]}...")
        
        assert "[API Limitation]" in content
        assert "Missing parameter error" in content
        assert "---" in content
    
    def test_mixed_content_blocks_only_tool_result_modified(self):
        """
        What it does: Verifies only tool_result blocks are modified, text blocks unchanged.
        Purpose: Ensure selective modification of content blocks.
        """
        print("Setup: Saving truncation info to cache...")
        from kiro.truncation_state import save_tool_truncation
        from kiro.models_anthropic import AnthropicMessage
        
        tool_use_id = "tooluse_test_mixed"
        save_tool_truncation(tool_use_id, "write_to_file", {"size_bytes": 5000, "reason": "test"})
        
        print("Setup: Creating request with mixed content blocks...")
        messages = [
            AnthropicMessage(
                role="user",
                content=[
                    {"type": "text", "text": "Here's the result:"},
                    {"type": "tool_result", "tool_use_id": tool_use_id, "content": "Error"}
                ]
            )
        ]
        
        print("Action: Processing messages through truncation recovery logic...")
        from kiro.truncation_recovery import should_inject_recovery, generate_truncation_tool_result
        from kiro.truncation_state import get_tool_truncation
        
        modified_messages = []
        for msg in messages:
            if msg.role == "user" and msg.content and isinstance(msg.content, list):
                modified_content_blocks = []
                has_modifications = False
                
                for block in msg.content:
                    block_type = self._get_block_value(block, "type")
                    block_tool_use_id = self._get_block_value(block, "tool_use_id")
                    original_content = self._get_block_value(block, "content", "")
                    
                    if block_type == "tool_result" and block_tool_use_id and should_inject_recovery():
                        truncation_info = get_tool_truncation(block_tool_use_id)
                        if truncation_info:
                            synthetic = generate_truncation_tool_result(
                                truncation_info.tool_name,
                                truncation_info.tool_call_id,
                                truncation_info.truncation_info
                            )
                            modified_content = f"{synthetic['content']}\n\n---\n\nOriginal tool result:\n{original_content}"
                            
                            if isinstance(block, dict):
                                modified_block = block.copy()
                                modified_block["content"] = modified_content
                            else:
                                modified_block = block.model_copy(update={"content": modified_content})
                            
                            modified_content_blocks.append(modified_block)
                            has_modifications = True
                            continue
                    
                    modified_content_blocks.append(block)
                
                if has_modifications:
                    modified_msg = msg.model_copy(update={"content": modified_content_blocks})
                    modified_messages.append(modified_msg)
                    continue
            
            modified_messages.append(msg)
        
        print("Checking: Text block unchanged...")
        modified_msg = modified_messages[0]
        text_block = modified_msg.content[0]
        assert self._get_block_value(text_block, "type") == "text"
        assert self._get_block_value(text_block, "text") == "Here's the result:"
        
        print("Checking: Tool_result block modified...")
        tool_result_block = modified_msg.content[1]
        assert self._get_block_value(tool_result_block, "type") == "tool_result"
        tool_content = self._get_block_value(tool_result_block, "content")
        assert "[API Limitation]" in tool_content
        assert "Error" in tool_content
        
        print("Checking: Order preserved...")
        assert len(modified_msg.content) == 2
    
    def test_no_modification_when_no_truncation(self):
        """
        What it does: Verifies messages are not modified when no truncation info exists.
        Purpose: Ensure normal messages pass through unchanged.
        """
        print("Setup: Creating request without truncation info in cache...")
        from kiro.models_anthropic import AnthropicMessage
        
        messages = [
            AnthropicMessage(
                role="user",
                content=[
                    {"type": "tool_result", "tool_use_id": "tooluse_nonexistent", "content": "Success"}
                ]
            )
        ]
        
        print("Action: Processing messages...")
        from kiro.truncation_recovery import should_inject_recovery
        from kiro.truncation_state import get_tool_truncation
        
        modified_messages = []
        tool_results_modified = 0
        
        for msg in messages:
            if msg.role == "user" and msg.content and isinstance(msg.content, list):
                modified_content_blocks = []
                has_modifications = False
                
                for block in msg.content:
                    block_type = self._get_block_value(block, "type")
                    block_tool_use_id = self._get_block_value(block, "tool_use_id")
                    
                    if block_type == "tool_result" and block_tool_use_id and should_inject_recovery():
                        truncation_info = get_tool_truncation(block_tool_use_id)
                        if truncation_info:
                            tool_results_modified += 1
                            modified_content_blocks.append(block)
                        else:
                            modified_content_blocks.append(block)
                    else:
                        modified_content_blocks.append(block)
                
                if has_modifications:
                    modified_msg = msg.model_copy(update={"content": modified_content_blocks})
                    modified_messages.append(modified_msg)
                    continue
            
            modified_messages.append(msg)
        
        print(f"Checking: tool_results_modified count...")
        assert tool_results_modified == 0
        
        print("Checking: Message content unchanged...")
        content = self._get_block_value(modified_messages[0].content[0], "content")
        assert content == "Success"
    
    def test_pydantic_immutability_new_object_created(self):
        """
        What it does: Verifies new AnthropicMessage object is created, not modified in-place.
        Purpose: Ensure Pydantic immutability is respected.
        """
        print("Setup: Saving truncation info and creating message...")
        from kiro.truncation_state import save_tool_truncation
        from kiro.models_anthropic import AnthropicMessage
        
        tool_use_id = "test_immutable_anthropic"
        save_tool_truncation(tool_use_id, "tool", {"size_bytes": 1000, "reason": "test truncation"})
        
        original_msg = AnthropicMessage(
            role="user",
            content=[
                {"type": "tool_result", "tool_use_id": tool_use_id, "content": "original"}
            ]
        )
        original_content = self._get_block_value(original_msg.content[0], "content")
        
        print("Action: Processing message...")
        from kiro.truncation_recovery import should_inject_recovery, generate_truncation_tool_result
        from kiro.truncation_state import get_tool_truncation
        
        if original_msg.role == "user" and original_msg.content and isinstance(original_msg.content, list):
            modified_content_blocks = []
            has_modifications = False
            
            for block in original_msg.content:
                block_type = self._get_block_value(block, "type")
                block_tool_use_id = self._get_block_value(block, "tool_use_id")
                original_block_content = self._get_block_value(block, "content", "")
                
                if block_type == "tool_result" and block_tool_use_id and should_inject_recovery():
                    truncation_info = get_tool_truncation(block_tool_use_id)
                    if truncation_info:
                        synthetic = generate_truncation_tool_result(
                            truncation_info.tool_name,
                            truncation_info.tool_call_id,
                            truncation_info.truncation_info
                        )
                        modified_content = f"{synthetic['content']}\n\n---\n\nOriginal tool result:\n{original_block_content}"
                        
                        if isinstance(block, dict):
                            modified_block = block.copy()
                            modified_block["content"] = modified_content
                        else:
                            modified_block = block.model_copy(update={"content": modified_content})
                        
                        modified_content_blocks.append(modified_block)
                        has_modifications = True
                        continue
                
                modified_content_blocks.append(block)
            
            if has_modifications:
                modified_msg = original_msg.model_copy(update={"content": modified_content_blocks})
        
        print("Checking: Original message unchanged...")
        assert self._get_block_value(original_msg.content[0], "content") == original_content
        
        print("Checking: New object created...")
        assert modified_msg is not original_msg
        
        print("Checking: Content modified in new object...")
        modified_content = self._get_block_value(modified_msg.content[0], "content")
        assert modified_content != original_content
        assert "[API Limitation]" in modified_content


# =============================================================================
# Tests for Content Truncation Recovery (Issue #56)
# =============================================================================

class TestContentTruncationRecovery:
    """
    Tests for content truncation recovery (synthetic user message) in Anthropic routes.
    
    Verifies that synthetic user message is added after truncated assistant message.
    Part of Truncation Recovery System (Issue #56).
    """
    
    @staticmethod
    def _get_block_value(block, key, default=""):
        """Helper to get value from dict or Pydantic object."""
        if isinstance(block, dict):
            return block.get(key, default)
        else:
            return getattr(block, key, default)
    
    def test_adds_synthetic_user_message_after_truncated_assistant(self):
        """
        What it does: Verifies synthetic user message is added after truncated assistant message.
        Purpose: Ensure content truncation recovery works for Anthropic API (Test Case C.2).
        """
        print("Setup: Saving content truncation info...")
        from kiro.truncation_state import save_content_truncation
        from kiro.models_anthropic import AnthropicMessage
        
        # For Anthropic, content can be string or list of blocks
        truncated_content_text = "This is a very long response that was cut off mid-sentence"
        save_content_truncation(truncated_content_text)
        
        print("Setup: Creating request with truncated assistant message...")
        messages = [
            AnthropicMessage(role="assistant", content=[{"type": "text", "text": truncated_content_text}])
        ]
        
        print("Action: Processing messages through content truncation recovery...")
        from kiro.truncation_recovery import should_inject_recovery, generate_truncation_user_message
        from kiro.truncation_state import get_content_truncation
        
        modified_messages = []
        for msg in messages:
            if msg.role == "assistant" and msg.content:
                # Extract text content for hash check
                text_content = ""
                if isinstance(msg.content, str):
                    text_content = msg.content
                elif isinstance(msg.content, list):
                    for block in msg.content:
                        if self._get_block_value(block, "type") == "text":
                            text_content += self._get_block_value(block, "text", "")
                
                if text_content:
                    truncation_info = get_content_truncation(text_content)
                    if truncation_info:
                        print(f"Found content truncation for hash: {truncation_info.message_hash}")
                        # Add original message first
                        modified_messages.append(msg)
                        # Then add synthetic user message
                        synthetic_user_msg = AnthropicMessage(
                            role="user",
                            content=[{"type": "text", "text": generate_truncation_user_message()}]
                        )
                        modified_messages.append(synthetic_user_msg)
                        continue
            modified_messages.append(msg)
        
        print("Checking: Two messages in result...")
        assert len(modified_messages) == 2
        
        print("Checking: First message is original assistant message...")
        assert modified_messages[0].role == "assistant"
        
        print("Checking: Second message is synthetic user message...")
        assert modified_messages[1].role == "user"
        synthetic_text = self._get_block_value(modified_messages[1].content[0], "text")
        assert "[System Notice]" in synthetic_text
        assert "truncated" in synthetic_text.lower()
    
    def test_no_synthetic_message_when_no_content_truncation(self):
        """
        What it does: Verifies no synthetic message is added for normal assistant message.
        Purpose: Ensure false positives don't occur.
        """
        print("Setup: Creating normal assistant message (no truncation)...")
        from kiro.models_anthropic import AnthropicMessage
        
        messages = [
            AnthropicMessage(role="assistant", content=[{"type": "text", "text": "This is a complete response."}])
        ]
        
        print("Action: Processing messages...")
        from kiro.truncation_state import get_content_truncation
        
        modified_messages = []
        for msg in messages:
            if msg.role == "assistant" and msg.content:
                text_content = ""
                if isinstance(msg.content, str):
                    text_content = msg.content
                elif isinstance(msg.content, list):
                    for block in msg.content:
                        if self._get_block_value(block, "type") == "text":
                            text_content += self._get_block_value(block, "text", "")
                
                if text_content:
                    truncation_info = get_content_truncation(text_content)
                    if truncation_info:
                        # Would add synthetic message here
                        pass
            modified_messages.append(msg)
        
        print("Checking: Only one message in result...")
        assert len(modified_messages) == 1
        
        print("Checking: Message unchanged...")
        text = self._get_block_value(modified_messages[0].content[0], "text")
        assert text == "This is a complete response."


# =============================================================================
# Tests for error response handling when Kiro API returns non-200
# =============================================================================

class TestMessagesErrorResponseHandling:
    """Tests for error response handling when Kiro API returns non-200."""

    @staticmethod
    def _build_mock_client(mock_class, status_code, aread_value=None, aread_side_effect=None):
        """Helper to build a mock KiroHttpClient and response."""
        mock_response = MagicMock()
        mock_response.status_code = status_code
        if aread_side_effect:
            mock_response.aread = AsyncMock(side_effect=aread_side_effect)
        else:
            mock_response.aread = AsyncMock(return_value=aread_value or b"")

        mock_instance = AsyncMock()
        mock_instance.request_with_retry = AsyncMock(return_value=mock_response)
        mock_instance.close = AsyncMock()
        mock_class.return_value = mock_instance
        return mock_instance, mock_response

    @patch('kiro.routes_anthropic.anthropic_to_kiro', return_value={"messages": [], "conversationId": "test"})
    @patch('kiro.routes_anthropic.KiroHttpClient')
    def test_returns_anthropic_error_on_kiro_400(
        self, mock_client_class, mock_converter, test_client, valid_proxy_api_key
    ):
        """
        What it does: Verifies a 400 from Kiro API is returned in Anthropic error format.
        Purpose: Ensure upstream errors are properly translated.
        """
        print("Setup: Mock KiroHttpClient returning 400...")
        self._build_mock_client(
            mock_client_class,
            status_code=400,
            aread_value=b'{"message": "Input is too long.", "reason": "CONTENT_LENGTH_EXCEEDS_THRESHOLD"}'
        )

        print("Action: POST /v1/messages...")
        response = test_client.post(
            "/v1/messages",
            headers={"x-api-key": valid_proxy_api_key},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )

        print(f"Checking: status={response.status_code}")
        assert response.status_code == 400
        data = response.json()
        assert data["type"] == "error"
        assert data["error"]["type"] == "api_error"

    @patch('kiro.routes_anthropic.anthropic_to_kiro', return_value={"messages": [], "conversationId": "test"})
    @patch('kiro.routes_anthropic.KiroHttpClient')
    def test_returns_anthropic_error_on_kiro_500_non_json(
        self, mock_client_class, mock_converter, test_client, valid_proxy_api_key
    ):
        """
        What it does: Verifies a 500 with non-JSON body is returned in Anthropic error format.
        Purpose: Ensure non-JSON upstream errors are handled gracefully.
        """
        print("Setup: Mock KiroHttpClient returning 500 with plain text...")
        self._build_mock_client(
            mock_client_class,
            status_code=500,
            aread_value=b"Internal Server Error"
        )

        print("Action: POST /v1/messages...")
        response = test_client.post(
            "/v1/messages",
            headers={"x-api-key": valid_proxy_api_key},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )

        print(f"Checking: status={response.status_code}")
        assert response.status_code == 500
        data = response.json()
        assert data["type"] == "error"
        assert "Internal Server Error" in data["error"]["message"]

    @patch('kiro.routes_anthropic.anthropic_to_kiro', return_value={"messages": [], "conversationId": "test"})
    @patch('kiro.routes_anthropic.KiroHttpClient')
    def test_returns_anthropic_error_on_unreadable_body(
        self, mock_client_class, mock_converter, test_client, valid_proxy_api_key
    ):
        """
        What it does: Verifies an unreadable response body is handled gracefully.
        Purpose: Ensure connection errors during body read don't crash the endpoint.
        """
        print("Setup: Mock KiroHttpClient returning 502 with aread raising RuntimeError...")
        self._build_mock_client(
            mock_client_class,
            status_code=502,
            aread_side_effect=RuntimeError("connection reset")
        )

        print("Action: POST /v1/messages...")
        response = test_client.post(
            "/v1/messages",
            headers={"x-api-key": valid_proxy_api_key},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )

        print(f"Checking: status={response.status_code}")
        assert response.status_code != 200
        data = response.json()
        assert data["type"] == "error"
        assert "Unknown error" in data["error"]["message"]

    @patch('kiro.routes_anthropic.debug_logger')
    @patch('kiro.routes_anthropic.anthropic_to_kiro', return_value={"messages": [], "conversationId": "test"})
    @patch('kiro.routes_anthropic.KiroHttpClient')
    def test_error_response_flushes_debug_logger(
        self, mock_client_class, mock_converter, mock_debug_logger, test_client, valid_proxy_api_key
    ):
        """
        What it does: Verifies debug_logger.flush_on_error is called on non-200 response.
        Purpose: Ensure debug logs are flushed for error diagnosis.
        """
        print("Setup: Mock KiroHttpClient returning 400...")
        self._build_mock_client(
            mock_client_class,
            status_code=400,
            aread_value=b'{"message": "Bad request", "reason": "INVALID"}'
        )

        print("Action: POST /v1/messages...")
        test_client.post(
            "/v1/messages",
            headers={"x-api-key": valid_proxy_api_key},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )

        print("Checking: debug_logger.flush_on_error was called...")
        mock_debug_logger.flush_on_error.assert_called()


# =============================================================================
# Tests for streaming error handling in /v1/messages endpoint
# =============================================================================

class TestMessagesStreamingErrorHandling:
    """Tests for streaming error handling in /v1/messages endpoint."""

    @staticmethod
    def _build_mock_client(mock_class, status_code=200):
        """Helper to build a mock KiroHttpClient returning given status."""
        mock_response = MagicMock()
        mock_response.status_code = status_code
        mock_response.aread = AsyncMock(return_value=b"")

        mock_instance = AsyncMock()
        mock_instance.request_with_retry = AsyncMock(return_value=mock_response)
        mock_instance.close = AsyncMock()
        mock_class.return_value = mock_instance
        return mock_instance, mock_response

    @patch('kiro.routes_anthropic.anthropic_to_kiro', return_value={"messages": [], "conversationId": "test"})
    @patch('kiro.routes_anthropic.stream_kiro_to_anthropic')
    @patch('kiro.routes_anthropic.KiroHttpClient')
    def test_streaming_exception_sends_error_event(
        self, mock_client_class, mock_stream_fn, mock_converter, test_client, valid_proxy_api_key
    ):
        """
        What it does: Verifies a streaming exception sends an SSE error event.
        Purpose: Ensure clients receive error notification during streaming.
        """
        print("Setup: Mock stream that raises after one chunk...")

        async def failing_stream(*args, **kwargs):
            yield 'event: message_start\ndata: {"type":"message_start"}\n\n'
            raise RuntimeError("stream broke")

        mock_stream_fn.side_effect = failing_stream
        self._build_mock_client(mock_client_class, status_code=200)

        print("Action: POST /v1/messages with stream=true...")
        response = test_client.post(
            "/v1/messages",
            headers={"x-api-key": valid_proxy_api_key},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True
            }
        )

        print("Checking: response body contains error event...")
        body = response.text
        assert "event: error" in body

    @patch('kiro.routes_anthropic.anthropic_to_kiro', return_value={"messages": [], "conversationId": "test"})
    @patch('kiro.routes_anthropic.stream_kiro_to_anthropic')
    @patch('kiro.routes_anthropic.KiroHttpClient')
    def test_streaming_finally_closes_http_client(
        self, mock_client_class, mock_stream_fn, mock_converter, test_client, valid_proxy_api_key
    ):
        """
        What it does: Verifies HTTP client is closed after streaming completes (even on error).
        Purpose: Prevent resource leaks on streaming failures.
        """
        print("Setup: Mock stream that raises after one chunk...")

        async def failing_stream(*args, **kwargs):
            yield 'event: message_start\ndata: {"type":"message_start"}\n\n'
            raise RuntimeError("stream broke")

        mock_stream_fn.side_effect = failing_stream
        mock_instance, _ = self._build_mock_client(mock_client_class, status_code=200)

        print("Action: POST /v1/messages with stream=true...")
        test_client.post(
            "/v1/messages",
            headers={"x-api-key": valid_proxy_api_key},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True
            }
        )

        print("Checking: http_client.close() was called...")
        mock_instance.close.assert_awaited()

    @patch('kiro.routes_anthropic.debug_logger')
    @patch('kiro.routes_anthropic.anthropic_to_kiro', return_value={"messages": [], "conversationId": "test"})
    @patch('kiro.routes_anthropic.stream_kiro_to_anthropic')
    @patch('kiro.routes_anthropic.KiroHttpClient')
    def test_streaming_error_flushes_debug_logger(
        self, mock_client_class, mock_stream_fn, mock_converter, mock_debug_logger,
        test_client, valid_proxy_api_key
    ):
        """
        What it does: Verifies debug_logger.flush_on_error is called on streaming error.
        Purpose: Ensure debug logs are flushed for streaming error diagnosis.
        """
        print("Setup: Mock stream that raises after one chunk...")

        async def failing_stream(*args, **kwargs):
            yield 'event: message_start\ndata: {"type":"message_start"}\n\n'
            raise RuntimeError("stream broke")

        mock_stream_fn.side_effect = failing_stream
        self._build_mock_client(mock_client_class, status_code=200)

        print("Action: POST /v1/messages with stream=true...")
        test_client.post(
            "/v1/messages",
            headers={"x-api-key": valid_proxy_api_key},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True
            }
        )

        print("Checking: debug_logger.flush_on_error was called with 500...")
        mock_debug_logger.flush_on_error.assert_called_with(500, "stream broke")

    @patch('kiro.routes_anthropic.debug_logger')
    @patch('kiro.routes_anthropic.anthropic_to_kiro', return_value={"messages": [], "conversationId": "test"})
    @patch('kiro.routes_anthropic.stream_kiro_to_anthropic')
    @patch('kiro.routes_anthropic.KiroHttpClient')
    def test_streaming_success_discards_debug_buffers(
        self, mock_client_class, mock_stream_fn, mock_converter, mock_debug_logger,
        test_client, valid_proxy_api_key
    ):
        """
        What it does: Verifies debug_logger.discard_buffers is called on successful stream.
        Purpose: Ensure debug buffers are cleaned up after successful requests.
        """
        print("Setup: Mock stream that yields valid events...")

        async def ok_stream(*args, **kwargs):
            yield 'event: message_start\ndata: {"type":"message_start"}\n\n'
            yield 'event: message_stop\ndata: {"type":"message_stop"}\n\n'

        mock_stream_fn.side_effect = ok_stream
        self._build_mock_client(mock_client_class, status_code=200)

        print("Action: POST /v1/messages with stream=true...")
        test_client.post(
            "/v1/messages",
            headers={"x-api-key": valid_proxy_api_key},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True
            }
        )

        print("Checking: debug_logger.discard_buffers was called...")
        mock_debug_logger.discard_buffers.assert_called()


# =============================================================================
# Tests for non-streaming response collection path
# =============================================================================

class TestMessagesNonStreamingPath:
    """Tests for non-streaming response collection path."""

    @staticmethod
    def _build_mock_client(mock_class, status_code=200):
        """Helper to build a mock KiroHttpClient returning given status."""
        mock_response = MagicMock()
        mock_response.status_code = status_code
        mock_response.aread = AsyncMock(return_value=b"")

        mock_instance = AsyncMock()
        mock_instance.request_with_retry = AsyncMock(return_value=mock_response)
        mock_instance.close = AsyncMock()
        mock_class.return_value = mock_instance
        return mock_instance, mock_response

    @patch('kiro.routes_anthropic.collect_anthropic_response')
    @patch('kiro.routes_anthropic.anthropic_to_kiro', return_value={"messages": [], "conversationId": "test"})
    @patch('kiro.routes_anthropic.KiroHttpClient')
    def test_non_streaming_collects_response(
        self, mock_client_class, mock_converter, mock_collect, test_client, valid_proxy_api_key
    ):
        """
        What it does: Verifies non-streaming path collects and returns the full response.
        Purpose: Ensure non-streaming responses are properly assembled.
        """
        print("Setup: Mock collect_anthropic_response returning valid response...")
        expected_response = {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello!"}],
            "model": "claude-sonnet-4-5",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }
        mock_collect.return_value = expected_response
        self._build_mock_client(mock_client_class, status_code=200)

        print("Action: POST /v1/messages (non-streaming)...")
        response = test_client.post(
            "/v1/messages",
            headers={"x-api-key": valid_proxy_api_key},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )

        print(f"Checking: status={response.status_code}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "msg_test"
        assert data["content"][0]["text"] == "Hello!"

    @patch('kiro.routes_anthropic.collect_anthropic_response', return_value={"id": "msg_test", "type": "message"})
    @patch('kiro.routes_anthropic.anthropic_to_kiro', return_value={"messages": [], "conversationId": "test"})
    @patch('kiro.routes_anthropic.KiroHttpClient')
    def test_non_streaming_closes_http_client(
        self, mock_client_class, mock_converter, mock_collect, test_client, valid_proxy_api_key
    ):
        """
        What it does: Verifies HTTP client is closed after non-streaming response.
        Purpose: Prevent resource leaks on non-streaming path.
        """
        print("Setup: Mock KiroHttpClient returning 200...")
        mock_instance, _ = self._build_mock_client(mock_client_class, status_code=200)

        print("Action: POST /v1/messages (non-streaming)...")
        test_client.post(
            "/v1/messages",
            headers={"x-api-key": valid_proxy_api_key},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )

        print("Checking: http_client.close() was called...")
        mock_instance.close.assert_awaited()

    @patch('kiro.routes_anthropic.debug_logger')
    @patch('kiro.routes_anthropic.collect_anthropic_response', return_value={"id": "msg_test", "type": "message"})
    @patch('kiro.routes_anthropic.anthropic_to_kiro', return_value={"messages": [], "conversationId": "test"})
    @patch('kiro.routes_anthropic.KiroHttpClient')
    def test_non_streaming_discards_debug_buffers(
        self, mock_client_class, mock_converter, mock_collect, mock_debug_logger,
        test_client, valid_proxy_api_key
    ):
        """
        What it does: Verifies debug_logger.discard_buffers is called on successful non-streaming response.
        Purpose: Ensure debug buffers are cleaned up after successful requests.
        """
        print("Setup: Mock KiroHttpClient returning 200...")
        self._build_mock_client(mock_client_class, status_code=200)

        print("Action: POST /v1/messages (non-streaming)...")
        test_client.post(
            "/v1/messages",
            headers={"x-api-key": valid_proxy_api_key},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )

        print("Checking: debug_logger.discard_buffers was called...")
        mock_debug_logger.discard_buffers.assert_called()


# =============================================================================
# Tests for conversion error handling
# =============================================================================

class TestMessagesConversionError:
    """Tests for conversion error handling."""

    @patch('kiro.routes_anthropic.anthropic_to_kiro', side_effect=ValueError("No messages provided"))
    def test_conversion_value_error_returns_400(
        self, mock_converter, test_client, valid_proxy_api_key
    ):
        """
        What it does: Verifies a ValueError during conversion returns 400 with invalid_request_error.
        Purpose: Ensure conversion validation errors are surfaced to the client.
        """
        print("Setup: Mock anthropic_to_kiro raising ValueError...")

        print("Action: POST /v1/messages...")
        response = test_client.post(
            "/v1/messages",
            headers={"x-api-key": valid_proxy_api_key},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )

        print(f"Checking: status={response.status_code}")
        assert response.status_code == 400
        data = response.json()
        assert data["type"] == "error"
        assert data["error"]["type"] == "invalid_request_error"


# =============================================================================
# Tests for general exception handling in /v1/messages
# =============================================================================

class TestMessagesGeneralExceptionHandling:
    """Tests for general exception handling in /v1/messages."""

    @patch('kiro.routes_anthropic.anthropic_to_kiro', return_value={"messages": [], "conversationId": "test"})
    @patch('kiro.routes_anthropic.KiroHttpClient')
    def test_unexpected_exception_returns_500(
        self, mock_client_class, mock_converter, test_client, valid_proxy_api_key
    ):
        """
        What it does: Verifies an unexpected exception returns 500 with api_error.
        Purpose: Ensure unhandled errors are caught and returned in Anthropic format.
        """
        print("Setup: Mock KiroHttpClient.request_with_retry raising RuntimeError...")
        mock_instance = AsyncMock()
        mock_instance.request_with_retry = AsyncMock(side_effect=RuntimeError("unexpected"))
        mock_instance.close = AsyncMock()
        mock_client_class.return_value = mock_instance

        print("Action: POST /v1/messages...")
        response = test_client.post(
            "/v1/messages",
            headers={"x-api-key": valid_proxy_api_key},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )

        print(f"Checking: status={response.status_code}")
        assert response.status_code == 500
        data = response.json()
        assert data["type"] == "error"
        assert data["error"]["type"] == "api_error"
        assert "Internal Server Error" in data["error"]["message"]

    @patch('kiro.routes_anthropic.debug_logger')
    @patch('kiro.routes_anthropic.anthropic_to_kiro', return_value={"messages": [], "conversationId": "test"})
    @patch('kiro.routes_anthropic.KiroHttpClient')
    def test_unexpected_exception_flushes_debug_logger(
        self, mock_client_class, mock_converter, mock_debug_logger, test_client, valid_proxy_api_key
    ):
        """
        What it does: Verifies debug_logger.flush_on_error is called on unexpected exception.
        Purpose: Ensure debug logs are flushed for unhandled error diagnosis.
        """
        print("Setup: Mock KiroHttpClient.request_with_retry raising RuntimeError...")
        mock_instance = AsyncMock()
        mock_instance.request_with_retry = AsyncMock(side_effect=RuntimeError("unexpected"))
        mock_instance.close = AsyncMock()
        mock_client_class.return_value = mock_instance

        print("Action: POST /v1/messages...")
        test_client.post(
            "/v1/messages",
            headers={"x-api-key": valid_proxy_api_key},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )

        print("Checking: debug_logger.flush_on_error was called with 500...")
        mock_debug_logger.flush_on_error.assert_called_with(500, "unexpected")


# =============================================================================
# Tests for truncation recovery through the actual endpoint
# =============================================================================

class TestMessagesEndpointTruncationRecovery:
    """
    Tests for truncation recovery code paths exercised through the /v1/messages endpoint.

    Covers lines 172-248 of routes_anthropic.py: dict/Pydantic block handling,
    tool_result modification, content truncation synthetic message injection.
    """

    @patch('kiro.routes_anthropic.anthropic_to_kiro', return_value={"messages": [], "conversationId": "test"})
    @patch('kiro.routes_anthropic.KiroHttpClient')
    def test_endpoint_modifies_tool_result_with_truncation(
        self, mock_client_class, mock_converter, test_client, valid_proxy_api_key
    ):
        """
        What it does: Verifies tool_result blocks are modified when truncation info exists in cache.
        Purpose: Exercise the truncation recovery code path through the actual endpoint (lines 172-208).
        """
        print("Setup: Saving truncation info to cache...")
        from kiro.truncation_state import save_tool_truncation
        tool_use_id = "tooluse_endpoint_test_1"
        save_tool_truncation(tool_use_id, "write_to_file", {"size_bytes": 5000, "reason": "test"})

        print("Setup: Mock KiroHttpClient returning 200...")
        mock_instance = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_instance.request_with_retry = AsyncMock(return_value=mock_response)
        mock_instance.close = AsyncMock()
        mock_client_class.return_value = mock_instance

        print("Action: POST /v1/messages with tool_result that has truncation info...")
        test_client.post(
            "/v1/messages",
            headers={"x-api-key": valid_proxy_api_key},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "messages": [
                    {"role": "user", "content": "What's the weather?"},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "tool_use", "id": tool_use_id, "name": "write_to_file", "input": {}}
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": tool_use_id, "content": "original result"}
                        ]
                    }
                ]
            }
        )

        print("Checking: anthropic_to_kiro was called with modified messages...")
        assert mock_converter.called
        call_args = mock_converter.call_args[0]
        request_data = call_args[0]
        # Find the user message with tool_result
        tool_result_msg = [m for m in request_data.messages if m.role == "user" and m.content and isinstance(m.content, list)][-1]
        tool_result_block = tool_result_msg.content[0]
        content = tool_result_block.get("content") if isinstance(tool_result_block, dict) else getattr(tool_result_block, "content", "")
        print(f"Content: {str(content)[:100]}...")
        assert "[API Limitation]" in str(content)
        assert "original result" in str(content)

    @patch('kiro.routes_anthropic.anthropic_to_kiro', return_value={"messages": [], "conversationId": "test"})
    @patch('kiro.routes_anthropic.KiroHttpClient')
    def test_endpoint_adds_synthetic_message_for_content_truncation(
        self, mock_client_class, mock_converter, test_client, valid_proxy_api_key
    ):
        """
        What it does: Verifies synthetic user message is added after truncated assistant message.
        Purpose: Exercise content truncation recovery through the endpoint (lines 218-248).
        """
        print("Setup: Saving content truncation info...")
        from kiro.truncation_state import save_content_truncation
        truncated_text = "This response was cut off mid-sentence by the API and never finished"
        save_content_truncation(truncated_text)

        print("Setup: Mock KiroHttpClient returning 200...")
        mock_instance = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_instance.request_with_retry = AsyncMock(return_value=mock_response)
        mock_instance.close = AsyncMock()
        mock_client_class.return_value = mock_instance

        print("Action: POST /v1/messages with truncated assistant message (string content)...")
        test_client.post(
            "/v1/messages",
            headers={"x-api-key": valid_proxy_api_key},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "messages": [
                    {"role": "user", "content": "Tell me a story"},
                    {
                        "role": "assistant",
                        "content": truncated_text
                    },
                    {"role": "user", "content": "Continue"}
                ]
            }
        )

        print("Checking: anthropic_to_kiro was called with extra synthetic message...")
        assert mock_converter.called
        call_args = mock_converter.call_args[0]
        request_data = call_args[0]
        messages = request_data.messages
        # Should have 4 messages: user, assistant, synthetic_user, user
        print(f"Message count: {len(messages)}")
        assert len(messages) == 4
        assert messages[2].role == "user"
        synthetic_content = messages[2].content
        if isinstance(synthetic_content, list):
            text = synthetic_content[0].get("text", "") if isinstance(synthetic_content[0], dict) else getattr(synthetic_content[0], "text", "")
        else:
            text = str(synthetic_content)
        print(f"Synthetic message: {text[:80]}...")
        assert "[System Notice]" in text


# =============================================================================
# Tests for profile ARN and HTTPException handling
# =============================================================================

class TestMessagesProfileArnAndHTTPException:
    """Tests for profile ARN handling and HTTPException re-raise."""

    @patch('kiro.routes_anthropic.anthropic_to_kiro', return_value={"messages": [], "conversationId": "test"})
    @patch('kiro.routes_anthropic.KiroHttpClient')
    def test_profile_arn_set_for_kiro_desktop_auth(
        self, mock_client_class, mock_converter, test_client, valid_proxy_api_key
    ):
        """
        What it does: Verifies profile_arn is passed when auth_type is KIRO_DESKTOP.
        Purpose: Exercise profile ARN code path (line 257).
        """
        print("Setup: Set auth_manager to KIRO_DESKTOP with profile_arn...")
        from kiro.auth import AuthType
        app = test_client.app
        mock_auth = MagicMock()
        mock_auth.auth_type = AuthType.KIRO_DESKTOP
        mock_auth.profile_arn = "arn:aws:iam::123456:role/test"
        mock_auth.api_host = "https://q.us-east-1.amazonaws.com"
        app.state.auth_manager = mock_auth

        mock_instance = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_instance.request_with_retry = AsyncMock(return_value=mock_response)
        mock_instance.close = AsyncMock()
        mock_client_class.return_value = mock_instance

        print("Action: POST /v1/messages...")
        test_client.post(
            "/v1/messages",
            headers={"x-api-key": valid_proxy_api_key},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )

        print("Checking: anthropic_to_kiro called with profile_arn...")
        assert mock_converter.called
        call_args = mock_converter.call_args[0]
        profile_arn = call_args[2]
        print(f"profile_arn: {profile_arn}")
        assert profile_arn == "arn:aws:iam::123456:role/test"

    @patch('kiro.routes_anthropic.anthropic_to_kiro', return_value={"messages": [], "conversationId": "test"})
    @patch('kiro.routes_anthropic.KiroHttpClient')
    def test_http_exception_reraises_and_closes_client(
        self, mock_client_class, mock_converter, test_client, valid_proxy_api_key
    ):
        """
        What it does: Verifies HTTPException is re-raised and client is closed.
        Purpose: Exercise HTTPException handling (lines 430-435).
        """
        print("Setup: Mock KiroHttpClient that raises HTTPException...")
        mock_instance = AsyncMock()
        mock_instance.request_with_retry = AsyncMock(
            side_effect=HTTPException(status_code=503, detail="Service Unavailable")
        )
        mock_instance.close = AsyncMock()
        mock_client_class.return_value = mock_instance

        print("Action: POST /v1/messages...")
        response = test_client.post(
            "/v1/messages",
            headers={"x-api-key": valid_proxy_api_key},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )

        print(f"Status: {response.status_code}")
        assert response.status_code == 503
        print("Checking: http_client.close() was called...")
        mock_instance.close.assert_called()

    @patch('kiro.routes_anthropic.debug_logger', new_callable=MagicMock)
    @patch('kiro.routes_anthropic.anthropic_to_kiro', return_value={"messages": [], "conversationId": "test"})
    @patch('kiro.routes_anthropic.KiroHttpClient')
    def test_http_exception_flushes_debug_logger(
        self, mock_client_class, mock_converter, mock_debug_logger, test_client, valid_proxy_api_key
    ):
        """
        What it does: Verifies debug_logger is flushed on HTTPException.
        Purpose: Exercise debug logger flush in HTTPException handler (lines 433-434).
        """
        print("Setup: Mock KiroHttpClient that raises HTTPException...")
        mock_instance = AsyncMock()
        mock_instance.request_with_retry = AsyncMock(
            side_effect=HTTPException(status_code=503, detail="Service Unavailable")
        )
        mock_instance.close = AsyncMock()
        mock_client_class.return_value = mock_instance

        print("Action: POST /v1/messages...")
        test_client.post(
            "/v1/messages",
            headers={"x-api-key": valid_proxy_api_key},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )

        print("Checking: debug_logger.flush_on_error was called...")
        mock_debug_logger.flush_on_error.assert_called()


# =============================================================================
# Tests for streaming GeneratorExit and debug logger kiro request logging
# =============================================================================

class TestMessagesStreamingDisconnectAndLogging:
    """Tests for client disconnect and kiro request body logging."""

    @patch('kiro.routes_anthropic.anthropic_to_kiro', return_value={"messages": [], "conversationId": "test"})
    @patch('kiro.routes_anthropic.KiroHttpClient')
    def test_kiro_request_body_logging_error_handled(
        self, mock_client_class, mock_converter, test_client, valid_proxy_api_key
    ):
        """
        What it does: Verifies exception during kiro request body logging is caught.
        Purpose: Exercise the except branch in request body logging (lines 283-284).
        """
        print("Setup: Mock anthropic_to_kiro returning non-serializable data...")
        mock_converter.return_value = {"data": object()}  # Not JSON serializable

        mock_instance = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_instance.request_with_retry = AsyncMock(return_value=mock_response)
        mock_instance.close = AsyncMock()
        mock_client_class.return_value = mock_instance

        print("Action: POST /v1/messages with stream=false...")
        with patch('kiro.routes_anthropic.collect_anthropic_response', new_callable=AsyncMock) as mock_collect:
            mock_collect.return_value = {
                "id": "msg_test", "type": "message", "role": "assistant",
                "content": [{"type": "text", "text": "Hi"}],
                "model": "claude-sonnet-4-5", "stop_reason": "end_turn",
                "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 5}
            }
            response = test_client.post(
                "/v1/messages",
                headers={"x-api-key": valid_proxy_api_key},
                json={
                    "model": "claude-sonnet-4-5",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": "Hello"}]
                }
            )

        print(f"Status: {response.status_code}")
        # Should still succeed despite logging error
        assert response.status_code == 200