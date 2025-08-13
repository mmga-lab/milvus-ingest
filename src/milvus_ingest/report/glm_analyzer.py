"""GLM API client for Milvus import performance analysis."""

import json
import time
from typing import Any

import requests

from ..logging_config import get_logger


class GLMAnalyzer:
    """GLM API client for analyzing Milvus import performance data."""

    def __init__(self, api_key: str, model: str = "glm-4.5-flash", timeout: int = 60):
        """Initialize GLM analyzer.

        Args:
            api_key: GLM API key
            model: GLM model to use (default: glm-4-flash)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.api_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        self.logger = get_logger(__name__)

    def analyze(
        self, data: dict[str, Any], prompt_template: str, max_retries: int = 3
    ) -> str:
        """Analyze data using GLM API.

        Args:
            data: Summarized data for analysis
            prompt_template: Prompt template with {data_json} placeholder
            max_retries: Maximum number of retry attempts

        Returns:
            Analysis report as markdown string

        Raises:
            Exception: If API call fails after all retries
        """
        # Format the prompt with data
        data_json = json.dumps(data, indent=2, ensure_ascii=False, default=str)
        prompt = prompt_template.format(data_json=data_json)

        # Prepare the API request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a Milvus database performance analyst. Provide detailed, actionable analysis in clean markdown format.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }

        # Log the request details (without API key)
        self.logger.info(f"Making GLM API request with model: {self.model}")
        self.logger.debug(f"Data summary size: {len(data_json)} characters")

        # Retry logic
        last_exception = None
        for attempt in range(max_retries):
            try:
                self.logger.info(f"GLM API attempt {attempt + 1}/{max_retries}")

                response = requests.post(
                    self.api_url, headers=headers, json=payload, timeout=self.timeout
                )

                if response.status_code == 200:
                    result = response.json()

                    # Extract the analysis content
                    if "choices" in result and len(result["choices"]) > 0:
                        content = result["choices"][0]["message"]["content"]

                        # Log usage statistics
                        usage = result.get("usage", {})
                        self.logger.info(
                            "GLM API call successful",
                            tokens_used=usage.get("total_tokens", 0),
                            prompt_tokens=usage.get("prompt_tokens", 0),
                            completion_tokens=usage.get("completion_tokens", 0),
                        )

                        return content
                    else:
                        raise Exception(f"Unexpected API response structure: {result}")

                elif response.status_code == 429:
                    # Rate limit exceeded - wait and retry
                    wait_time = min(2**attempt, 30)  # Exponential backoff, max 30s
                    self.logger.warning(
                        f"Rate limit exceeded, waiting {wait_time}s before retry"
                    )
                    time.sleep(wait_time)
                    continue

                elif response.status_code == 401:
                    raise Exception("Authentication failed - check API key")

                else:
                    error_msg = f"API request failed with status {response.status_code}: {response.text}"
                    self.logger.warning(error_msg)
                    last_exception = Exception(error_msg)

                    if attempt < max_retries - 1:
                        # Wait before retry for server errors
                        wait_time = min(2**attempt, 10)
                        self.logger.info(f"Retrying in {wait_time}s...")
                        time.sleep(wait_time)

            except requests.exceptions.Timeout:
                error_msg = f"Request timed out after {self.timeout}s"
                self.logger.warning(error_msg)
                last_exception = Exception(error_msg)

                if attempt < max_retries - 1:
                    self.logger.info("Retrying with increased timeout...")

            except requests.exceptions.ConnectionError as e:
                error_msg = f"Connection error: {e}"
                self.logger.warning(error_msg)
                last_exception = Exception(error_msg)

                if attempt < max_retries - 1:
                    wait_time = min(2**attempt, 15)
                    self.logger.info(f"Retrying connection in {wait_time}s...")
                    time.sleep(wait_time)

            except Exception as e:
                error_msg = f"Unexpected error: {e}"
                self.logger.error(error_msg)
                last_exception = e

                if attempt < max_retries - 1:
                    self.logger.info("Retrying after unexpected error...")
                    time.sleep(1)

        # All retries failed
        self.logger.error(f"GLM API call failed after {max_retries} attempts")
        if last_exception:
            raise last_exception
        else:
            raise Exception("GLM API call failed with unknown error")

    def test_connection(self) -> bool:
        """Test the GLM API connection with a simple request.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            test_data = {"test": "Simple connectivity test"}
            test_prompt = "Respond with 'GLM API connection successful' if you receive this message."

            result = self.analyze(test_data, test_prompt, max_retries=1)

            if "successful" in result.lower():
                self.logger.info("GLM API connection test passed")
                return True
            else:
                self.logger.warning(
                    "GLM API connection test failed - unexpected response"
                )
                return False

        except Exception as e:
            self.logger.error(f"GLM API connection test failed: {e}")
            return False

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation).

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        # Rough estimation: ~1 token per 4 characters for Chinese/English mixed text
        # This is a simplified estimate - actual tokenization may vary
        return len(text) // 4

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the configured model.

        Returns:
            Dictionary with model information
        """
        return {
            "model": self.model,
            "api_url": self.api_url,
            "timeout": self.timeout,
            "estimated_max_tokens": 8192 if "flash" in self.model else 32768,
            "supports_json": True,
            "supports_chinese": True,
        }
