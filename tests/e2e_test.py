"""End-to-end tests for all tools and components."""

import asyncio
import os
import tempfile

import pytest

from lazydspy.tools.data_ops import check_schema, sample_data, validate_jsonl
from lazydspy.tools.domain_ops import estimate_cost, get_defaults, list_optimizers
from lazydspy.tools.file_ops import create_dir, read_file, write_file
from lazydspy.tools.session_ops import SessionComplete, finish_session


def run_async(coro):
    """Helper to run async functions."""
    return asyncio.get_event_loop().run_until_complete(coro)


class TestFileOpsE2E:
    """E2E tests for file operations."""

    def test_write_and_read_file(self, tmp_path):
        """Test write_file and read_file work together."""
        test_file = tmp_path / "test.txt"
        content = "Hello, World!\nLine 2"

        # Write
        result = run_async(write_file({"path": str(test_file), "content": content}))
        assert "文件已成功写入" in result["content"][0]["text"]
        assert test_file.exists()

        # Read back
        result = run_async(read_file({"path": str(test_file)}))
        assert content in result["content"][0]["text"]

    def test_create_nested_dir(self, tmp_path):
        """Test create_dir with nested directories."""
        nested = tmp_path / "a" / "b" / "c"

        result = run_async(create_dir({"path": str(nested)}))
        assert "目录已创建" in result["content"][0]["text"]
        assert nested.exists()

    def test_read_nonexistent_file(self, tmp_path):
        """Test read_file handles missing file gracefully."""
        result = run_async(read_file({"path": str(tmp_path / "nonexistent.txt")}))
        assert "文件不存在" in result["content"][0]["text"]


class TestDataOpsE2E:
    """E2E tests for data operations."""

    def test_validate_valid_jsonl(self, tmp_path):
        """Test validate_jsonl with valid data."""
        jsonl_file = tmp_path / "valid.jsonl"
        jsonl_file.write_text(
            '{"query": "q1", "answer": "a1"}\n'
            '{"query": "q2", "answer": "a2"}\n',
            encoding="utf-8",
        )

        result = run_async(
            validate_jsonl(
                {
                    "path": str(jsonl_file),
                    "required_fields": ["query", "answer"],
                }
            )
        )
        assert "验证通过" in result["content"][0]["text"]

    def test_validate_invalid_jsonl(self, tmp_path):
        """Test validate_jsonl detects missing fields."""
        jsonl_file = tmp_path / "invalid.jsonl"
        jsonl_file.write_text('{"query": "q1"}\n', encoding="utf-8")

        result = run_async(
            validate_jsonl(
                {
                    "path": str(jsonl_file),
                    "required_fields": ["query", "answer"],
                }
            )
        )
        assert "缺少字段" in result["content"][0]["text"]

    def test_check_schema_pass(self):
        """Test check_schema with matching fields."""
        result = run_async(
            check_schema(
                {
                    "data": {"query": "test", "answer": "response"},
                    "expected_fields": ["query", "answer"],
                }
            )
        )
        assert "检查通过" in result["content"][0]["text"]

    def test_check_schema_missing(self):
        """Test check_schema detects missing fields."""
        result = run_async(
            check_schema(
                {
                    "data": {"query": "test"},
                    "expected_fields": ["query", "answer"],
                }
            )
        )
        assert "缺少字段" in result["content"][0]["text"]

    def test_sample_data_generation(self):
        """Test sample_data generates correct format."""
        result = run_async(
            sample_data(
                {
                    "input_fields": ["query", "context"],
                    "output_fields": ["answer"],
                    "num_samples": 3,
                }
            )
        )
        text = result["content"][0]["text"]
        assert "3 条样例数据" in text
        assert "query" in text
        assert "answer" in text


class TestDomainOpsE2E:
    """E2E tests for domain operations."""

    def test_estimate_cost_gepa_quick(self):
        """Test cost estimation for GEPA quick mode."""
        result = run_async(
            estimate_cost(
                {
                    "optimizer": "gepa",
                    "mode": "quick",
                    "dataset_size": 100,
                }
            )
        )
        text = result["content"][0]["text"]
        assert "成本估算结果" in text
        assert "预估成本" in text
        assert "gepa" in text.lower()

    def test_estimate_cost_miprov2_full(self):
        """Test cost estimation for MIPROv2 full mode."""
        result = run_async(
            estimate_cost(
                {
                    "optimizer": "miprov2",
                    "mode": "full",
                    "dataset_size": 50,
                }
            )
        )
        text = result["content"][0]["text"]
        assert "miprov2" in text.lower()

    def test_list_optimizers_returns_both(self):
        """Test list_optimizers returns all optimizers."""
        result = run_async(list_optimizers({}))
        text = result["content"][0]["text"]
        assert "GEPA" in text
        assert "MIPROv2" in text

    def test_get_defaults_gepa(self):
        """Test get_defaults for GEPA."""
        result = run_async(get_defaults({"optimizer": "gepa", "mode": "quick"}))
        text = result["content"][0]["text"]
        assert "breadth" in text
        assert "depth" in text

    def test_get_defaults_miprov2(self):
        """Test get_defaults for MIPROv2."""
        result = run_async(get_defaults({"optimizer": "miprov2", "mode": "full"}))
        text = result["content"][0]["text"]
        assert "search_size" in text


class TestSessionOpsE2E:
    """E2E tests for session operations."""

    def test_finish_session_raises_exception(self):
        """Test finish_session raises SessionComplete."""
        with pytest.raises(SessionComplete) as exc_info:
            run_async(
                finish_session(
                    {
                        "summary": "All scripts generated",
                        "next_steps": ["Run pipeline.py", "Check results"],
                    }
                )
            )

        exc = exc_info.value
        assert exc.summary == "All scripts generated"
        assert exc.next_steps == ["Run pipeline.py", "Check results"]

    def test_finish_session_default_values(self):
        """Test finish_session with minimal input."""
        with pytest.raises(SessionComplete) as exc_info:
            run_async(finish_session({}))

        exc = exc_info.value
        assert exc.summary == "任务已完成"
        assert exc.next_steps == []


class TestIntegrationE2E:
    """Integration tests combining multiple components."""

    def test_full_workflow_simulation(self, tmp_path):
        """Simulate a full workflow: create dir, write file, validate."""
        # Step 1: Create output directory
        output_dir = tmp_path / "generated" / "session-001"
        run_async(create_dir({"path": str(output_dir)}))
        assert output_dir.exists()

        # Step 2: Generate sample data
        sample_result = run_async(
            sample_data(
                {
                    "input_fields": ["query"],
                    "output_fields": ["answer"],
                    "num_samples": 2,
                }
            )
        )

        # Step 3: Write a script file
        script_content = '''# /// script
# requires-python = ">=3.12"
# dependencies = ["dspy", "pydantic>=2"]
# ///

print("Hello from generated script")
'''
        script_path = output_dir / "pipeline.py"
        run_async(write_file({"path": str(script_path), "content": script_content}))
        assert script_path.exists()

        # Step 4: Write sample data
        sample_data_path = output_dir / "sample-data" / "train.jsonl"
        run_async(create_dir({"path": str(sample_data_path.parent)}))
        run_async(
            write_file(
                {
                    "path": str(sample_data_path),
                    "content": '{"query": "test", "answer": "response"}\n',
                }
            )
        )
        assert sample_data_path.exists()

        # Step 5: Validate the data
        validate_result = run_async(
            validate_jsonl(
                {
                    "path": str(sample_data_path),
                    "required_fields": ["query", "answer"],
                }
            )
        )
        assert "验证通过" in validate_result["content"][0]["text"]

        # Step 6: Signal completion
        with pytest.raises(SessionComplete) as exc_info:
            run_async(
                finish_session(
                    {
                        "summary": f"Generated files in {output_dir}",
                        "next_steps": [
                            "Run: uv run pipeline.py --data sample-data/train.jsonl",
                        ],
                    }
                )
            )

        assert "Generated files" in exc_info.value.summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
