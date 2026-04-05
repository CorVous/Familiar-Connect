# Python Template

Cassidy's opinionated Python template. Enter at your own peril.

## Development

Requires [uv](https://docs.astral.sh/uv/)

Setup: `uv sync --dev`

Lint: `uv run ruff check`

Format: `uv run ruff format`

Type-check: `uv run ty check`

Test: `uv run pytest`

Security audit: `uv audit --preview-features audit`

## Template Setup

1. **Create a new repository** from this template on GitHub (click "Use this template")

2. **Clone your new repository** and navigate to it

3. **Update project metadata** in `pyproject.toml`:
   - Change `name` to your project name (use kebab-case, e.g., `my-awesome-tool`)
   - Update the `[project.scripts]` entry point name to match (e.g., `my-awesome-tool = "my_awesome_tool.cli:main"`)
   - Update `authors` with your information
   - Modify `requires-python` if needed
   - Update dependencies as required

4. **Rename the package directory**:
   - Rename `src/python_template/` to `src/your_package_name/` (use snake_case, e.g., `my_awesome_tool`)
   - Update imports in `pyproject.toml` (the `[project.scripts]` entry and `version-file` path)
   - Update imports in test files (`from python_template` → `from your_package_name`)
   - Update imports in `src/your_package_name/__init__.py`

5. **Update `.vscode/launch.json`** (if using VS Code):
   - Change `"module": "python_template"` to `"module": "your_package_name"` in all configurations

6. **Update the README**:
   - Replace the title and description
   - Remove or customize this Template Setup section

7. **Initialize your environment**:
   ```bash
   uv sync --dev
   ```

8. **Verify everything works**:
   ```bash
   uv run pytest
   uv run ruff check
   uv run ty check
   ```
