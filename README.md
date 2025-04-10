# sightsweep

## Prerequisites

*   **Python:** Version >= 3.8 (as specified in `pyproject.toml`).
*   **uv:** Install the `uv` package manager, typically using pip:
    ```bash
    pip install uv
    ```
    (See the [official `uv` guide](https://github.com/astral-sh/uv#installation) for alternative installation methods if needed.)

## Installation

1.  **Navigate to the project root directory** (the one containing `pyproject.toml`):
    ```bash
    cd path/to/sightsweep
    ```

2.  **Install using `uv`:**

    *   **Standard Install:**
        ```bash
        uv pip install .
        ```
        This installs the package and its dependencies (from `requirements.txt`) into your environment.

    *   **Editable (Development) Install:**
        ```bash
        uv pip install -e .
        ```
        This links the installed package to your source code, so changes are reflected without reinstalling.