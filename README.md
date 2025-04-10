# sightsweep

## Prerequisites

*   **Python:** Version >= 3.8 (as specified in `pyproject.toml`).
*   **uv:** Install the `uv` package manager, typically using pip:
    ```bash
    pip install uv
    ```
    (See the [official `uv` guide](https://github.com/astral-sh/uv#installation) for alternative installation methods if needed.)

## Installation

It's highly recommended to install `sightsweep` within a virtual environment.

1.  **Navigate to the project root directory** (the one containing `pyproject.toml`):
    ```bash
    cd path/to/sightsweep
    ```

2.  **Create a virtual environment using `uv`:**
    This command creates a virtual environment named `.venv` in the current directory.
    ```bash
    uv venv
    ```
    *(You can choose a different name: `uv venv my_env_name`)*

3.  **Activate the virtual environment:**

    *   **Bash / Zsh:**
        ```bash
        source .venv/bin/activate
        ```
    *   **CMD (Windows):**
        ```cmd
        .venv\Scripts\activate
        ```
    Your terminal prompt should now indicate that you are inside the virtual environment (e.g., `(.venv) your-prompt$`).

4.  **Install `sightsweep` within the activated environment:**

    *   **Standard Install:**
        ```bash
        # Make sure your virtual environment is active before running this!
        uv pip install .
        ```
        This installs the package and its dependencies (from `requirements.txt`) into the isolated `.venv` environment.

    *   **Editable (Development) Install:**
        ```bash
        # Make sure your virtual environment is active!
        uv pip install -e .
        ```
        This links the installed package to your source code within the `.venv`, so changes are reflected without reinstalling.

## Basic Usage

*(Add your basic usage instructions here, assuming the virtual environment is active)*

---

**To leave the virtual environment:**

Simply run the command:
```bash
deactivate