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

## Download SAM Model
To download the SAM2 model, go to [this](https://github.com/facebookresearch/sam2?tab=readme-ov-file#model-description) page and download the `sam2.1_hiera_base_plus.pt` checkpoint 

## Download Dataset
For this project, we are using the google landmarks dataset from [here.](https://github.com/cvdfoundation/google-landmark/tree/master)

To download it using a shell script, run the following commands in bash:

```bash
cd data
mkdir train
cd train
bash ../download-dataset.sh train x # Replace x with the number of batches you want to download (Max 499)
```

Do the same thing for the index (validation) and test set


## Basic Usage

*(Add your basic usage instructions here, assuming the virtual environment is active)*

---

**To leave the virtual environment:**

Simply run the command:
```bash
deactivate