
# MEGPrep: A Scalable and Reproducible Pipeline for Large-Scale MEG Preprocessing

[![Documentation Status](https://readthedocs.org/projects/megprep/badge/?version=latest)](https://megprep.readthedocs.io/)
[![Docker Pulls](https://img.shields.io/docker/pulls/cmrlab/megprep)](https://hub.docker.com/r/cmrlab/megprep)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**MEGPrep** is a fully automated preprocessing pipeline for MEG (Magnetoencephalography) data, built on the **MNE-Python** framework and leveraging the power of **Nextflow**. 

It is specifically designed to address the challenges of large-scale MEG data processing with a strong emphasis on reproducibility, efficiency, and user-friendliness in various research environments.

---

## 🌟 Key Features

### 🛡️ Reliability and Robustness
Standardized environments through containerization (**Docker** and **Singularity**) guarantee consistent results across computational setups. This minimizes variability and ensures reproducibility across different systems, facilitating cross-subject and cross-site studies.

### ⚡ Acceleration and Parallelization
By using the **Nextflow** framework, MEGPrep dramatically accelerates the pipeline. It is optimized for high parallelization, capable of managing heavy workloads and significantly speeding up data processing through concurrent execution of tasks.

### 🧩 Modularity and Integrability
Designed with modularity in mind, enabling users to customize workflows easily. It integrates seamlessly with various libraries (including `mne-python`) for enhanced processing and analysis.

### 🤖 Automated Processes
Streamlines preprocessing with automated detection processes to reduce manual intervention:
*   Automatic Artifacts Rejection
*   ICA (Independent Component Analysis) Automatic Detection
*   Auto-coregistration

### 📊 Interoperability and Standards
Includes an interactive reporting feature based on **Streamlit**, allowing users to visualize quality control metrics at each step and receive alerts for anomalies.

### ⚙️ Parameter Configuration
Offers an easy-to-use configuration system. Researchers can adapt the preprocessing pipeline to unique datasets without complex coding.

---

## 🚀 Installation

MEGPrep is distributed as a Docker container. You must have **Docker** installed on your system.

1.  **Install Docker**: Follow the instructions for your operating system on the [Docker official website](https://docs.docker.com/get-docker/).
2.  **Pull the Image**:
    ```bash
    docker pull cmrlab/megprep:<version>
    ```
    *(Replace `<version>` with the specific version tag, e.g., `0.0.3` or `latest`)*

---

## 💻 Usage

### Basic Command Structure
```bash
docker run cmrlab/megprep:<version> [nextflow_options]
```

### Main Options

| Option | Description |
| :--- | :--- |
| `-c`, `--config` | Specify the Nextflow config file (default: `nextflow.config`) |
| `-i`, `--input` | Specify the input directory |
| `-o`, `--output` | Specify the output directory (including report results) |
| `-r`, `--view-report` | Run Streamlit to view the report (does not run Nextflow) |
| `--fs_license_file` | Specify the FreeSurfer license file path |
| `--fs_subjects_dir` | Specify the FreeSurfer `SUBJECTS_DIR` containing processed T1 results |
| `--t1_dir` | Specify the T1 image directory |
| `--t1_input_type` | Specify the T1 input type |
| `--anat_only` | Run only the FreeSurfer/Anatomy related steps |
| `--meg_only` | Run only the MEG related steps |
| `--resume` | Resume the previous run (Nextflow option) |

### Example: Running a Full Pipeline
Here is a comprehensive example mapping input/output volumes and license files:

```bash
docker run -it --rm \
    -v /data/datasets/SMN4Lang:/input \
    -v /data/datasets/SMN4Lang/preprocessed:/output \
    -v /data/datasets/SMN4Lang/smri:/smri \
    -v /data/megprep/license.txt:/fs_license.txt \
    -v /data/megprep/nextflow/nextflow.config:/program/nextflow/nextflow.config \
    cmrlab/megprep:0.0.3 \
    -i /input \
    -o /output \
    --fs_license_file /fs_license.txt \
    --fs_subjects_dir /smri \
    --resume
```

---

## 📈 Quality Control Reports

MEGPrep generates interactive quality control reports via Streamlit.

### How to View Reports
Use the `-r` flag and map port `8501`:

```bash
docker run -p 8501:8501 -v /data/liaopan/datasets/SMN4Lang/g:/output cmrlab/megprep:<version> -r
```

**Access via browser:**  
👉 `http://<server_ip>:8501` (or `http://localhost:8501` if running locally)

---

## 🐛 Bug Reports & Feedback

If you encounter any bugs, anomalies, or have suggestions for improvements, please report them via the **GitHub Issues** page.

When reporting a bug, please include:
1.  **System Information**: OS version, Docker version.
2.  **Command Used**: The exact command line you executed.
3.  **Logs**: The relevant part of the error log or traceback (please use code blocks).
4.  **Description**: A clear description of what you expected to happen versus what actually happened.

👉 [Report an Issue](https://github.com/liaopan/megprep/issues)

---

## 🛠️ Development

We welcome contributions to MEGPrep! If you want to contribute code or improve documentation, please follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/liaopan/megprep.git
    cd megprep
    ```

2. **Environment Setup:**
    If you plan to develop or run the pipeline locally (outside Docker), you must install Nextflow.

    **Prerequisites:**
    *   **System**: Any POSIX-compatible system (Linux, macOS, etc.), or Windows through WSL.
    *   **Dependencies**: Bash 3.2+ and **Java 17** (up to 23).

    **Installation:**
    Please refer to the [Nextflow Official Documentation](https://www.nextflow.io/docs/latest/install.html).

    If you use SDKMAN (recommended), initialize it:
    ```bash
    source "$HOME/.sdkman/bin/sdkman-init.sh"
    ```

    **Configuration:**
    Ensure the Nextflow binary is in your PATH.
    *   Common location: `$HOME/.local/bin/nextflow`

    **Useful Nextflow Developer Commands:**

    *   **Check Installation**:
    ```bash
    nextflow info
    ```

    *   **Run with Trace** (creates an execution trace file):
    ```bash
    nextflow run <script.nf> -with-trace
    ```

3.  **Build Docker Image Locally (Optional):**
    If you modified the Dockerfile or dependencies, you can build the image manually using Docker or the provided helper script.
    
    **Using the build script (Recommended):**
    ```bash
    bash build_megprep.sh
    ```
    
    **Using Docker directly:**
    ```bash
    docker build -t megprep:local -f megprep.Dockerfile .
    ```

4.  **Submit a Pull Request:**
    *   Fork the repository.
    *   Create a new branch for your feature or fix.
    *   Commit your changes and push to your fork.
    *   Submit a Pull Request to the `main` branch.