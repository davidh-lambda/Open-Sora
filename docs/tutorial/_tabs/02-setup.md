---
layout: post
icon: fas fa-cogs
title: Setup
date: 2024-10-02
toc: true
---

# **Clone**, **Install** & **Set Up** Your Cluster

In this section, we'll guide you through the essential steps to set up your environment for training the Open-Sora 1.2 model. This includes preparing a shared folder, installing Miniconda across all nodes, cloning the necessary codebase, and ensuring that all nodes have access to required files - and most importantly - the same environments. (Any discrepancies can lead to hard-to-diagnose bugs during training.)


## **Using a Shared Folder**
> **Note:** Make sure to use a shared folder that is accessible by all nodes in your cluster. Alternatively, ensure that the environment is identical across the entire cluster **after installing all dependencies** to avoid potential issues.
{: .prompt-tip }

To start, it's crucial to have a shared folder that's accessible to all nodes in your cluster. This ensures consistency and prevents hard-to-find bugs caused by environmental differences across nodes.

This tutorial will assume you have a shared folder across your cluster.
Let's set it's path:
```bash
export SHAREDDIR=./shared/folder/path
```


## **Compiling a List of Cluster Nodes**
First compile a list of all nodes in your cluster and save it in a convenient location, such as `~/nodes.txt`{: .filepath}. This file will be used for executing commands across all nodes efficiently.

Create `nodes.txt`{: .filepath} containing the names of all nodes that you want to use:
```bash
node-001
node-002
node-003
...
```
(Of course, replace `node-001`, `node-002`, etc., with the actual hostnames or IP addresses of your cluster nodes.)

> **Note:** Ensure that you have SSH access configured for all nodes to execute remote commands.
{: .prompt-tip}

## **Installing Miniconda Across All Nodes**

Miniconda is a minimal installer for Conda, which is an open-source package management system and environment management system. We'll install Miniconda to manage Python environments and dependencies such as CUDA and NCCL. Afterwards, we'll ensure that all nodes have access to the exact same environment.

1. **First, we'll install Miniconda3 on a single node:**
    1. **Install Miniconda3 to the shared folder**, for example, `$SHAREDDIR/miniconda3/`.  
       You can download the installer using the following command:
       ```bash
       wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $SHAREDDIR/miniconda.sh
       ```
    2. **Run the installer**.  
       (This will install Miniconda3 in the shared directory `$SHAREDDIR/miniconda3`).
       ```bash
       bash $SHAREDDIR/miniconda.sh -b -p $SHAREDDIR/miniconda3
       ```
3. **Initializing Conda on All Nodes**  
    After installation of miniconda, you'll need to append the Conda initialization snippet to the `.bashrc` file on all nodes. This ensures that the Conda environment is properly set up whenever you log in.

    1. **Create a file `$SHAREDDIR/conda_init.sh` using the following command:**
       ```bash
       cat <<EOF > $SHAREDDIR/conda_init.sh
       # >>> conda initialize >>>
       # !! Contents within this block are managed by 'conda init' !!
       __conda_setup="\$('$SHAREDDIR/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
       if [ \$? -eq 0 ]; then
           eval "\$__conda_setup"
       else
           if [ -f "$SHAREDDIR/miniconda3/etc/profile.d/conda.sh" ]; then
               . "$SHAREDDIR/miniconda3/etc/profile.d/conda.sh"
           else
               export PATH="$SHAREDDIR/miniconda3/bin:\$PATH"
           fi
       fi
       unset __conda_setup
       # <<< conda initialize <<<
       EOF
       ```
    2. **Now, append this snippet to the `.bashrc` file on all nodes.**  
       You can do this efficiently by using the following command:
       ```bash
       parallel -a ~/nodes.txt ssh {} 'cat $SHAREDDIR/conda_init.sh >> ~/.bashrc'
       ```
       (This command reads the list of nodes from `~/nodes.txt` and executes the `cat` command on each node to append `conda_init.sh` to `~/.bashrc`.)
    3. **Ensure that the Conda environment is available on all nodes:**
      ```bash
      parallel -a ~/nodes.txt ssh {} 'source ~/.bashrc && command -v conda >/dev/null 2>&1 && echo "{}: Conda is installed" || echo "{}: Conda is NOT installed"'
      ```
4. **Ensure Only Conda's Python Libraries Are Used**

    To prevent Python from inadvertently importing packages from `~/.local/python`, it's essential to disable user site-packages. This ensures that only the libraries managed by Conda are used during execution.

    **Export the `PYTHONNOUSERSITE` Environment Variable**

    Add the following line to the end of your `.bashrc` file on all nodes:

    ```bash
    export PYTHONNOUSERSITE=True
    ```

    > **Important:** Setting `PYTHONNOUSERSITE=True` ensures that Python does not consider the user-specific site-packages directory (`~/.local/lib/pythonX.Y/site-packages`) when importing modules. This helps maintain a clean and predictable Python environment, preventing conflicts with Conda-managed packages. (Bugs are especially hard to find, if all but only a few nodes use the conda environment, while some nodes use different versions that have been installed locally.)
    {: .prompt-tip}

    You can append this line to all nodes using the following command:

    ```bash
    parallel -a ~/nodes.txt ssh {} 'echo "export PYTHONNOUSERSITE=True" >> ~/.bashrc'
    ```



## **Cloning and Configuring the Codebase**
We'll now clone our fork of the Open-Sora repository.

1. **Clone the Repository**
   ```bash
   git clone https://github.com/LambdaLabsML/Open-Sora.git
   ```
2. **Checkout the Training Branch**  
   Navigate to the repository and switch to the `retrain_from_scratch` branch: (TODO: push this to main?)
   ```bash
   cd Open-Sora
   git checkout retrain_from_scratch
   ```
3. **Run the Installation Script**  
   The installation script will handle the entire setup process, including creating the Conda environment and installing all necessary dependencies. Run the script:
   ```bash
   bash install.sh
   ```

   > **Note:** This process may take some time. Feel free to grab a coffee while you wait or read what it does [below](#installation-details)!
   Click here to read more about the details of the installation process.
   {: .prompt-tip}
4. **Activating the Conda Environment**  
   After the installation completes, activate the new Conda environment `osora-12`:
   ```bash
   conda activate osora-12
   ```
5. **Verifying the Installation**  
   To confirm that everything is set up correctly, run the `install-check.py` script:

   ```bash
   python install-check.py
   ```

   You should see the following output:

   ```shell
   Starting environment check...

   Checking nvcc version... OK
   Checking Python version... OK
   Checking PyTorch version... OK
   Checking CUDA version... OK
   Checking Apex... OK
   Checking Flash Attention... OK
   Checking xFormers... OK

   SUCCESS: All checks passed!
   ```


### **Installation Details**
TODO: either change to torch 2.2 and simplify or check if this is reproducable and adapt the install-checker.

**Optional:** The following details provide additional insights into what the installation script does. You can skip this section if you're not interested in the specifics.

{% details Click here to read more details about the installation script. %}

- **Creating a New Conda Environment**  
  The script creates a new Conda environment named `osora-12`.
- **Installing CUDA Dependencies**  
  CUDA dependencies are installed via the [NVIDIA channel](https://anaconda.org/nvidia/cudatoolkit), all fixed to version 12.1 to ensure compatibility.
- **Compiling and Installing NCCL**  
  NCCL (version 2.20.5-1) is compiled and installed. This version has been tested and works well in our setup.
- **Installing PyTorch 2.3.1**  
  PyTorch 2.3.1 is installed. This version allows dynamic linking of NCCL, enabling you to test other NCCL versions without reinstalling PyTorch.
- **Installing xFormers**  
  [xFormers](https://github.com/facebookresearch/xformers) version 0.0.26 is installed. This package provides efficient Transformer building blocks and is compiled against PyTorch.
- **Installing FlashAttention**  
  [FlashAttention](https://github.com/HazyResearch/flash-attention) version 2.5.8 is installed. It's an efficient attention implementation that speeds up Transformer models, also compiled against the specific PyTorch version.
- **Installing Other Dependencies**  
  - **ColossalAI**  
    [ColossalAI](https://github.com/hpcaitech/ColossalAI) is installed from a custom branch that works with PyTorch 2.3. This package provides a unified interface for large-scale model training, also compiled against the specific PyTorch version.
  - **Diffusers**  
    [Diffusers](https://github.com/huggingface/diffusers) is installed, which is a dependency of ColossalAI. This package offers tools for diffusion models.
  - **Bug Fixes**  
    - **YAPF Package Bugfix**
    - **Protobuf Package Bugfix**
    These fixes ensure compatibility and prevent common issues during installation.
{% enddetails %}


## **Preparing the Cluster**

Ensuring that all nodes in your cluster have access to the necessary files and environments is critical for distributed training.

> **Important:** The following steps will also download the parts of the network that are not trained from scratch, such as the pre-trained [3D VAE](https://github.com/hpcaitech/Open-Sora/blob/main/docs/report_03.md#video-compression-network) for efficient video compression and the [T5 text model](https://huggingface.co/docs/transformers/model_doc/t5).
{: .prompt-tip}

### **Distributing Necessary Files Across Nodes**

1. **Perform a Test Inference & Download Required Model Weights**  
   We'll run an inference script to (1) test if the installation is working correctly and (2) to automatically download the required pre-trained models into the Hugging Face cache directory `~/.cache/huggingface/hub`{: .filepath}.
   ```bash
   python scripts/inference.py configs/opensora-v1-1/inference/sample.py --prompt "A beautiful sunset over the city" --num-frames 32 --image-size 480 854
   ```
2. **Copying the Hugging Face Cache to Shared Storage**  
   Since only the current node has the downloaded models so far, we'll copy them to a shared directory accessible by all nodes.
   ```bash
   cp -r ~/.cache/huggingface/hub $SHAREDDIR/opensora_hub_ckpts
   ```
3. **Distributing the Cache to All Nodes**  
   Now, copy the cache directory to the Hugging Face cache directory on each node.
   This ensures that all nodes have access to the pre-trained models required for training and inference.
   (If you don't do that and the nodes don't download the models themselves, the error messages don't explicitly tell you that weights are missing. )
   ```bash
   parallel -a ~/nodes.txt ssh {} 'cp -r $SHAREDDIR/opensora_hub_ckpts/* ~/.cache/huggingface/hub/'
   ```


## **Initializing Weights & Biases (wandb) on All Nodes**

[Weights & Biases](https://wandb.ai/) (wandb) is a tool for tracking experiments, visualizing metrics, and collaborating with others. We'll initialize wandb on all nodes to monitor our training process.

### **Setting Up wandb**

1. **Install wandb in the Conda Environment**

   If wandb is not already installed, install it in your `osora-12` environment:

   ```bash
   conda activate osora-12
   pip install wandb
   ```
2. **Log In to wandb**

   You'll need to log in to wandb on each node. However, since we have a shared home directory or can execute commands on all nodes, we can automate this process.

   **Option 1: Using Shared Configuration**  
   If your home directory is shared across all nodes, logging in once is sufficient.

   **Option 2: Automating Login on All Nodes**  
   If you need to log in on each node individually, you can use the following command:

   ```bash
   parallel -a ~/nodes.txt ssh {} 'wandb login YOUR_WANDB_API_KEY'
   ```

   Replace `YOUR_WANDB_API_KEY` with your actual wandb API key, which you can retrieve under the following link: [wandb.ai/authorize](https://wandb.ai/authorize).

   > **Note:** Ensure that you're securely handling your API key. Avoid exposing it in shared scripts or logs.
   {: .prompt-tip}
3. **Verify wandb Initialization**

   You can verify that wandb is set up correctly by running a simple script that logs a test metric.

   ```python
   import wandb

   wandb.init(project="open-sora-test")
   wandb.log({"test_metric": 1})
   ```

   Save this script as `wandb_test.py` and run it on all nodes:

   ```bash
   parallel -a ~/nodes.txt ssh {} 'python wandb_test.py'
   ```

   Check your wandb dashboard to see if the test runs have been logged.

---

**Next Steps**:  
By completing these steps, you've set up your cluster environment, cloned and configured the Open-Sora codebase, ensured all nodes have the necessary files and models, and initialized wandb for experiment tracking. You're now ready to proceed to the next stage: downloading and preprocessing the dataset.

Proceed to the [**Dataset** — Downloading & Preprocessing](../03-dataset) section to begin working with the data required for training the Open-Sora model.
