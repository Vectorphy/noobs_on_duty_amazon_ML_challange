# Guide: Training the V3 Pipeline on AWS

Because the V3 pipeline is designed to be **CPU-only** (no GPU required for DuckDB or LightGBM), it is highly cost-effective to run on AWS. You have two primary options: using a dedicated **EC2 Instance** (best for background jobs and stability) or using **SageMaker** (best for an interactive, Colab-like experience).

---

## Option 1: Amazon EC2 (Recommended for full pipeline runs)

Using an EC2 instance gives you full control, avoids notebook timeouts, and allows you to run the training script in the background using `tmux` or `nohup`.

### 1. Launch the Instance
1. Go to the AWS EC2 Console and click **Launch Instance**.
2. **OS Image (AMI):** Select **Ubuntu Server 22.04 LTS** or **24.04 LTS**.
3. **Instance Type:** 
   - Since V3 relies heavily on DuckDB (which scales well with CPU cores) and memory-bounded LightGBM, choose a compute or memory-optimized instance.
   - **Recommended:** `c6i.2xlarge` (8 vCPUs, 16GB RAM) or `m6i.2xlarge` (8 vCPUs, 32GB RAM). If you want to train significantly faster, `c6i.4xlarge` (16 vCPUs) is excellent.
4. **Storage:** Increase the Root volume (EBS) to at least **100 GB (gp3)** to hold the OS, the dataset, and the generated DuckDB memory-mapped files.
5. **Key Pair:** Select or create an SSH key pair to connect to the instance.

### 2. Connect and Set Up the Environment
Connect to your instance via SSH:
```bash
ssh -i /path/to/your-key.pem ubuntu@<your-ec2-public-ip>
```

Update packages and install Python, pip, and git:
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-venv git tmux -y
```

### 3. Clone Repository & Setup Virtual Environment
```bash
# Clone your repository (assuming it is pushed to GitHub/GitLab)
git clone <your-repo-url> amazon-ml-challenge
cd amazon-ml-challenge

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r code/business_entity_resolution/requirements.txt
pip install -r code/business_entity_resolution/requirements-models.txt
```

### 4. Upload the Dataset
You need to place the dataset in `student_resource/dataset/`.
- If your dataset is on an S3 bucket: `aws s3 cp s3://your-bucket/dataset/ student_resource/dataset/ --recursive`
- If you are transferring from your local machine, use `scp` or `rsync`:
  ```bash
  rsync -avz -e "ssh -i /path/to/key.pem" /local/path/to/student_resource/dataset ubuntu@<ec2-ip>:~/amazon-ml-challenge/student_resource/
  ```

### 5. Run the Training Pipeline
Since the full corpus run takes time, it's best to run it inside `tmux` so it doesn't die if your SSH connection drops:
```bash
tmux new -s training
```
Then start the training script:
```bash
python code/business_entity_resolution/src/train_v3.py \
    --train-dir student_resource/dataset/train \
    --artifacts-dir code/business_entity_resolution/artifacts/v3_aws
```
*(To detach from the tmux session, press `Ctrl+B`, then `D`. To re-attach later, run `tmux attach -t training`)*

---

## Option 2: Amazon SageMaker (For an interactive notebook experience)

If you prefer the Jupyter Notebook experience (similar to what you did on Google Colab), you can use SageMaker.

### 1. Create a SageMaker Notebook Instance
1. Go to the **Amazon SageMaker Console** -> **Notebook instances** -> **Create notebook instance**.
2. **Notebook instance name:** `amazon-ml-v3-pipeline`
3. **Notebook instance type:** Select `ml.c5.2xlarge` or `ml.m5.2xlarge`. 
4. **Volume size:** Set to at least **100 GB**.
5. **IAM role:** Create a new role or use an existing one with S3 access if your data is stored in S3.
6. Click **Create notebook instance** and wait for it to be `InService`.

### 2. Upload Files and Notebook
1. Click **Open JupyterLab**.
2. Upload the `business_entity_resolution_v3_pipeline.ipynb` notebook we just created.
3. Open a terminal within JupyterLab (`File -> New -> Terminal`) and clone your repository or upload the files directly.
4. If using the terminal, install the requirements:
   ```bash
   cd SageMaker/your-repo-folder
   pip install -r code/business_entity_resolution/requirements.txt
   pip install -r code/business_entity_resolution/requirements-models.txt
   ```

### 3. Run the Notebook
Open the uploaded `business_entity_resolution_v3_pipeline.ipynb`. Modify the `PROJECT_DIR` in the notebook to point to your SageMaker directory instead of Google Drive:
```python
# Change this cell in the notebook:
PROJECT_DIR = Path('/home/ec2-user/SageMaker/amazon-ml-challenge')
```
Run the cells sequentially just like you did in Colab.

---

### Cost Optimization Tips
- **Turn off instances:** EC2 and SageMaker instances charge by the hour. **Always Stop** or **Terminate** your instances when the training finishes!
- **Spot Instances:** If you want to save up to 90% on EC2 costs, consider requesting a **Spot Instance** when launching the EC2 server. Just be aware that AWS can interrupt Spot instances if capacity is needed elsewhere.
- **Avoid GPUs:** Do not use `p` or `g` instance families (like `g4dn.xlarge`). The V3 pipeline doesn't use GPU, so you'd be paying a massive premium for hardware that sits idle.
