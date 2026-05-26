## llamafactory训练的qwen3-vl安装环境：
### 安装python3.11, 我使用的是python3.11.9
### 安装最新版本llamafactory，支持训练qwen3-vl
pip install /home/ma-user/modelarts/package/moxing_framework-2.3.13-py2.py3-none-any.whl
cd /home/ma-user/llm_train/LLaMAFactory/
# pip install /home/ma-user/modelarts/package/moxing_framework-2.2.10-py2.py3-none-any.whl
rm -rf /home/ma-user/llm_train/LLaMAFactory/LLaMA-Factory-main/
python -c "import moxing as mox; mox.file.copy_parallel('s3://bucket-pangu-green-guiyang/zhaojie/MLLMData/ml_caption/code/llm_train/LLaMAFactory/LLaMA-Factory-main/', '/home/ma-user/llm_train/LLaMAFactory/LLaMA-Factory-main/',threads=256)"
pip install --upgrade pip
cd /home/ma-user/llm_train/LLaMAFactory/LLaMA-Factory-main/
pip install -e ".[torch-npu,metrics]"
pip install absl-py
pip install --upgrade botocore boto3
pip install --upgrade cffi
pip install --upgrade  kiwisolver
pip install typing==3.7.4.3
pip install deepspeed==0.14.4
pip install torch_npu==2.8.0
pip install torchvision==0.23.0
pip install torchaudio==2.8.0
pip install decorator #不安装decorator 会有精度报错
pip install soundfile
pip install transformers==4.57.1
pip uninstall vllm_ascend -y
pip uninstall vllm -y
cd /home/ma-user/llm_train/LLaMAFactory/
