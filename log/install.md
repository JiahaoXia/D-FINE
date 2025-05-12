### Setup Environment

```bash
# base env: unidepth-py3.11.11
python -m venv venv_dfine
source venv_dfine/bin/activate

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```