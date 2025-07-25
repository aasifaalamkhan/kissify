# download_models.py
import os
from huggingface_hub import snapshot_download, login

token = os.getenv('HUGGING_FACE_HUB_TOKEN')
if token:
    print('Logging in to Hugging Face Hub...')
    login(token=token, add_to_git_credential=False)
else:
    print('HUGGING_FACE_HUB_TOKEN not set as build arg. Proceeding without explicit login. Gated models may fail.')

hf_cache_dir = os.getenv('HF_HOME', '/app/hf_cache')
print(f'Using HF_HOME: {hf_cache_dir}')

print('Pre-fetching SG161222/Realistic_Vision_V5.1_noVAE...')
snapshot_download(repo_id='SG161222/Realistic_Vision_V5.1_noVAE', allow_patterns=['*.safetensors', '*.json'], cache_dir=hf_cache_dir, resume_download=True, token=token)

print('Pre-fetching guoyww/animatediff-motion-adapter-v1-5-2...')
snapshot_download(repo_id='guoyww/animatediff-motion-adapter-v1-5-2', allow_patterns=['*.safetensors', '*.json'], cache_dir=hf_cache_dir, resume_download=True, token=token)

print('Pre-fetching lllyasviel/control_v11p_sd15_openpose...')
snapshot_download(repo_id='lllyasviel/control_v11p_sd15_openpose', allow_patterns=['*.safetensors', '*.json'], cache_dir=hf_cache_dir, resume_download=True, token=token)

print('Pre-fetching lllyasviel/control_v11f1p_sd15_depth...')
snapshot_download(repo_id='lllyasviel/control_v11f1p_sd15_depth', allow_patterns=['*.safetensors', '*.json'], cache_dir=hf_cache_dir, resume_download=True, token=token)

print('Pre-fetching h94/IP-Adapter...')
snapshot_download(repo_id='h94/IP-Adapter', allow_patterns=['models/*', 'ip-adapter_sd15.bin'], cache_dir=hf_cache_dir, resume_download=True, token=token)

print('Pre-fetching openai/clip-vit-large-patch14...')
snapshot_download(repo_id='openai/clip-vit-large-patch14', cache_dir=hf_cache_dir, resume_download=True, token=token)

print('Pre-fetching lllyasviel/ControlNet (for aux detectors)...')
snapshot_download(repo_id='lllyasviel/ControlNet', cache_dir=hf_cache_dir, resume_download=True, token=token)

print('Model pre-fetching complete.')