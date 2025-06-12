BootStrap: docker 
From: python:3.10.16-slim-bookworm

%files
    requirements.txt
    $SSL_CERT_FILE

%post
    apt-get update  
    python -m venv .venv
    .venv/bin/pip install --no-cache-dir -r requirements.txt

%environment
    export PATH=/.venv/bin:$PATH