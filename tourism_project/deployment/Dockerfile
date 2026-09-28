# Base image: Python 3.12 (XGBoost 3.x requires Python 3.10+)
FROM python:3.12-slim

# Hugging Face Spaces run containers as user ID 1000; create that user so the
# app can write to its home directory (needed for the model download cache)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install dependencies first, so this layer is cached between code changes
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the deployment files (app.py)
COPY --chown=user . .

# Run Streamlit on port 7860, the port Hugging Face Spaces expects by default
EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
