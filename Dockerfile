
#Dockerfile

FROM python:3.9

LABEL author="Phat-Lam"

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory

WORKDIR $HOME/app
COPY --chown=user . $HOME/app


USER root

RUN apt-get update 
RUN apt-get --yes install ffmpeg 

# Switch back to non-root user
USER user

RUN pip install --upgrade pip --timeout 120
RUN pip install -U openai-whisper --timeout 1000
RUN pip install scikit-learn==1.3.2 --timeout 1000
RUN pip install streamlit --timeout 500
RUN pip install ffmpeg-python --timeout 1000
RUN pip install toml
RUN pip install librosa

# RUN pip uninstall ffmpeg --yes
# RUN pip uninstall ffmpeg-python --yes


# EXPOSE 8501

# HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]



CMD ["streamlit","run", "./app.py", "--server.address", "0.0.0.0", "--server.port", "7860"]

