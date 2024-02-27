# Speaker Diarization - Whisper 

- **Step 1:** Get Whisper mdoel

```python

git clone https://github.com/openai/whisper.git

```
- **Step 2:** Create **pre_trained_model** folder to store weights

```python

mkdir pre_trained_model

```

- **Step 3:** Download weights from https://huggingface.co/models?search=openai/whisper and store to created folder in step 2.

- **Step 4:** Run demo at local
  - cmd:
  
  ```python
  python [audio_file_path] [dest_folder_path] [model_type] [run_device] [speaker_number]
  ```

  ```python
  python data/test.wav . "base" "cpu" 2
  ```

  - app:
  ```python
  streamlit run app.py
  ```

  
