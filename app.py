import streamlit as st
import numpy as np
import os
import whisper
import json
import whisper
from sklearn.cluster import AgglomerativeClustering
import torch
import librosa

UPLOAD_FOLDER = "./uploads"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['wav']


def process_wav(audio_file, speaker_number, model_type, run_device = 'cpu', sr = 16000):
    embedding_dims = {"tiny": 384, 'small': 768, 'base': 512, 'medium':1024}
    #---- get results from whisper model
    whisper_model = whisper.load_model(model_type, run_device)
    wp_results = whisper_model.transcribe(audio_file)
    for ide in range(len(wp_results['segments'])):
        del wp_results['segments'][ide]['seek']
        del wp_results['segments'][ide]['tokens']
        del wp_results['segments'][ide]['compression_ratio']
        del wp_results['segments'][ide]['temperature']
        del wp_results['segments'][ide]['avg_logprob']
        del wp_results['segments'][ide]['no_speech_prob']

    #---- solve each segment
    segments = wp_results["segments"]

    # >= 2 sentences
    if len(segments) > 1:
        embeddings = np.zeros(shape=(len(segments), embedding_dims[model_type]))

        for i, segment in enumerate(segments):
            start = int(segment["start"] * sr)
            end = int(segment["end"] * sr)

            # Extract a segment
            audio = audio_file[start: end]
            mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)
            
            #--- this code to create the correct shape of mel spectrogram
            while True:
                nF, nT = np.shape(mel)
                if nT > 3000:
                    mel = mel[:,0:3000]
                    break
                else:
                    mel = torch.cat((mel, mel), -1)
            mel = torch.unsqueeze(mel, 0)
            wp_emb = whisper_model.embed_audio(mel)
            #print(np.shape(wp_emb))

            emb_1d  = np.mean(wp_emb.cpu().detach().numpy(), axis=0)
            emb_1d  = np.mean(emb_1d, axis=0)
            #print(np.shape(emb_1d))
            #exit()
            embeddings[i] = emb_1d


        #--- clustering spk emb
        clustering = AgglomerativeClustering(speaker_number, compute_distances=True).fit(embeddings)
        labels = clustering.labels_

        for i in range(len(segments)):
            wp_results['segments'][i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

    # only one sentence
    else:
         wp_results['segments'][0]["speaker"] = 'SPEAKER 1'

    return wp_results 


def main():

    title_style = """
    <style>
    .title {
        text-align: center;
        font-size: 45px;
    }
    </style>
    """
    st.markdown(
    title_style,
    unsafe_allow_html=True
    )
    title  = """
    <h1 class = "title" >Speaker Diarization</h1>
    </div>
    """
    st.markdown(title,
                unsafe_allow_html=True)
    # st.title("Speaker Diarization")


    # Get user inputs
    file = st.file_uploader("Upload a WAV file:", type=["wav"])
    num_speakers = st.number_input("Number of speakers:", min_value=2, max_value=100)

    model_list = ['tiny', 'small', 'base', 'medium']
    model_type = st.selectbox("Select model type: ", model_list)

    # Display the result
    st.write("Your uploaded wav file: ")
    st.audio(file, format = 'audio/wav')
    if st.button("Submit"):
        if file is not None:
            
            # Read audio file using pydub
            audio_file, _ = librosa.load(file, sr=16000)

            # Process the uploaded file using the AI model
            wp_results = process_wav(audio_file, num_speakers, model_type)

            # Write result:
            st.write("Text:")
            st.write(wp_results['text'])
            st.write("Segments:" )
            for seg in wp_results['segments']:
                st.write(seg)
            st.write("Language: ", wp_results['language'])
        else:
            print("Error")
    st.write("\n\n---\n\n")  
    st.write("Built with Docker and Streamlit")
    return 



if __name__ == "__main__":
    main()
