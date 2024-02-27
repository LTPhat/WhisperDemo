
#---- for general
import os
import json
import sys
import numpy as np
import time
import re

#----- for whisper
import whisper
import datetime
from sklearn.cluster import AgglomerativeClustering
import torch


#----- for nemo
#import nemo.collections.asr as nemo_asr


#--------------------------------------------------------------------------------------
def main ():
    #--- input argument
    audio_file_path  = sys.argv[1]
    dest_folder_path = sys.argv[2]
    model_type       = sys.argv[3]
    run_device       = sys.argv[4]
    speaker_number   = int(sys.argv[5])

    #--- settings
    spk_dia_dict = {}   

    #---- get results form whisper model
    whisper_model = whisper.load_model(model_type, run_device)    
    wp_results = whisper_model.transcribe(audio_file_path)
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
        embeddings = np.zeros(shape=(len(segments), 384))# 384 concaternate from two embeddings from two pre-trained NEMO models (192)

        for i, segment in enumerate(segments):
            start = segment["start"]
            end = segment["end"]

            split_file_name = './tmp_segment.wav'

            cmd= 'ffmpeg -i '+audio_file_path+' -acodec copy -ss '+str(start)+' -to '+str(end)+' '+split_file_name
            os.system(cmd)

            audio = whisper.load_audio(split_file_name)
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

            cmd = 'rm '+split_file_name
            os.system(cmd)

        #--- clustering spk emb
        clustering = AgglomerativeClustering(speaker_number, compute_distances=True).fit(embeddings)
        labels = clustering.labels_

        for i in range(len(segments)):
            wp_results['segments'][i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

    # only one sentence            
    else:
         wp_results['segments'][0]["speaker"] = 'SPEAKER 1'
    
    #--- write to json output
    pp, filename = os.path.split(audio_file_path)
    name, extension = os.path.splitext(filename)
    output_file = os.path.join(dest_folder_path, name + '.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(wp_results, f, ensure_ascii=False)


#---------------------------------------------------------------------------------------------

if __name__ == "__main__":
  main()


