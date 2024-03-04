import os
from pydub import AudioSegment

cafe_audio_path = "background-noise/cafe.mp3"
highway_audio_path = "background-noise/highway.mp3"
park_audio_path = "background-noise/park.mp3"

cafe_audio = AudioSegment.from_file(cafe_audio_path)
cafe_audio += 10

highway_audio = AudioSegment.from_file(highway_audio_path)
highway_audio += 10

park_audio = AudioSegment.from_file(park_audio_path)
park_audio += 10

audios = {
    "cafe": cafe_audio,
    "highway": highway_audio,
    "park": park_audio
}

root = "2023-12-04-HMI-dataset"
folders = [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]

for folder in folders:
    files = [f for f in os.listdir(os.path.join(root, folder)) if os.path.isfile(os.path.join(root, folder, f))]
    
    for file in files:
        file_path = os.path.join(root, folder, file)
        file_name = os.path.splitext(file)[0]

        main_audio = AudioSegment.from_file(file_path)

        for audio in audios:
            output_file_name = f"{file_name}-{audio}.webm"
            output_audio_path = os.path.join(root, folder, output_file_name)
            
            background_audio = audios[audio]
            final_audio = main_audio.overlay(background_audio)
            final_audio.export(output_audio_path, format="webm")

            print(f"Exported {output_audio_path}")

    # print(f"{folder} finished processing")

print("finished")