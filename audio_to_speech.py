import whisper
import os
from gloss_converter import to_gloss
from gloss_to_video import gloss_to_video_sequence, play_video_sequence
from keypoint_extractor import extract_keypoints
print("File exists:", os.path.exists("testcase2.mp3"))

model = whisper.load_model("medium")
result = model.transcribe("testcase1.mp3",  task="transcribe", language="en", verbose=True)
print("Final text output:")
print(result["text"])

gloss = to_gloss(result.get("text", ""))
print("Final gloss output:")
print(gloss)




video_list = gloss_to_video_sequence(gloss)
print("Video sequence:", video_list)

if video_list:
    keypoint_files = extract_keypoints(
        video_list,
        output_dir="keypoints",
    )
    print("Keypoint data saved at:", keypoint_files)

play_video_sequence(video_list)
