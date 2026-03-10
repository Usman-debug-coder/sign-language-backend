import os
import cv2

# Folder where your sign videos are stored
VIDEO_FOLDER = "videos"

# Mapping gloss words to their corresponding video files
# Add or remove entries based on your downloaded PSL videos
GLOSS_TO_VIDEO = {
    "GIVE": "give.mp4",
    "PEN": "pen.mp4",
    "HELLO": "hello.mp4",
    "HOW": "how.mp4",
    "YOU": "you.mp4",
    "TODAY": "today.mp4",
    "CLASS": "class.mp4",
    "CANCEL": "cancel.mp4",
}


def gloss_to_video_sequence(gloss_text: str):
    """
    Convert gloss string into a list of video file paths.
    Example gloss: "PLEASE GIVE PEN"
    """
    if not gloss_text:
        print("[ERROR] Empty gloss string")
        return []

    words = gloss_text.split()
    video_paths = []

    for word in words:
        if word in GLOSS_TO_VIDEO:
            filename = GLOSS_TO_VIDEO[word]
            full_path = os.path.join(VIDEO_FOLDER, filename)

            if os.path.exists(full_path):
                video_paths.append(full_path)
            else:
                print(f"[WARNING] File missing: {full_path}")
        else:
            print(f"[WARNING] No video mapping for gloss word: {word}")

    return video_paths


def play_video_sequence(video_list):
    """
    Plays only the FIRST HALF of each video using OpenCV.
    """

    if not video_list:
        print("[ERROR] No videos to play.")
        return

    for video_path in video_list:
        print(f"Playing first half of: {video_path}")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {video_path}")
            continue

        # Total frames in the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # We only want to play HALF
        half_frames = total_frames // 2
        current_frame = 0

        while current_frame < half_frames:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow("PSL Gloss Player (Half Video)", frame)

            # Exit if 'q' pressed
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

            current_frame += 1

        cap.release()

    cv2.destroyAllWindows()
