import streamlit as st
import numpy as np
import librosa.display
import cv2
import matplotlib.pyplot as plt
import os
import time
import tempfile
from moviepy.editor import *
from moviepy.config import change_settings
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
from pydub import AudioSegment
import logging
from dotenv import load_dotenv
from streamlit_oauth import OAuth2Component
load_dotenv()

AUTHORIZE_URL = os.environ.get('AUTHORIZE_URL')
TOKEN_URL = os.environ.get('TOKEN_URL')
REFRESH_TOKEN_URL = os.environ.get('REFRESH_TOKEN_URL')
REVOKE_TOKEN_URL = os.environ.get('REVOKE_TOKEN_URL')
CLIENT_ID = os.environ.get('CLIENT_ID')
CLIENT_SECRET = os.environ.get('CLIENT_SECRET')
REDIRECT_URI = os.environ.get('REDIRECT_URI')
SCOPE = os.environ.get('SCOPE')

change_settings({"FFMPEG_BINARY": "./ffmpeg"})


class StreamlitLogger(logging.Logger):
    def __init__(self, progress_bar, duration):
        super().__init__(name="StreamlitLogger", level=logging.DEBUG)
        self.progress_bar = progress_bar
        self.duration = duration

    def debug(self, msg, *args, **kwargs):
        # Extracting progress percentage from moviepy's logging messages
        if "chunk" in msg:
            current_time = float(msg.split(" ")[1].replace("s", ""))
            progress_percentage = int((current_time / self.duration) * 100)
            self.progress_bar.progress(progress_percentage)


def generate_spectrogram(y, sr):
    plt.figure(figsize=(10, 4))

    # Use provided parameters
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=4096, hop_length=2048)), ref=np.max)

    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', fmax=sr / 2)
    plt.axis('off')  # Turn off axis
    plt.savefig("spectrogram.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    return "spectrogram.png"


def create_video(y, sr, spectrogram_path, audio_path, progress_bar=None):
    duration = librosa.get_duration(y=y, sr=sr)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    spectrogram = cv2.imread(spectrogram_path)
    height, width, _ = spectrogram.shape

    out = cv2.VideoWriter('video_no_audio.mp4', fourcc, 20.0, (width, height))

    for i in np.linspace(0, duration, int(duration * 20)):  # Assuming 20 fps
        frame = spectrogram.copy()
        position = int(i / duration * width)
        cv2.line(frame, (position, 0), (position, height), (0, 0, 255), 2)
        out.write(frame)

    out.release()

    # Convert .m4a audio to .mp3 using pydub
    audio = AudioSegment.from_file(audio_path, format="m4a")
    audio.export("temp_audio.mp3", format="mp3")

    # Custom logger for moviepy
    logger = StreamlitLogger(progress_bar, duration)

    logging.getLogger("moviepy").setLevel(logging.DEBUG)
    logging.getLogger("moviepy").handlers = [logger]  # Set the custom logger

    # Combine video and audio using moviepy
    video_clip = VideoFileClip('video_no_audio.mp4')
    audio_clip = AudioFileClip("temp_audio.mp3")
    final_video = video_clip.set_audio(audio_clip)
    final_video.write_videofile("output.mp4", codec='libx264', audio_codec='aac')

    # Clean up temporary files
    os.remove('video_no_audio.mp4')
    os.remove('temp_audio.mp3')

    return "output.mp4"

def upload_to_youtube(video_file, title, description, progress_bar):
    # Use the OAuth 2.0 token stored in st.session_state.credentials
    youtube = build('youtube', 'v3', credentials=st.session_state.credentials)

    body = {
        'snippet': {
            'title': title,
            'description': description,
            'categoryId': '22'  # "People & Blogs" category
        },
        'status': {
            'privacyStatus': 'unlisted'  # or 'private' or 'public'
        }
    }

    media = MediaFileUpload(video_file, chunksize=-1, resumable=True, mimetype='video/mp4')

    request = youtube.videos().insert(part=','.join(body.keys()), body=body, media_body=media)
    response = None

    while response is None:
        status, response = request.next_chunk()
        if status:
            progress = int(status.progress() * 100)
            progress_bar.progress(progress)

    return response


st.title("M4A to MP4 Spectrogram Converter")

# Create OAuth2Component instance
oauth2 = OAuth2Component(CLIENT_ID, CLIENT_SECRET, AUTHORIZE_URL, TOKEN_URL, REFRESH_TOKEN_URL, REVOKE_TOKEN_URL)

# Check if token exists in session state
if 'token' not in st.session_state:
    # If not, show authorize button
    result = oauth2.authorize_button("Authorize", REDIRECT_URI, SCOPE)
    if result and 'token' in result:
        # If authorization successful, save token in session state
        st.session_state.token = result.get('token')
        # Convert token into Credentials object
        creds = Credentials(
            token=st.session_state.token.get("access_token"),
            refresh_token=st.session_state.token.get("refresh_token"),
            token_uri=TOKEN_URL,
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            scopes=[SCOPE]
        )
        st.session_state.credentials = creds
        st.experimental_rerun()
else:
    # If token exists in session state, show the token
    token = st.session_state['token']
    st.json(token)
    if st.button("Refresh Token"):
        # If refresh token button is clicked, refresh the token
        token = oauth2.refresh_token(token)

        st.session_state.token = token
        # Convert the refreshed token into Credentials object
        creds = Credentials(
            token=st.session_state.token.get("access_token"),
            refresh_token=st.session_state.token.get("refresh_token"),
            token_uri=TOKEN_URL,
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            scopes=[SCOPE]
        )
        st.session_state.credentials = creds
        st.experimental_rerun()

uploaded_file = st.file_uploader("Choose an m4a file...", type="m4a")


if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".m4a")
    temp_file.write(uploaded_file.read())
    temp_path = temp_file.name
    temp_file.close()

    # Load m4a file directly using librosa with audioread backend
    y, sr = librosa.load(temp_path, sr=None)

    # Generate Spectrogram
    spectrogram_path = generate_spectrogram(y, sr)

    # Extracting file creation date and filename for title and description
    created_at = time.ctime(os.path.getctime(temp_path))  # Get file's "created at" date
    file_name = os.path.basename(uploaded_file.name).replace('.m4a', '')  # Extract filename without extension
    video_title = f"{created_at} rehearsal at {file_name}"
    video_description = video_title

    # Create video
    with st.spinner("Processing video..."):
        progress_bar = st.progress(0)  # Initialize progress bar
        create_video_path = create_video(y, sr, spectrogram_path, temp_path, progress_bar)
        st.success(f"Video created!")
        st.video(create_video_path)

    # Upload to YouTube
    with st.spinner("Uploading to YouTube..."):
        progress_bar = st.progress(0)
        try:
            response = upload_to_youtube(create_video_path, video_title, video_description, progress_bar)
            st.success(f"Video uploaded! [View on YouTube](https://www.youtube.com/watch?v={response['id']})")
        except HttpError as e:
            st.error(f"An error occurred: {e}")

    # Clean up the temporary file
    os.remove(temp_path)
    os.remove("output.mp4")
    os.remove("spectrogram.png")