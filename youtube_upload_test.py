import os
import google.oauth2.credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define the scopes required for the YouTube Data API
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

def get_authenticated_service(client_id):
    """Authenticate and return a YouTube Data API service."""
    flow = InstalledAppFlow.from_client_secrets_file(
        client_id, SCOPES
    )
    credentials = flow.run_console()
    return build("youtube", "v3", credentials=credentials)

def upload_video(youtube, video_file, title, description, tags):
    """Upload a video to YouTube."""
    request_body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": "24"  # 24 corresponds to the 'Entertainment' category
        },
        "status": {
            "privacyStatus": "public",  # Set video privacy to public
            "selfDeclaredMadeForKids": False  # Indicate if the video is made for kids
        },
    }

    # MediaFileUpload object to represent the video
    media = MediaFileUpload(video_file, chunksize=-1, resumable=True)

    # Execute the API request to upload the video
    response = youtube.videos().insert(
        part="snippet,status",
        body=request_body,
        media_body=media
    ).execute()

    print("Video uploaded successfully!")
    print("Video ID:", response["id"])

if __name__ == "__main__":
    # Replace these with your video file path, title, description, and tags
    video_file = "output_videos/170648283627549802329779732480.mp4"
    title = "Superman 🦸"
    description = "Your Short Description"
    tags = ["tag1", "tag2", "tag3"]

    # Load client ID from environment variable
    client_id = os.getenv("CLIENT_ID")

    # Authenticate with YouTube Data API
    youtube = get_authenticated_service(client_id)

    # Upload the video
    upload_video(youtube, video_file, title, description, tags)
