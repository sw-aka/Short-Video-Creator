from pytube import YouTube

def download_video(url, resolution='1280x720'):
    try:
        # Create a YouTube object
        yt = YouTube(url)

        # print(yt.streams)
        # for stream in yt.streams:
        #     print(stream)
        # return
        # Get the video stream with the specified resolution
        video_stream = yt.streams.filter(res='720p', mime_type='video/mp4').first()

        if video_stream is None:
            print('No stream matching the requiements.')
        # Download the video
        print(f'Downloading video: {yt.title} ({resolution})')
        video_stream.download()

        print('Download completed!')
    except Exception as e:
        print(f'An error occurred: {e}')

if __name__ == "__main__":
    # Example URL
    video_url = 'https://www.youtube.com/watch?v=VIDEO_ID'

    # Call the download_video function with the desired resolution
    download_video(video_url, resolution='1280x720')
