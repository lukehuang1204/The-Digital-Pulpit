import streamlit as st
import yt_dlp
import tempfile
import os
import re
import subprocess
from openai import OpenAI
from pydub import AudioSegment # Import pydub

# Word Cloud Imports
import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # Use Agg backend for Streamlit
from io import BytesIO

# --- Scripture Function Schema (for OpenAI function calling) ---
scripture_function = {
    "name": "detect_scriptures",
    "description": "從中文講道逐字稿中擷取所有聖經經文引用（含諧音、轉述）",
    "parameters": {
        "type": "object",
        "properties": {
            "references": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "book": {
                            "type": "string",
                            "description": "中文書卷名（和合本次序）",
                            "enum": [
                                "創世記","出埃及記","利未記","民數記","申命記",
                                "約書亞記","士師記","路得記","撒母耳記上","撒母耳記下",
                                "列王紀上","列王紀下","歷代志上","歷代志下","以斯拉記",
                                "尼希米記","以斯帖記","約伯記","詩篇","箴言",
                                "傳道書","雅歌","以賽亞書","耶利米書","耶利米哀歌",
                                "以西結書","但以理書","何西阿書","約珥書","阿摩司書",
                                "俄巴底亞書","約拿書","彌迦書","那鴻書","哈巴谷書",
                                "西番雅書","哈該書","撒迦利亞書","瑪拉基書",
                                "馬太福音","馬可福音","路加福音","約翰福音","使徒行傳",
                                "羅馬書","哥林多前書","哥林多後書","加拉太書","以弗所書",
                                "腓立比書","歌羅西書","帖撒羅尼迦前書","帖撒羅尼迦後書","提摩太前書",
                                "提摩太後書","提多書","腓利門書","希伯來書","雅各書",
                                "彼得前書","彼得後書","約翰一書","約翰二書","約翰三書",
                                "猶大書","啟示錄"
                            ]
                        },
                        "chapter":  {"type": "integer",  "minimum": 1},
                        "verse_start": {"type": "integer", "minimum": 1},
                        "verse_end":   {"type": "integer", "minimum": 1},
                        "detected_text":  {"type": "string"},
                        "context":        {"type": "string"},
                        "mention_count":  {"type": "integer", "minimum": 1},
                        "confidence":     {"type": "number",  "minimum": 0, "maximum": 1}
                    },
                    "required": [
                        "book","chapter",
                        "verse_start","detected_text",
                        "mention_count","confidence"
                    ]
                }
            }
        },
        "required": ["references"]
    }
}

# Initialize OpenAI client
openai_api_key = st.secrets.get("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

st.set_page_config(page_title="YouTube Transcriber", layout="centered")

st.title("華人教會講道語料轉錄與分析")

# Create tabs for different functionality
tab1, tab2 = st.tabs(["YouTube Transcription", "Text Input Word Cloud"])

# Function to generate word cloud from text
def generate_wordcloud(text, title=""):
    # Skip if text is empty or only whitespace
    if not text or text.strip() == "":
        st.warning("Text is empty. Cannot generate word cloud.")
        return None
    
    # Chinese word segmentation using jieba
    st.info("Segmenting Chinese text...")
    words = jieba.lcut(text)
    
    # Remove stopwords and short words
    stopwords = ["的", "了", "是", "在", "我們", " ", "\n", "，", "。", "、", 
                "！", "？", "這個", "那個", "一個", "嗎", "呢", "吧", "啊", 
                "哦", "嗯", "就是", "所以", "因為", "然後", "他們", "你們", 
                "自己", "知道", "今天", "事情", "時候", "什麼", "這樣", "那樣", 
                "可能", "覺得", "比較", "非常", "這個", "那個", "我", "你", 
                "他", "她", "它", "人", "說", "有", "會", "要", "看", "來", 
                "去", "做", "想", "和", "與", "跟", "但", "不", "也", "都", 
                "就", "才", "弟兄", "姐妹", "大家"]
    
    filtered_words = [word for word in words 
                    if word not in stopwords 
                    and len(word) > 1 
                    and not word.isdigit()]
    
    # Join words with space for WordCloud
    processed_text = ' '.join(filtered_words)
    
    # Try to find a suitable font for Chinese text
    # First, check for NotoSansTC-Regular.ttf in the current directory (repo root)
    custom_font = "NotoSansTC-Regular.ttf"
    font_path = None
    if os.path.exists(custom_font):
        font_path = custom_font
    else:
        font_paths = [
            '/System/Library/Fonts/STHeiti Medium.ttc',  # macOS
            '/System/Library/Fonts/PingFang.ttc',  # macOS
            'C:/Windows/Fonts/simhei.ttf',  # Windows
            'C:/Windows/Fonts/msyh.ttc',  # Windows
            '/usr/share/fonts/truetype/arphic/uming.ttc',  # Linux
            '/usr/share/fonts/truetype/droid/DroidSansFallback.ttf'  # Linux
        ]
        for path in font_paths:
            if os.path.exists(path):
                font_path = path
                break
    
    if not font_path:
        st.warning("No suitable Chinese font found. Word cloud may not display Chinese characters correctly.")
        # Try to proceed without specifying font
        wc = WordCloud(
            width=800,
            height=600,
            background_color='white',
            max_words=200,
            collocations=False  # Avoid duplicate phrases
        ).generate(processed_text)
    else:
        # Generate word cloud with specified font
        wc = WordCloud(
            font_path=font_path,
            width=800,
            height=600,
            background_color='white',
            max_words=200,
            collocations=False  # Avoid duplicate phrases
        ).generate(processed_text)
    
    # Create figure for display
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout()
    
    # Return the figure and word cloud object
    return fig, wc

# YouTube Transcription Tab
with tab1:
    # Function to validate YouTube URL and extract video ID
    def extract_video_id(url):
        # Regular expression to extract YouTube video ID
        patterns = [
            r'^https?://(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',
            r'^https?://(?:www\.)?youtu\.be/([a-zA-Z0-9_-]{11})',
            r'^https?://(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})'
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    # Input field for YouTube URL
    youtube_url = st.text_input("YouTube Video URL")

    # Example URL suggestion
    st.caption("Example: https://www.youtube.com/watch?v=FEfdf7D9WSg")

    # Model selection 
    st.subheader("Transcription Model")
    transcription_model = st.radio(
        "Select transcription model:",
        ["whisper-1"],
        index=0,
        help="Whisper is better for non-English content."
    )

    # Transcribe button
    if st.button("Transcribe"):
        if not youtube_url:
            st.error("Please enter a YouTube video URL.")
        else:
            video_id = extract_video_id(youtube_url)
            if not video_id:
                st.error("Invalid YouTube URL. Please enter a valid YouTube URL.")
            else:
                temp_dir = None
                audio_path = None
                try:
                    # Create progress bars for the whole process
                    download_progress = st.progress(0)
                    st.write("Downloading and preparing audio...")
                    
                    with st.spinner("Processing..."):
                        # Update progress bar at key points
                        download_progress.progress(10, text="Initializing...")
                        try:
                            # Create temp directory
                            temp_dir = tempfile.mkdtemp()


                            # File size limit for API uploads
                            LIMIT = 24 * 1024 * 1024  # 24MB safety under typical cap
                            BITRATE_BPS = 48_000
                            BYTES_PER_SEC = BITRATE_BPS // 8
                            max_secs_per_part = int(LIMIT // BYTES_PER_SEC) - 10  # margin

                            # download at a low bitrate so files fit under 24 MB
                            ydl_opts = {
                                "format": "bestaudio[abr<=48]/bestaudio",
                                "outtmpl": os.path.join(temp_dir, f"{video_id}.%(ext)s"),
                                "postprocessors": [
                                    {
                                        "key": "FFmpegExtractAudio",
                                        "preferredcodec": "mp3",
                                        "preferredquality": "48",   # 48 kbits ≈ 360 MB/h
                                    }
                                ],
                                "quiet": True,
                                "no_warnings": True,
                            }

                            # Get video info first
                            with yt_dlp.YoutubeDL({'quiet': True, 'no_warnings': True}) as ydl:
                                download_progress.progress(20, text="Fetching video metadata...")
                                info = ydl.extract_info(youtube_url, download=False)
                                video_title = info.get('title', f"Video {video_id}")
                                video_duration = info.get('duration')          # seconds
                                st.info(f"Found video: {video_title} ({video_duration/60:.1f} min)")

                            # Download the video audio
                            download_progress.progress(30, text="Downloading audio...")
                            download_info = None
                            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                                download_info = ydl.extract_info(youtube_url, download=True)

                            # Find the actual downloaded file path
                            # yt-dlp returns info about the downloaded file
                            if download_info and 'requested_downloads' in download_info and download_info['requested_downloads']:
                                 # Get the path of the first downloaded file
                                audio_path = download_info['requested_downloads'][0]['filepath']
                            elif download_info and '_format_sort_fields' in download_info:
                                 # Fallback: try to guess the path based on outtmpl and actual extension
                                 actual_ext = download_info.get('ext', 'mp3') # Guess mp3 if not found
                                 audio_path = os.path.join(temp_dir, f"{video_id}.{actual_ext}")
                                 if not os.path.exists(audio_path):
                                     # Last resort: find any file in the temp directory
                                     files = os.listdir(temp_dir)
                                     if files:
                                         audio_path = os.path.join(temp_dir, files[0])
                                     else:
                                         st.error("Failed to locate the downloaded audio file.")
                                         st.stop()
                            else:
                                st.error("Failed to get download information from yt-dlp.")
                                st.stop()

                            # Check if file exists
                            if not os.path.exists(audio_path):
                                st.error("Failed to download the audio file.")
                                st.stop()

                            # Update progress
                            download_progress.progress(60, text="Processing audio file...")
                            
                            # Check file size after download and split if needed
                            file_size = os.path.getsize(audio_path)
                            
                            if file_size > LIMIT:
                                download_progress.progress(70, text="Splitting audio into segments...")
                                st.warning(
                                    f"Audio file is {file_size / (1024 * 1024):.1f} MB (> {LIMIT / (1024 * 1024):.1f} MB). "
                                    f"Splitting into chunks for API upload."
                                )
                                
                                audio = AudioSegment.from_file(audio_path)
                                parts = []
                                for start in range(0, len(audio), max_secs_per_part * 1000):
                                    seg = audio[start:start + max_secs_per_part * 1000]
                                    p = os.path.join(temp_dir, f"{video_id}_{start//1000:06d}.mp3")
                                    seg.export(p, format="mp3", bitrate="48k")
                                    parts.append(p)
                                audio_segments_to_process = parts or [audio_path]
                                
                                # Update progress
                                download_progress.progress(85, text=f"Split into {len(audio_segments_to_process)} segments")
                            else:
                                # File is small enough, no splitting needed
                                download_progress.progress(80, text="No splitting needed, preparing audio...")
                                audio_segments_to_process = [audio_path]
                            
                            # Update progress after download and processing
                            download_progress.progress(100, text="Audio prepared successfully")

                            # Get file extension of the file(s) to be sent to OpenAI
                            # Check if the file is in a supported format (mp3, mp4, mpeg, mpga, m4a, wav, webm)
                            supported_formats = ['.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm']

                            # Check the first segment's extension (assuming all chunks have the same format)
                            if audio_segments_to_process:
                                file_ext_to_send = os.path.splitext(audio_segments_to_process[0])[1].lower()
                                if file_ext_to_send not in supported_formats:
                                    st.warning(f"Audio file format '{file_ext_to_send}' may not be supported by OpenAI. Supported formats: {', '.join(supported_formats)}")
                                    # Note: We already tried converting to mp3 if ffmpeg was available during download.
                                    # If it's still not a supported format here, we proceed anyway and hope OpenAI can handle it,
                                    # or the fallback to whisper might work better.

                            st.success(f"Successfully downloaded and prepared audio ({os.path.getsize(audio_segments_to_process[0]) / (1024 * 1024):.1f}MB per chunk if split)") # Print final file size

                        except Exception as e:
                            st.error(f"Error downloading or preparing audio: {str(e)}") # More specific error message
                            import traceback
                            st.code(traceback.format_exc(), language="python")
                            st.stop()

                    # Create a new progress bar for transcription
                    transcription_progress = st.progress(0)
                    st.write("Transcribing audio...")
                    
                    full_transcript_text = ""
                    for i, segment_path in enumerate(audio_segments_to_process):
                        # Calculate progress percentage for current segment
                        segment_progress = int((i / len(audio_segments_to_process)) * 100)
                        transcription_progress.progress(segment_progress, text=f"Transcribing segment {i+1}/{len(audio_segments_to_process)}")
                        
                        with st.spinner(f"Processing segment {i+1}/{len(audio_segments_to_process)}"):
                            try:
                                # Open audio file as binary
                                with open(segment_path, "rb") as audio_file: 
                                    # Check if we should use the GPT-4o models or Whisper-1
                                    if transcription_model != "whisper-1":
                                        try:
                                            st.info(f"Attempting transcription of segment {i+1} with {transcription_model} model...")
                                            transcription = client.audio.transcriptions.create(
                                                model=transcription_model,
                                                file=audio_file,
                                                response_format="text"
                                            )
                                            model_used = transcription_model
                                        except Exception as model_error:
                                            st.warning(f"{transcription_model} model failed for segment {i+1}: {str(model_error)}")
                                            st.info(f"Falling back to Whisper model for segment {i+1}...")
                                            # Need to reopen the file as the previous operation consumed it
                                            audio_file.seek(0)
                                            transcription = client.audio.transcriptions.create(
                                                model="whisper-1",
                                                file=audio_file,
                                                response_format="text"
                                            )
                                            model_used = "whisper-1"
                                    else:
                                        st.info(f"Attempting transcription of segment {i+1} with Whisper model...")
                                        transcription = client.audio.transcriptions.create(
                                            model="whisper-1",
                                            file=audio_file,
                                            response_format="text"
                                        )
                                        model_used = "whisper-1"

                                # Since response_format is "text", the response is directly the transcript text
                                transcript_text = transcription
                                full_transcript_text += transcript_text + "\n" # Append transcript with a newline
                                # Update progress for completed segment
                                segment_progress = int(((i + 1) / len(audio_segments_to_process)) * 100)
                                transcription_progress.progress(segment_progress, text=f"Completed segment {i+1}/{len(audio_segments_to_process)}")
                                st.success(f"Transcription of segment {i+1} completed using {model_used} model")

                            except Exception as e:
                                st.error(f"Error during transcription of segment {i+1}: {str(e)}")
                                # If any segment fails, stop processing and report the error
                                full_transcript_text += f"\n[Transcription failed for segment {i+1}: {str(e)}]\n"
                                break # Stop processing further segments

                    # Display the full transcript
                    if full_transcript_text:
                        st.subheader("Transcript")
                        st.text_area("Transcript", full_transcript_text, height=400, label_visibility="collapsed")

                        # Add download button for the transcript
                        st.download_button(
                            label="Download Transcript",
                            data=full_transcript_text,
                            file_name=f"{video_title}_transcript.txt",
                            mime="text/plain"
                        )

                        # --- Scripture Detection Section (Function Calling) ---
                        st.subheader("經文引用偵測 (Scripture Reference Detection)")
                        rerun_scripture = st.button("🔄 重新偵測")
                        if rerun_scripture or True:
                            with st.spinner("Detecting scripture references using OpenAI function calling..."):
                                try:
                                    import json
                                    # Prepare system and user prompts
                                    system_prompt = (
                                        "你是一個聖經經文偵測助手。規則：\n"
                                        "1. 只處理新舊約 66 卷。\n"
                                        "2. 中文書名請用和合本標準名稱。\n"
                                        "3. 諧音、縮寫也算引用。模糊時以最高可能性的書卷回傳。\n"
                                        "4. 不要省略缺失的章節或節號。若無法確定，省略該筆而非猜測。\n"
                                        "5. 回傳時必須以 detect_scriptures 函數形式輸出；禁止任何額外文字。\n"
                                        "6. 如果僅僅只是使用聖經的典故，則不算作引用經文"
                                    )
                                    user_prompt = (
                                        "以下是中文講道逐字稿，請依規則偵測經文引用：\n"
                                        "\"\"\"\n"
                                        f"{full_transcript_text.strip()}\n"
                                        "\"\"\"\n"
                                        "同一段經文若在稿中多次出現，只輸出一次並以 mention_count 累加。"
                                    )
                                    # Use OpenAI Chat Completions API with function calling
                                    response = client.chat.completions.create(
                                        model="gpt-4-1106-preview",
                                        messages=[
                                            {"role": "system", "content": system_prompt},
                                            {"role": "user", "content": user_prompt}
                                        ],
                                        functions=[scripture_function],
                                        function_call="auto"
                                    )
                                    # Parse function call results
                                    scripture_results = []
                                    # The OpenAI chat completions API returns choices with message/function_call
                                    for choice in response.choices:
                                        message = choice.message
                                        if (
                                            hasattr(message, "function_call")
                                            and message.function_call is not None
                                            and getattr(message.function_call, "name", None) == "detect_scriptures"
                                        ):
                                            try:
                                                args = message.function_call.arguments
                                                if isinstance(args, str):
                                                    args = json.loads(args)
                                                scripture_results.append(args)
                                            except Exception:
                                                pass
                                    if scripture_results:
                                        st.json(scripture_results, expanded=True)
                                        st.download_button(
                                            label="Download Scripture References (JSON)",
                                            data=json.dumps(scripture_results, ensure_ascii=False, indent=2),
                                            file_name=f"{video_title}_scripture_references.json",
                                            mime="application/json"
                                        )
                                    else:
                                        st.warning("No valid scripture references detected or failed to parse function call output.")
                                except Exception as e:
                                    st.error(f"Error detecting scripture references: {str(e)}")
                                    import traceback
                                    st.code(traceback.format_exc(), language="python")

                        # Generate and display word cloud from transcript
                        st.subheader("Word Cloud Analysis")
                        
                        with st.spinner("Generating word cloud..."):
                            try:
                                
                                # Generate word cloud using the transcript
                                result = generate_wordcloud(full_transcript_text, video_title)
                                
                                if result:
                                    fig, wc = result
                                    # Display word cloud
                                    st.pyplot(fig)
                                    
                                    # Create download button for the word cloud image
                                    img_buf = BytesIO()
                                    wc.to_image().save(img_buf, format='PNG')
                                    img_buf.seek(0)
                                    
                                    st.download_button(
                                        label="Download Word Cloud",
                                        data=img_buf,
                                        file_name=f"{video_title}_wordcloud.png",
                                        mime="image/png"
                                    )
                                    img_buf.close()
                            except Exception as e:
                                st.error(f"Error generating word cloud: {str(e)}")
                                import traceback
                                st.code(traceback.format_exc(), language="python")
                    else:
                        st.warning("No transcript generated.")

                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc(), language="python")
                finally:
                    # Clean up temporary files and directory
                    if temp_dir and os.path.exists(temp_dir):
                        try:
                            # Remove all files in the temp directory first
                            for file_name in os.listdir(temp_dir):
                                file_path = os.path.join(temp_dir, file_name)
                                if os.path.isfile(file_path):
                                    os.remove(file_path)
                            # Then remove the directory
                            os.rmdir(temp_dir)
                        except:
                            pass # Ignore cleanup errors

# Text Input Word Cloud Tab
with tab2:
    st.subheader("Generate Word Cloud from Text")
    
    # Text input area
    user_text = st.text_area("Enter or paste Chinese text:", height=300, 
                            help="Paste Chinese text here to generate a word cloud.")
    
    # Optional file upload
    uploaded_file = st.file_uploader("Or upload a text file:", type=["txt"], 
                                   help="Upload a .txt file with Chinese text.")
    
    if uploaded_file is not None:
        # Read the uploaded file
        file_text = uploaded_file.read().decode("utf-8")
        if file_text:
            # Show the text in the text area
            st.text_area("File content:", file_text, height=200)
            # Use the file text instead of user input
            user_text = file_text
    
    # Generate button
    if st.button("Generate Word Cloud"):
        if not user_text or user_text.strip() == "":
            st.error("Please enter text or upload a file.")
        else:
            with st.spinner("Generating word cloud..."):
                try:
                    # Generate word cloud
                    result = generate_wordcloud(user_text)
                    
                    if result:
                        fig, wc = result
                        # Display word cloud
                        st.pyplot(fig)
                        
                        # Create download button for the word cloud image
                        img_buf = BytesIO()
                        wc.to_image().save(img_buf, format='PNG')
                        img_buf.seek(0)
                        
                        st.download_button(
                            label="Download Word Cloud",
                            data=img_buf,
                            file_name="wordcloud.png",
                            mime="image/png"
                        )
                        img_buf.close()
                except Exception as e:
                    st.error(f"Error generating word cloud: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc(), language="python")
