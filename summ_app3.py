import sqlite3
import requests
from time import sleep
import zipfile
import sqlite3
import os
from datetime import datetime
import re 
import hashlib 
from openai import OpenAI
import streamlit as st
from dotenv import load_dotenv  # Import the load_dotenv function

# Load the environment variables from the .env file
load_dotenv()

# Fetch API keys from environment variables
ASSEMBLY_AI_API_KEY = os.getenv("ASSEMBLY_AI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LISTEN_API_KEY = os.getenv("LISTEN_API_KEY")


def get_db_connection():
    """Create and return a connection to the database."""
    conn = sqlite3.connect('podcasts.db')
    return conn

def get_episode_data(episode_id):
    """Fetch episode data including transcription and summary if available."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT audio_url, title, transcription, summary 
            FROM episodes 
            WHERE episode_id = ?
        ''', (episode_id,))
        return cursor.fetchone()

def update_episode_with_transcription_summary(episode_id, transcription, summary):
    """Update episode with new transcription and summary."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE episodes 
            SET transcription = ?, summary = ?
            WHERE episode_id = ?
        ''', (transcription, summary, episode_id))
        conn.commit()


def fetch_data(episode_id):
    """Fetches data based on input type (single episode or playlist).
    
    If the ID has 11 characters, it is always considered a playlist;
    otherwise, it's considered a single episode ID.
    """
    if len(episode_id) == 11:  # Check if the ID has 11 characters
        return retrieve_urls_podcast(episode_id)
    else:
        print(f"Failed to retrieve playlist ID. Please check the playlist ID. It should contain 11 characters")
    

def retrieve_urls_podcast(episode_id):
    """Retrieves podcast URLs for a given playlist ID and updates with new episodes if necessary."""
    episodes_info = []
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT 1 FROM playlists WHERE playlist_id = ?', (episode_id,))
        playlist_exists = cursor.fetchone() is not None

        if playlist_exists:
            cursor.execute('''
                SELECT episodes.episode_id, episodes.audio_url, episodes.title
                FROM episodes
                JOIN playlist_episodes ON episodes.episode_id = playlist_episodes.episode_id
                WHERE playlist_episodes.playlist_id = ?
            ''', (episode_id,))
            episodes_info = cursor.fetchall()

    if not episodes_info:  # Attempt to fetch from external API if no episodes found or playlist doesn't exist
        episodes_info = fetch_playlist_from_api(episode_id, playlist_exists)

    return [(e[0], e[1], e[2]) for e in episodes_info]

def fetch_playlist_from_api(episode_id, playlist_exists):
    """Fetch playlist from external API and update database accordingly."""
    url_playlists_endpoint = 'https://listen-api.listennotes.com/api/v2/playlists'
    headers = {'X-ListenAPI-Key': LISTEN_API_KEY}
    response = requests.get(f"{url_playlists_endpoint}/{episode_id}", headers=headers)
    episodes_info = []
    
    if response.status_code == 200:
        data = response.json()
        with get_db_connection() as conn:
            cursor = conn.cursor()
            if not playlist_exists:
                # Optionally add playlist description from API data if available
                playlist_description = data.get('description', 'New Playlist Description')
                cursor.execute('INSERT INTO playlists (playlist_id, description) VALUES (?, ?)', 
                               (episode_id, playlist_description))
            
            for episode in data['items']:
                episode_data = episode['data']
                audio_url = episode_data.get('audio') or episode_data.get('audio_url')
                if not audio_url:
                    print(f"Error: Missing audio URL for episode ID {episode_data.get('id')}")
                    continue

                cursor.execute('SELECT 1 FROM episodes WHERE episode_id = ?', (episode_data['id'],))
                if cursor.fetchone() is None:
                    cursor.execute('''
                        INSERT INTO episodes (episode_id, title, audio_url) 
                        VALUES (?, ?, ?)
                    ''', (episode_data['id'], episode_data['title'], audio_url))
                
                cursor.execute('''
                    INSERT OR IGNORE INTO playlist_episodes (playlist_id, episode_id) 
                    VALUES (?, ?)
                ''', (episode_id, episode_data['id']))
            
            conn.commit()

            cursor.execute('''
                SELECT episodes.episode_id, episodes.audio_url, episodes.title
                FROM episodes
                JOIN playlist_episodes ON episodes.episode_id = playlist_episodes.episode_id
                WHERE playlist_episodes.playlist_id = ?
            ''', (episode_id,))
            episodes_info = cursor.fetchall()
    else:
        print(f"Failed to retrieve playlist. Please check the playlist ID: {episode_id}")
    
    return episodes_info

def send_transc_request(audio_url):
    """Send transcription request to AssemblyAI and return the transcript ID."""
    transcript_endpoint = "https://api.assemblyai.com/v2/transcript"
    headers = {
        "authorization": ASSEMBLY_AI_API_KEY,  # Your AssemblyAI API key
        "content-type": "application/json"
    }
    response = requests.post(transcript_endpoint, json={"audio_url": audio_url}, headers=headers)
    if response.status_code == 200:
        return response.json()["id"]
    else:
        print(f"Failed to start transcription process. Details: {response.text}")
        return None

def obtain_transcript(transcript_id):
    """Poll AssemblyAI for the transcription result and return the transcription text."""
    polling_endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
    headers = {"authorization": ASSEMBLY_AI_API_KEY}  # Your AssemblyAI API key
    while True:
        response = requests.get(polling_endpoint, headers=headers)
        response_json = response.json()
        if response_json["status"] == 'completed':
            return response_json['text']
        elif response_json["status"] == 'failed':
            print("Transcription failed.")
            return None
        sleep(5)  # To avoid making too frequent requests

def generate_summary_with_openai(transcription, summarization_prompt):
    """Generate a summary for the given transcription text using OpenAI's GPT model."""
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)  # Ensure you've set up OpenAI client
        
        # Check if a specific summarization_prompt is provided, if not, use a generic prompt
        if not summarization_prompt:
            system_prompt = "You are a helpful assistant that provides summaries by first mentioning the title of the podcast. Your objective is to identify emerging and established tech trends, along with valuable insights, to support M&A decision-making among big tech companies seeking to expand their activities through innovation."
        else:
            # If a specific summarization prompt is provided, prepend it to the transcription
            system_prompt = summarization_prompt

        # Create the message payload, incorporating the system_prompt and transcription
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcription}
        ]

        # Make the API call to generate the summary
        response = openai_client.chat.completions.create(
            model="gpt-4-0125-preview",  # Adjust model as necessary
            messages=messages
        )
        
        # Extract and return the summary
        summary = response.choices[0].message.content.strip()
        return summary
    
    except Exception as e:
        print(f"An error occurred while generating summary: {e}")
        return "Error generating summary."


def sanitize_filename(title):
    """Generate a safe filename from a title by hashing it."""
    hash_object = hashlib.sha1(title.encode())
    return hash_object.hexdigest()[:10]

def save_file(content, filename, folder='output'):
    """Save content to a file in the specified folder and return the path."""
    if not os.path.exists(folder):
        os.makedirs(folder)
    file_path = os.path.join(folder, filename)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)
    return file_path

def create_zip_file(file_paths, zip_name, folder='output'):
    """Create a zip file from a list of file paths."""
    if not os.path.exists(folder):
        os.makedirs(folder)
    zip_path = os.path.join(folder, zip_name)
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file_path in file_paths:
            zipf.write(file_path, arcname=os.path.basename(file_path))
    return zip_path

def concatenate_texts(texts, separator="\n\n"):
    """Concatenate a list of texts with a specified separator."""
    return separator.join(texts)




def main():
    st.markdown('# **Podcast Summarizer App**')
    st.sidebar.header('Input Parameters')

    with st.sidebar.form(key='transcription_form'):
        episode_id_to_transcribe = st.text_input('Playlist ID for Transcription:', key='trans_id')
        submit_transcription = st.form_submit_button('Transcribe and Summarize')

    with st.sidebar.form(key='summary_form'):
        episode_id_to_summarize = st.text_input('Playlist ID for New Summary:', key='sum_id')
        new_summary_prompt = st.text_input('Enter new summarization prompt:', key='new_sum_prompt')
        submit_summary = st.form_submit_button('Generate New Summary')
    
    # Removed explicit generic summarization prompt for transcription
    if submit_transcription:
        process_transcription_and_summary(episode_id_to_transcribe, None)  # Passing None to use the default prompt

    if submit_summary:
        generate_new_summary_for_existing_transcript(episode_id_to_summarize, new_summary_prompt)


def process_transcription_and_summary(episode_id, summarization_prompt):
    episodes_info = fetch_data(episode_id)
    if not episodes_info:
        st.error("No episodes found. Please check the ID and try again.")
        return
    
    transcript_texts, summary_texts = [], []
    progress_bar = st.progress(0)  # Initialize progress bar
    
    for index, episode_info in enumerate(episodes_info):
        episode_id, audio_url, title = episode_info
        unique_key_base = f"{episode_id}_{index}"
        st.write(f"### Processing Episode: {title}")
        
        episode_data = get_episode_data(episode_id)
        if episode_data and episode_data[2] and episode_data[3]:
            transcription, summary = episode_data[2], episode_data[3]
        else:
            transcript_id = send_transc_request(audio_url)
            transcription = "Transcription request failed."
            summary = "Summary generation skipped."
            if transcript_id:
                transcription = obtain_transcript(transcript_id) or "Transcription failed."
                if transcription != "Transcription failed.":
                    summary = generate_summary_with_openai(transcription, summarization_prompt) or "Summary generation failed."
                    update_episode_with_transcription_summary(episode_id, transcription, summary)
        
        st.text_area("Transcription", transcription, height=150, key=f"trans_{unique_key_base}")
        st.text_area("Summary", summary, height=150, key=f"sum_{unique_key_base}")
        
        transcript_texts.append(transcription)
        summary_texts.append(summary)
        
        progress_bar.progress((index + 1) / len(episodes_info))
    
    final_transcription = "\n\n".join(transcript_texts)
    final_summary = "\n\n".join(summary_texts)
    
    save_and_download(final_transcription, "all_transcripts.txt")
    save_and_download(final_summary, "all_summaries.txt")

def generate_new_summary_for_existing_transcript(episode_id, new_summary_prompt):
    episodes_info = fetch_data(episode_id)
    if not episodes_info:
        st.error("No episodes found. Please check the ID and try again.")
        return
    
    summary_texts = []
    progress_bar = st.progress(0)
    
    for index, episode_info in enumerate(episodes_info):
        episode_id, _, title = episode_info
        st.write(f"### Generating New Summary for: {title}")
        
        episode_data = get_episode_data(episode_id)
        if not episode_data or not episode_data[2]:
            st.warning(f"Transcription not available for episode: {title}. Skipping.")
            continue
        
        new_summary = generate_summary_with_openai(episode_data[2], new_summary_prompt)
        st.text_area("New Summary", new_summary, height=150, key=f"new_sum_{episode_id}_{index}")
        
        summary_texts.append(new_summary)
        progress_bar.progress((index + 1) / len(episodes_info))
    
    if summary_texts:
        final_summary = "\n\n".join(summary_texts)
        save_and_download(final_summary, "new_all_summaries.txt")

def save_and_download(content, filename):
    """Generate a download link for the saved content."""
    file_path = save_file(content, filename)
    with open(file_path, "rb") as fp:
        st.download_button(label=f"Download {filename}", data=fp, file_name=filename, mime="text/plain")

if __name__ == "__main__":
    main()