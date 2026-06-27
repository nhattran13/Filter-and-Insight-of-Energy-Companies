import asyncio
import json
import time
import csv
from websockets import connect

CSV_FILENAME = "bluesky_stream_data.csv"

async def stream_continuous():
    url = "wss://jetstream1.us-east.bsky.network/subscribe?wantedCollections=app.bsky.feed.post"
    
    program_start = time.time()
    total_duration = 60  

    print(f"--- Starting 60-second continuous stream to {CSV_FILENAME} ---")

    # Open file and write headers
    with open(CSV_FILENAME, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Headline (CID)", "Date_Posted", "Post_Text", "Source_URI", "Author_DID"])
        
        try:
            async with connect(url) as websocket:
                # Loop until 60 seconds have passed
                while (time.time() - program_start) < total_duration:
                    try:
                        # Short timeout allows the 'while' condition to be checked frequently
                        message = await asyncio.wait_for(websocket.recv(), timeout=0.5)
                        data = json.loads(message)
                        
                        if data.get("kind") == "commit" and data.get("commit", {}).get("operation") == "create":
                            commit = data["commit"]
                            record = commit.get("record", {})
                            
                            # Data Extraction
                            cid = commit.get("cid", "N/A")
                            created_at = record.get("createdAt", "N/A")
                            text = record.get("text", "").replace('\n', ' ')
                            uri = commit.get("uri", "N/A")
                            did = data.get("did", "N/A")
                            
                            # Logic to skip empty posts (like images only)
                            if not text:
                                continue

                            # Print and Save
                            print(f"[{int(time.time() - program_start)}s] {text[:80]}...")
                            writer.writerow([cid, created_at, text, uri, did])
                            
                    except asyncio.TimeoutError:
                        # No message received in the last 0.5s, just loop back and check the clock
                        continue
                    except Exception as e:
                        print(f"Inner Error: {e}")
                        break

        except Exception as e:
            print(f"Connection Error: {e}")
    
    print(f"\n--- 60 seconds complete. File saved. ---")