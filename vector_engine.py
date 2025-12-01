import pandas as pd
import chromadb
import os
import shutil
import time


class FloodVectorDB:
    def __init__(self, river_name):
        self.collection_name = f"{river_name}_patterns"
        # We store the database in a folder named 'chroma_db'
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.window_size = 6  # 6 days of history

    def update_index(self, data_file):
        """
        Reads the CSV, creates sliding windows, and indexes them in batches.
        """
        if not os.path.exists(data_file):
            print(f"⚠️ [ChromaDB] Data file {data_file} not found.")
            return

        print(f"🔄 [ChromaDB] Reading {data_file}...")
        df = pd.read_csv(data_file)

        # We need at least 7 days of data
        if len(df) < self.window_size + 1:
            return

        # 1. Reset Collection
        try:
            self.client.delete_collection(self.collection_name)
        except:
            pass

        collection = self.client.get_or_create_collection(name=self.collection_name)

        ids = []
        embeddings = []
        metadatas = []

        print(f"🔄 [ChromaDB] Processing {len(df)} rows... (This may take a moment)")

        # 2. SLIDING WINDOW LOOP
        for i in range(len(df) - self.window_size):
            # The Pattern (Input): 6 days of water levels
            window_data = df.iloc[i: i + self.window_size]
            vector = window_data['water_level_cm'].astype(float).tolist()

            # The Result (Target): The 7th day
            target_row = df.iloc[i + self.window_size]
            target_level = float(target_row['water_level_cm'])
            target_date = target_row['timestamp']

            # Store in temp lists
            ids.append(f"{self.collection_name}_{i}")
            embeddings.append(vector)
            metadatas.append({
                "date": target_date,
                "next_day_level": target_level
            })

        # 3. BATCH INSERTION (The Fix)
        # We insert in chunks of 2000 to avoid "Batch size greater than max" error
        batch_size = 2000
        total_batches = (len(ids) // batch_size) + 1

        print(f"💾 [ChromaDB] Saving {len(ids)} patterns in {total_batches} batches...")

        for i in range(0, len(ids), batch_size):
            # Slice the lists
            i_end = i + batch_size
            batch_ids = ids[i: i_end]
            batch_embeddings = embeddings[i: i_end]
            batch_metadatas = metadatas[i: i_end]

            if batch_ids:
                collection.add(ids=batch_ids, embeddings=batch_embeddings, metadatas=batch_metadatas)
                # Optional: Print progress for every 10th batch
                if (i // batch_size) % 10 == 0:
                    print(f"   - Saved batch {i // batch_size + 1}/{total_batches}")

        print(f"✅ [ChromaDB] Successfully indexed {len(ids)} patterns for {self.collection_name}")

    def find_match(self, current_data_window):
        """
        Takes the LAST 6 values (live data), searches the DB,
        and returns the most similar historical date.
        """
        try:
            collection = self.client.get_or_create_collection(name=self.collection_name)

            results = collection.query(
                query_embeddings=[current_data_window],
                n_results=1
            )

            if not results['metadatas'] or not results['metadatas'][0]:
                return None

            best_match = results['metadatas'][0][0]
            distance = results['distances'][0][0]

            return {
                "predicted_level": best_match['next_day_level'],
                "historical_date": best_match['date'],
                "similarity_score": distance
            }
        except Exception as e:
            print(f"⚠️ Search Error: {e}")
            return None