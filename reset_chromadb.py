# reset_chromadb.py
import os
import shutil

def reset_chromadb():
    """Safely reset all ChromaDB instances"""
    
    # Define all ChromaDB paths
    chroma_paths = [
        "chatbot_project/chroma_db",
        "chatbot_project/qa_bot/chroma_db", 
        "chroma_db"
    ]
    
    print("🧹 Cleaning ChromaDB instances...")
    
    for path in chroma_paths:
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
                print(f"✅ Deleted: {path}")
            except Exception as e:
                print(f"❌ Error deleting {path}: {e}")
        else:
            print(f"⚠️  Path not found: {path}")
    
    # Create fresh ChromaDB directory
    os.makedirs("chatbot_project/chroma_db", exist_ok=True)
    print("✅ Created fresh ChromaDB directory")
    
    print("🎉 ChromaDB reset complete!")

if __name__ == "__main__":
    reset_chromadb()
