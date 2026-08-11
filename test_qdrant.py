import asyncio

from app.vectorstore.qdrant_client import client


async def test_connection():
    try:
        # Fetch existing collections (should be empty right now)
        collections = await client.get_collections()
        print("✅ Successfully connected to Qdrant!")
        print(f"Collections: {collections.collections}")
    except (ConnectionError, RuntimeError) as e:
        print(f"❌ Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_connection())