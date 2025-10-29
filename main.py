import argparse
from openai import OpenAI, RateLimitError  # Import RateLimitError explicitly
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")

# Validate API keys
if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY or PINECONE_API_KEY in .env file")

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Connect to or create Pinecone index
INDEX_NAME = "secondactpro"
INDEX_DIMENSION = 1536  # Matches text-embedding-3-small

if INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating index '{INDEX_NAME}'...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=INDEX_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )
    print("Index creation initiated. Waiting might be needed if first run.")
index = pc.Index(INDEX_NAME)

# Sample freelancer data (manual for testing)
freelancers = [
    {
        "id": "freelancer_1",
        "text": "Alice, expert Python developer with 10 years in AI and machine learning, available full-time.",
        "metadata": {
            "name": "Alice",
            "skills": "Python, Machine Learning, AI",
            "availability": "full-time",
            "category": "tech"
        }
    },
    {
        "id": "freelancer_2",
        "text": "Bob, skilled graphic designer with 8 years in UI/UX, available part-time.",
        "metadata": {
            "name": "Bob",
            "skills": "UI/UX, Graphic Design",
            "availability": "full-time",
            "category": "design"
        }
    },
    {
        "id": "freelancer_3",
        "text": "Charlie, experienced project manager with 12 years in consulting, available part-time.",
        "metadata": {
            "name": "Charlie",
            "skills": "Project Management, Consulting",
            "availability": "part-time",
            "category": "consulting"
        }
    },
    {
        "id": "freelancer_4",
        "text": "John, experienced Devops Engineer with 12 years in Amazon, available part-time.",
        "metadata": {
            "name": "John",
            "skills": "Devops Engineer, Developer",
            "availability": "part-time",
            "category": "Developer"
        }
    }
]

# Generate embeddings and upsert
vectors = []
try:
    for freelancer in freelancers:
        embedding_response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=freelancer["text"]
        )
        vector = {
            "id": freelancer["id"],
            "values": embedding_response.data[0].embedding,
            "metadata": freelancer["metadata"]
        }
        vectors.append(vector)
    index.upsert(vectors=vectors)
    print(f"Upserted {len(vectors)} freelancers into index '{INDEX_NAME}'.")
except RateLimitError as e:  # Use RateLimitError directly
    print(f"Rate limit exceeded during upsert: {str(e)}. Please check your OpenAI quota at https://platform.openai.com.")
    exit(1)
except Exception as e:
    print(f"Error during upsert: {str(e)}")
    exit(1)

# Parse command-line argument for prompt
parser = argparse.ArgumentParser(description="Find best freelancers based on a prompt.")
parser.add_argument("prompt", type=str, help="The prompt to find matching freelancers (e.g., 'Python developer available full-time')")
args = parser.parse_args()

# Query the index for best matches
try:
    embedding_response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=args.prompt
    )
    query_embedding = embedding_response.data[0].embedding

    pinecone_results = index.query(
        vector=query_embedding,
        top_k=3,  # Return top 3 matches
        include_metadata=True
    )

    # Generate explanation
    matches_info = []
    explanation_input = f"You are matching freelancers based on relevance to the query.\n\nQuery: {args.prompt}\n\nResults:\n"
    for match in pinecone_results.matches:
        metadata = match.metadata
        score = round(match.score, 3)
        name = metadata.get("name", "Unnamed Freelancer")
        ID = metadata.get("ID", "id")
        skill = metadata.get("skills", "N/A")
        availability = metadata.get("availability", "Unknown")
        category = metadata.get("category", "General")

        matches_info.append({
            "name": name,
            "skills": skill,
            "availability": availability,
            "category": category,
            "id": ID,
            "score": score
        })

        explanation_input += f"- {name}: {skill}, {availability}, {category} (score: {score})\n"

    explanation_input += "\nExplain in 2-3 short sentences why these freelancers were chosen as the best matches."

    llm_response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": explanation_input}],
        max_tokens=150,
        temperature=0.7
    )

    explanation = llm_response.choices[0].message.content.strip()

    # Output results
    print("\nBest Freelancers:")
    for match in matches_info:
        print(f"- {match['name']} (Skills: {match['skills']}, Availability: {match['availability']}, Category: {match['category']}, Score: {match['score']})")
    print("\nExplanation:")
    print(explanation)

except RateLimitError as e:  # Use RateLimitError directly
    print(f"Rate limit exceeded during query: {str(e)}. Please check your OpenAI quota at https://platform.openai.com.")
except Exception as e:
    print(f"Error during query or explanation generation: {str(e)}")