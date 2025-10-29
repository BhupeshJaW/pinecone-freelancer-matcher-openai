# pinecone-freelancer-matcher-openai
AI-powered Freelancer Matcher using OpenAI Embeddings and Pinecone Vector Search.

# 🤖 AI Freelancer Matcher — OpenAI + Pinecone

Find the best freelancers for your project using **AI-powered semantic search**.

This open-source Python project uses **OpenAI embeddings** + **Pinecone vector search** to match freelancer profiles with your query — e.g., *“Python developer available full-time.”*

---

## 🚀 Features
✅ Uses **text-embedding-3-small** for semantic understanding  
✅ Stores and queries data using **Pinecone vector database**  
✅ Automatically creates the index if it doesn’t exist  
✅ Handles **RateLimitError** gracefully  
✅ Provides **human-like explanations** via GPT-4o-mini  

---

## 🧩 Setup

### 1️⃣ Clone the Repo

git clone https://github.com/<your-username>/freelancer-matcher-ai.git
cd freelancer-matcher-ai

### 2️⃣ Install Dependencies
pip install -r requirements.txt

### 3️⃣ Configure Environment

Rename .env.example to .env and fill in your API keys:

OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENV=us-east-1

### 4️⃣ Run the Script
python main.py "Python developer available full-time"

Upserted 4 freelancers into index 'secondactpro'.

Best Freelancers:
- Alice (Skills: Python, Machine Learning, AI, Availability: full-time, Category: tech, Score: 0.98)
- John (Skills: Devops Engineer, Developer, Availability: part-time, Category: Developer, Score: 0.82)

Explanation:
These freelancers were selected because their skill sets and availability closely align with the query “Python developer available full-time.”


🧠 How It Works

Embeds freelancer profiles using OpenAI’s text-embedding-3-small.

Stores those embeddings in Pinecone.

Takes a user query (like “graphic designer part-time”).

Finds semantically closest profiles.

Uses GPT-4o-mini to generate a natural-language explanation.

🤝 Contributing

Pull requests are welcome!
If you’d like to add new datasets, improve matching logic, or integrate a UI, feel free to open an issue.

📜 License

This project is licensed under the MIT License — free for personal and commercial use.

⭐ If you like this project, give it a star!
Let’s make AI-powered hiring open-source!