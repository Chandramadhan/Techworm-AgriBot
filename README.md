# 🌾 Agri Bot (Multilingual) 🌾

Agri Bot is an AI-powered chatbot designed to assist farmers and agricultural enthusiasts by providing accurate and multilingual farming-related information. The chatbot leverages large language models and online search tools to fetch real-time data on agricultural topics.

## 🚀 Features

- **Multilingual Support**: Supports multiple languages including English, Hindi, Telugu, Tamil, Bengali, Marathi, and Punjabi.
- **AI-Powered Conversations**: Uses Llama 3-70B to provide intelligent and contextual responses.
- **Real-Time Information Retrieval**: Integrates Wikipedia, Arxiv, and DuckDuckGo search for the latest data.
- **Context-Aware Memory**: Remembers previous interactions for a seamless user experience.
- **User-Friendly Interface**: Built with Streamlit for an intuitive chat experience.

## 🛠️ Tech Stack

- **Frontend**: Streamlit (Python)
- **Backend**: LangChain, OpenAI LLM (via Groq API)
- **Search Tools**: Wikipedia, Arxiv, DuckDuckGo
- **Translation**: Google Translator API
- **Memory Management**: LangChain ConversationBufferMemory

## 📌 Prerequisites

- Python 3.8+
- An OpenAI-compatible API key (Groq API)
- Required Python libraries:
  ```bash
  pip install streamlit langchain openai langdetect deep-translator dotenv
  ```

## 🔧 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/agri-bot.git
   ```
2. Navigate to the project directory:
   ```bash
   cd agri-bot
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment variables:
   - Create a `.env` file and add your Groq API key:
     ```
     GROQ_API_KEY=your_api_key_here
     ```

## ▶️ Run the App

```bash
streamlit run app.py
```

## 🎨 UI Overview

- **Chat Interface**: Users can ask questions in any supported language.
- **Sidebar Language Selection**: Choose preferred language for conversation.
- **Live Translations**: User input is translated to English for processing, then translated back to the original language.
- **Background and Styling**: Custom CSS ensures readability and aesthetics.

## 🔍 How It Works

1. User inputs a question in their preferred language.
2. The bot detects the language and translates the question into English.
3. The translated question is processed by the AI chatbot.
4. The response is translated back to the user's language and displayed in the chat.

## 🏗 Future Improvements

- Add support for voice input and responses.
- Improve accuracy with domain-specific fine-tuned models.
- Enhance UI/UX for a better user experience.

