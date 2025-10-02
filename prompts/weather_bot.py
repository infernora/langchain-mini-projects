#this is a weather bot that uses dynamic prompting to give suggestions

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
import os, random
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct", 
    task = "text-generation",
    temperature=0.6,
    max_new_tokens=200
)

model = ChatHuggingFace(llm = llm)

# Dynamic context (simulating external data)
weather_data = {
    "Paris": random.choice(["sunny", "rainy", "cloudy"]),
    "Tokyo": random.choice(["sunny", "rainy", "cloudy"]),
    "New York": random.choice(["sunny", "rainy", "cloudy"])
}

chat_template = ChatPromptTemplate([
    ("system", "You are a helpful travel assistant who always adapts advice to the weather."),
    ("human", "I'm traveling to {city}. The weather there is {weather}. Suggest 2 activities I should do.")
])

# user input
city = "Paris"
weather = weather_data[city]

prompt = chat_template.invoke({"city": city, "weather": weather})

response = model.invoke(prompt)

print("Dynamic Context →", city, "weather is", weather)
print("\nAssistant Suggestion:")
print(response.content)
