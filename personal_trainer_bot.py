#this is a personal fitness trainer bot which uses dynamic prompting, multi-turn conversation and message placeholders(chat history)


from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from dotenv import load_dotenv

load_dotenv()

chat_file = os.path.join(os.path.dirname(__file__), "trainer_history.txt")

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct", 
    task = "text-generation",
    temperature=0.6,
    max_new_tokens=200
)

model = ChatHuggingFace(llm = llm)

chat_template = ChatPromptTemplate([
    ('system', 'You are a personal fitness trainer and nutritionist'),  #set the role
    MessagesPlaceholder(variable_name = 'trainer_history'),
    ('human','{question}')
])

trainer_history = []
try:
    with open(chat_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            role, content = line.split(":", 1)
            trainer_history.append((role.lower(), content.strip()))
except FileNotFoundError:
    print("No previous history found. Starting fresh.")


while True:
    question = input("You: ")
    if question == 'exit':
        break

    prompt = chat_template.invoke({
        'trainer_history': trainer_history,
        'question': question
    })

    response = model.invoke(prompt)
    print(response.content)

    trainer_history.append(("human", question))
    trainer_history.append(("ai", response.content))

    with open(chat_file, "w", encoding="utf-8") as f:
        for role, text in trainer_history:
            safe_text = text.replace("\n", "<nl>")
            f.write(f"{role}:{safe_text}\n")


