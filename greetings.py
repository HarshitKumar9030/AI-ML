import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def preprocess_prompt(prompt):
    prompt = re.sub(r'[^\w\s]', '', prompt)
    prompt = prompt.lower()
    return prompt

greeting_data = [("hi there!", "greeting"),
                 ("hello", "greeting"),
                 ("hey", "greeting"),
                 ("greetings", "greeting"),
                 ("wassup", "greeting"),
                 ("yo", "greeting"),
                 ("howdy", "greeting"),
                 ("hi", "greeting"),
                 ("good morning", "not_greeting"),
                 ("good evening", "not_greeting"),
                 ("goodnight", "not_greeting"),
                 ("see you later", "not_greeting"),
                 ("catch you later", "not_greeting")]

prompts, labels = zip(*greeting_data)
prompts = [preprocess_prompt(prompt) for prompt in prompts]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(prompts)
y = labels

model = MultinomialNB()
model.fit(X, y)

def is_greeting(prompt):
    prompt = preprocess_prompt(prompt)
    prompt = vectorizer.transform([prompt])
    prediction = model.predict(prompt)[0]
    return prediction == "greeting"

prompt = input("Enter a prompt: ")
if is_greeting(prompt):
    print("The prompt is a greeting.")
else:
    print("The prompt is not a greeting.")
