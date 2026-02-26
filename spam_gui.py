import tkinter as tk
from tkinter import messagebox
import joblib
import re
import string

# Load saved model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def predict_message():
    message = text_box.get("1.0", tk.END).strip()

    if message == "":
        messagebox.showwarning("Warning", "Please enter a message!")
        return

    cleaned = clean_text(message)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]

    if prediction == 1:
        result_label.config(text="⚠️ SPAM MESSAGE", fg="white", bg="#e74c3c")
    else:
        result_label.config(text="✅ SAFE MESSAGE", fg="white", bg="#2ecc71")

def clear_text():
    text_box.delete("1.0", tk.END)
    result_label.config(text="", bg="#f4f6f7")

# GUI Setup
root = tk.Tk()
root.title("📧 Email Spam Detection System")
root.geometry("600x450")
root.configure(bg="#f4f6f7")

title_label = tk.Label(root, text="Email Spam Detection", 
                       font=("Helvetica", 20, "bold"), 
                       bg="#f4f6f7", fg="#2c3e50")
title_label.pack(pady=20)

text_box = tk.Text(root, height=8, width=60, font=("Arial", 12))
text_box.pack(pady=10)

button_frame = tk.Frame(root, bg="#f4f6f7")
button_frame.pack(pady=10)

check_btn = tk.Button(button_frame, text="Check Message",
                      font=("Arial", 12, "bold"),
                      bg="#3498db", fg="white",
                      width=15, command=predict_message)
check_btn.grid(row=0, column=0, padx=10)

clear_btn = tk.Button(button_frame, text="Clear",
                      font=("Arial", 12, "bold"),
                      bg="#95a5a6", fg="white",
                      width=10, command=clear_text)
clear_btn.grid(row=0, column=1, padx=10)

result_label = tk.Label(root, text="", font=("Arial", 16, "bold"),
                        width=30, height=2, bg="#f4f6f7")
result_label.pack(pady=20)

root.mainloop()