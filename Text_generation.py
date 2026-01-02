from transformer import pipeline

# Instantiate the pipeline
generator = pipeline("text-generation", model="gpt2")

response = "Dear valued customer, I am glad to hear you had a good stay with us."

text = "I had a wonderful stay at the Riverview Hotel! The staff were incredibly attentive and the amenities were top-notch. The only hiccup was a slight delay in room service, but that didn't overshadow the fantastic experience I had.
"


# Complete the prompt
prompt = f"Customer review:\n{text}\n\nHotel reponse to the customer:\n{response}"

# Complete the model pipeline
outputs = generator(prompt, max_length=150, pad_token_id=generator.tokenizer.eos_token_id, truncation=True)

print(outputs[0]["generated_text"])
