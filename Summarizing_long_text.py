from transformers import pipeline

# Create the summarization pipeline
summarizer = pipeline(task="summarization", model="cnicu/t5-small-booksum")
# summarizer = pipeline(task="summarization", model="cnicu/t5-small-booksum", min_new_tokens=10, max_new_tokens=200)


# Original text
original_text = "Greece has many islands, with estimates ranging from somewhere around 1,200 to 6,000, depending on the minimum size to take into account. The number of inhabited islands is variously cited as between 166 and 227. The Greek islands are traditionally grouped into the following clusters: the Argo-Saronic Islands in the Saronic Gulf near Athens; the Cyclades, a large but dense collection occupying the central part of the Aegean Sea; the North Aegean islands, a loose grouping off the west coast of Turkey; the Dodecanese, another loose collection in the southeast between Crete and Turkey; the Sporades, a small tight group off the coast of Euboea; and the Ionian Islands, chiefly located to the west of the mainland in the Ionian Sea. Crete with its surrounding islets and Euboea are traditionally excluded from this grouping."

# Summarize the text
summary_text = summarizer(original_text)

# Compare the length
print(f"Original text length: {len(original_text)}")
print(f"Summary length: {len(summary_text[0]['summary_text'])}")
