from simpletransformers.language_representation import RepresentationModel
from simpletransformers.config.model_args import ModelArgs

model_args = ModelArgs(max_seq_length=156)

model = RepresentationModel(
    "roberta",
    "roberta-large",
    args=model_args,
)
sentence_list = ["Natural language processing (NLP) is a subfield of linguistics, computer science, information engineering, and artificial intelligence concerned with the interactions between computers and human (natural) languages"]
word_embeddings = model.encode_sentences(sentence_list, combine_strategy="mean")

print(word_embeddings[0])
