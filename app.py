import streamlit as st
import torch
import pickle
from model_components import Transformer, InputReady, Masking, text_preprocessing, word_tokenize, adjust_seq

with open("in_vocab.pkl", "rb") as f:
    in_vocab = pickle.load(f)
with open("out_vocab.pkl", "rb") as f:
    out_vocab = pickle.load(f)

inv_out_vocab = {v: k for k, v in out_vocab.items()}

d_model = 512
num_heads = 4
nodes = 512
num_layers = 3
dropout_rate = 0.1
max_len = 300

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Transformer(d_model, num_heads, nodes, dropout_rate, num_layers, len(out_vocab)).to(device)
encoder_input_ready = InputReady(d_model, len(in_vocab)).to(device)
decoder_input_ready = InputReady(d_model, len(out_vocab)).to(device)
masking = Masking(max_len).to(device)

model.load_state_dict(torch.load("transformer_model.pth", map_location=device))
model.eval()

def translate(sentence):
    tokens = word_tokenize(text_preprocessing(sentence))
    tokens = adjust_seq(tokens, max_len=max_len)
    tokens = [in_vocab.get(tok, in_vocab['<unk>']) for tok in tokens]
    tokens += [in_vocab['<pad>']] * (max_len - len(tokens))
    encoder_input = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    enc_ready = encoder_input_ready(encoder_input)
    mask = masking.padding_mask(encoder_input, in_vocab['<pad>'])

    decoded = [out_vocab['<sos>']]
    for _ in range(max_len):
        dec_tensor = torch.tensor(decoded, dtype=torch.long).unsqueeze(0).to(device)
        dec_ready = decoder_input_ready(dec_tensor)
        self_mask = masking.decoder_mask(dec_tensor, out_vocab['<pad>'])
        output = model(enc_ready, dec_ready, mask, self_mask, mask)
        next_token = output.argmax(dim=-1)[:, -1].item()
        if next_token == out_vocab['<eos>']:
            break
        decoded.append(next_token)

    return " ".join(inv_out_vocab[idx] for idx in decoded[1:])

st.title("Neural Machine Translation - Hinglish -> English")
sentence = st.text_area("Enter a sentence:")

if st.button("Translate"):
    if sentence.strip():
        translation = translate(sentence)
        st.write("**Translation:**", translation)
    else:
        st.warning("Please enter some text.")
