from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# Load model and tokenizer
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt", src_lang="en_XX")

# Input article in English
article_en = "The State being an exceptional performer in the healthcare sector has recognised the importance of the Life Sciences sector and its contribution to improving healthcare and quality of life."
model_inputs = tokenizer(article_en, return_tensors="pt")

# Translate from English to Tamil
generated_tokens = model.generate(
    **model_inputs,
    forced_bos_token_id=tokenizer.lang_code_to_id["ta_IN"]  # Tamil language code
)

# Decode the translation
translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
print(translation)
