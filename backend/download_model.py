import os
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = os.getenv("VQA_MODEL_ID", "")
MODEL_SIZE = os.getenv("MODEL_SIZE", "256M")

if not MODEL_ID:
	MODEL_SIZES = {
		'256M': 'HuggingFaceTB/SmolVLM2-256M-Video-Instruct',
		'500M': 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct',
		'2.2B': 'HuggingFaceTB/SmolVLM2-2.2B-Instruct',
		'1B': 'HuggingFaceTB/SmolVLM2-2.2B-Instruct',
		'2B': 'HuggingFaceTB/SmolVLM2-2.2B-Instruct',
	}
	key = MODEL_SIZE.upper()
	if key not in MODEL_SIZES:
		for k in MODEL_SIZES.keys():
			if k in key or key in k:
				key = k
				break
		else:
			key = '256M'
	MODEL_ID = MODEL_SIZES.get(key, MODEL_SIZES['256M'])

print(f"Pre-downloading model {MODEL_ID} to cache...")
AutoProcessor.from_pretrained(MODEL_ID)
AutoModelForImageTextToText.from_pretrained(MODEL_ID)
print("✅ Model downloaded and cached.")
