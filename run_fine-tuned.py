from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PegasusForConditionalGeneration, PegasusTokenizer

# My fine-tuned
tokenizer = AutoTokenizer.from_pretrained("SophieTr/fine-tune-Pegasus")
model = AutoModelForSeq2SeqLM.from_pretrained("SophieTr/fine-tune-Pegasus")
 
# Original
model_origin = PegasusForConditionalGeneration.from_pretrained('sshleifer/distill-pegasus-xsum-16-4')
tokenizer_origin = PegasusTokenizer.from_pretrained('sshleifer/distill-pegasus-xsum-16-4')

inp = "In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output. The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs."
input_ids = tokenizer(inp, return_tensors="pt").input_ids
outputs = model.generate(input_ids=input_ids)
print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
print("length: ", output.shape, "   ", input_ids.shape)

input_ids_benchmark = tokenizer_origin(inp, return_tensors="pt").input_ids
outputs_benchmark = model_origin.generate(input_ids=input_ids_benchmark)
print("Generated from benchmark\n:", tokenizer.batch_decode(outputs_benchmark, skip_special_tokens=True))
