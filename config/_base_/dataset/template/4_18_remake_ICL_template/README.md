# VI:Generation command with gpt4:
## step1
You are an image annotation professional in the deep learning field.
You want to create data about a visual inspection task to be trained on a large visual language model (LVLM). So, please create a prompt that would determine whether a product in an image is good or bad.
Please consider prompts in English that meet the following requirements

Requirements
1. for a given image, come up with a question and an answer that asks whether the product in the picture is good or defective. However, be sure to refer to the attached [Reference Text] and [Generation Conditions] for the question and answer text. 
2. the question text:
[Reference text]
#"In the image <image> provided, can you identify any apparent defects such as {subfolder_string} on this {product}" 
#"In this image <image>, do you see any faults, specifically like {subfolder_string}, on the {product}?
3. about the answer text when the image is good:
[Reference text].
#"Yes. A clear indication of {defect} on this {product} suggests it's defective."
#Unmistakably, the {product} shows {defect}, pointing to a defect." 
4. regarding the response text when the image is defective:
[Reference text].
#"No. No visible faults like {subfolder_string} on the {product}."
#"No. This {product} displays no {subfolder_string} issues, suggesting it's in prime condition."

[Generated Condition]
1. Create sentences of various lengths by changing the wording of the sentences. 
2. create as many different phrases of questions and answers as possible by changing the words and expressions. 
3. always include the word "<image>" immediately after the word "image" in all questions. 
4. Be sure to include {product} and {subfolder_string} in the question and answer statements when the image is good. This part will be set by the user later as a python f-string. 
5. Be sure to include {defect} in the response text when the image is defective. This part will be set by the user as a python f-string later. 
6. create 10 question sentences and 10 answer sentences for each of the cases where the image is defective and defective, respectively. However, the sentences should be generated in such a way that a natural conversation can take place even if the questions and answers are chosen at random. 
7. The order of sentence generation should be as follows 
(1) Generate all questions
(2) Generate answers if all images are good 
(3) Generate an answer if all images are defective 
8. the generated prompts are treated as input to the LLM. Therefore, please generate grammatically correct and naturally expressed English text.

# refine sentence
## step2
Feedback
#Review again the following [Generation Conditions] for generating sentences.
#Next, increase the complexity of the sentences by increasing the variety of words and phrases.
#However, when increasing sentence complexity, do not add information that is not related to the visual information obtained from the image.

[Generation Conditions]
1. create sentences of various lengths by changing the wording of the sentences
2. produce as many differently worded questions and answers as possible by changing the words and expressions.
3. always include the word "<image>" immediately after the word "image" in all questions.
4. Be sure to include {product} and {subfolder_string} in the question and answer statements when the image is good. This part will be set by the user later as a python f-string.
5. Be sure to include {defect} in the response text when the image is defective. This part will be set by the user as a python f-string later.
6. create 10 question sentences and 10 answer sentences for each of the cases where the image is defective and defective, respectively. However, the sentences should be generated in such a way that a natural conversation can take place even if the questions and answers are chosen at random.
7. The order of sentence generation should be as follows
(1) Generate all questions
(2) Generate answers if all images are good
(3) Generate an answer if all images are defective
8. the generated prompts are treated as input to the LLM. Therefore, please generate grammatically correct and naturally expressed English text.

# add LCL question of query image
## step3
Additional instructions.

Please add the meaning "in this context" to all of the questions based on the following reference sentences. Output all sentences with the added meaning.

Reference Sentences
#"Considering the context, in the image <image> provided, can you identify any apparent defects such as {subfolder_string} on this {product}?"
#"Given the context, regarding the image <image>, does this {product} exhibit any noticeable issues, for example, {subfolder_string}?"
#"With the context in mind, looking at the image <image>, are there discernible problems like {subfolder_string} present on the {product}?"
#"Reflecting on the context, upon examining the image <image>, are there defects such as {subfolder_string} visible on this {product}?"
#"In the light of the context, can any {subfolder_string}-type anomalies be spotted on the {product} in the image <image>?"



# Instruction issue VI
You are an image annotation professional in the deep learning field.
You want to create data about a visual inspection task to be trained on a large visual language model (LVLM). So, please create an instruction tuning style prompt that would determine whether a product in an image is non-defective or defective.
Please consider prompts in English that meet the following requirements

Requirements
1. for a given image, come up with a question and answer that asks you to determine whether the product in the image is non-defective or defective. However, be sure to refer to the accompanying [Reference Text] and [Generation Conditions] for the instructions and answer text. 
2. instruction text:
[Reference Text]
#"Can you confirm any obvious defects like those on {product} in the provided image <image>? If present, classify it as defective; if absent, classify it as non-defective."
#"In this image <image>, can you see any defects on {product}, specifically like {subfolder_string}? If present, classify it as defective; if absent, classify it as non-defective." 3.
3. regarding the response text when the image is non-defective:
[Reference text]
#"Defective. this is because this {product} exhibits {defect}."
#"Defective. Because this {product} has {defect} which are not found in non-defective products." 
4. the response text when the image is defective:
[Reference text]
#"Non-defective. None of the potential defects {sub_strings} can be confirmed on this {product}."
#"Non-defective. This is because {sub_strings} were not found on this {product}."

[Generation Condition] 
1. Create sentences of various lengths by changing the wording of the sentences. 
2. create as many differently worded directives and answers as possible by changing the wording and expressions. 
3. always include the word "<image>" immediately after the word "image" in all questions. 
4. Be sure to include {product} and {subfolder_string} in both the directive and the answer when the image is non-defective. This part will be set by the user later as a python f-string. 
5. Be sure to include {defect} in the response text when the image is defective. This part will be set by the user later as a python f-string. 
6. create 10 instruction sentences and 10 response sentences for each of the cases where the image is defective and defective, respectively. However, the sentences should be generated in such a way that a natural conversation can take place even if the instructions and answers are chosen randomly. 
7. The order of sentence generation should be as follows 
(1) Generate all instructions
(2) Generate answers if all images are non-defective 
(3) Generate an answer if all images are defective 
8. the generated prompts are treated as input to the LLM. Therefore, please generate grammatically correct and naturally expressed English text.