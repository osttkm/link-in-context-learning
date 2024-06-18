# VQA-AC:Generation command with gpt4:
You are a professional image annotator. Please consider the English prompts according to the following requirements.
Requirement.
#For a given image, come up with a question and answer about what defects you see in the image.
For the #question, generate a sentence that includes the meaning "Is {product} in this image <image> defective?" and generate a sentence that includes the meaning "What defects do you see in the product in this image <image>?
#For the answer statement if a defect exists, generate a sentence that includes the meaning "The {product} in this image has a {defect}." The response sentence for the case where a defect exists should be "{product} in this image has a {defect}.
#In case there is no defect, please generate a response sentence that includes the following meaning: "There is no defect in {product} in this image. The response sentence for the case where there is no defect should be "There is no defect in {product} in this image.

Generation conditions
#Vary the length of the sentences.
#Please generate English sentences with as much variety of expressions as possible by changing the words and expressions between sentences.
#Each interrogative sentence must contain {product}. This part will be set later by the user.
#Be sure to include {defect} and {product} in your response. This part is set later by the user.
#Create an answer for each of the 30 questions and 30 defects that may or may not exist. However, generate sentences such that a natural conversation can take place even if the questions and answers are randomly selected.
#The order of sentence generation should be as follows: (1) Generate all the questions, (2) Generate all the answers, and (3) Generate all the answers.
##1) Generate all question sentences, #2) Generate all answer sentences when a defect exists, and #3) Generate all answer sentences when no defect exists.
#Finally, the generated prompts will be treated as LLM input. Therefore, please generate English sentences that are grammatically correct and naturally expressed.
#Generate sentences that adhere to all of the Generation Requirements items.

# refine sentence
Please review the following generation requirements again regarding sentence generation. Next, increase the complexity of the sentences and generate the sentences according to the following generation requirements.

Generation conditions
#Vary the length of the sentences.
#Please generate English sentences with as much variety of expressions as possible by changing the words and expressions between sentences.
#Each interrogative sentence must contain {product}. This part will be set later by the user.
#Be sure to include {defect} and {product} in your response. This part is set later by the user.
#Create an answer for each of the 30 questions and 30 defects that may or may not exist. However, generate sentences such that a natural conversation can take place even if the questions and answers are randomly selected.
#The order of sentence generation should be as follows: (1) Generate all the questions, (2) Generate all the answers, and (3) Generate all the answers.
##1) Generate all question sentences, #2) Generate all answer sentences when a defect exists, and #3) Generate all answer sentences when no defect exists.
#Finally, the generated prompts will be treated as LLM input. Therefore, please generate English sentences that are grammatically correct and naturally expressed.
#Question sentences must contain {product}.This part will be set later by the user.
#Answer sentences must include {defect} and {product}. This part is set later by the user.#Create a response for each of the 30 questions and 30 defects that may or may not exist.However, generate sentences in such a way that a natural conversation can take place even if the questions and answers are selected at random.##1) Generate all of the sentences in the following order.
##1) Generate all question sentences, #2) Generate all answer sentences when a defect exists, and #3) Generate all answer sentences when no defect exists.
##Finally, the generated prompts are treated as LLM input. Therefore, please generate English sentences that are grammatically correct and naturally expressed.
#Generate sentences that adhere to all of the items in all of the generation requirements.


# PG VQA  with gpt4:
You are a professional image annotator. Please consider the English prompts according to the following requirements.
Requirement.
#For a given image, come up with a question and answer about what product is in the image.
For the #question, generate a sentence that includes the meaning "What product is shown in this image?" Please generate a sentence that includes the meaning "What products are shown in this image?
#For the answer statement, generate a sentence that includes the meaning "In this image you can see {product}." The answer sentence should include the meaning "In this image you can see {product}.

Generation conditions
#The length of sentences must be varied.
#Please generate English sentences with as much variety of expressions as possible by changing the words and expressions between sentences.
#Response sentences must include {product}. This part will be set up later by the user.
#Generate 30 types of questions and 30 types of answers. However, please generate sentences in such a way that a natural conversation can take place even if the questions and answers are selected at random.
##The order of sentence generation should follow (1) through (2) below.
##1) Generate all question sentences, ##2) Generate all answer sentences.
##Finally, the generated prompts are treated as LLM input. Therefore, please generate English sentences that are grammatically correct and naturally expressed.
#Generate sentences that adhere to all of the Generation Requirements items.

# refine sentence

Please review the following generation requirements again regarding sentence generation. Next, increase the complexity of the sentences and generate the sentences according to the following generation requirements.

Generation Requirements
#Vary the length of sentences.
#Vary the words and expressions between sentences to generate English sentences with as much variety of expressions as possible.
#Response sentences must include {product}. This part will be set up later by the user.
#Create 30 types of questions and 30 types of answers. However, please generate sentences in such a way that a natural conversation can take place even if the questions and answers are selected at random.
##The order of sentence generation should follow (1) through (2) below.
##1) Generate all question sentences, ##2) Generate all answer sentences
##Finally, the generated prompts are treated as LLM input. Therefore, please generate English sentences that are grammatically correct and naturally expressed.
#Generate sentences that adhere to all of the Generation Requirements items.


# Loc_VI generation  with cloud3:

You are a professional image annotator. Please consider English prompts according to the following requirements.

Requirement.
##For a given image, come up with a question and answer about what defects you see in the image.
##For the question statement, generate a statement that includes the meaning "Where is the {defect} in {product} in this image <image>?" Generate a sentence that includes the meaning "The {defect} is located in this image <image>.
##For answer statement A, generate a statement that includes the meaning "{defectA} is located at ____ of {product} and {defectB} is located at ____ of {product}." Generate a sentence containing the meaning "{defectA} is present in ____ of {product}.
##For answer statement B, generate a statement that means "{defectA} exists in ____ of {product}, {defectB} exists in ____ of {product}, and {defectC} exists in ____ of {product}." Generate a sentence containing the meaning "{defectA} exists in XXX of {product}, {defectB} exists in XXX of {product}, and {defectC} exists in XXX of {product}.

Generation conditions
#Vary the length of the sentence.
#Vary the words and expressions between sentences to generate English sentences with as many different expressions as possible.
#Be sure to include "{product}" and "{defect}" in each question. These contain the name of the product and the name of the defect, and this part will be set by the user later.
#Be sure to include "{defect}" and "{product}" in each answer. Also, XX indicates the location where the defect exists, be sure to use "{position}" for this part. These parts will be set by the user later.
#Create 30 questions, 30 answers A, and 30 answers B. However, please generate the sentences in such a way that a natural conversation can take place even if the questions and answers 1 and 2 are chosen at random.
#The order of sentence generation should be as follows
##1) Generate all question sentences, ##2) Generate answer sentence A ##2) Generate answer sentence B
##Finally, the generated prompts are treated as LLM input. Therefore, please generate grammatically correct and naturally expressed English sentences.
#Please generate sentences that conform to all of the "Generation Requirements".

# refine
Please review again the following generation requirements for generating statements. Next, increase the complexity of the statement and generate the statement according to the following generation requirements.

Generation Requirements
#Vary the length of the sentence.
#Vary the words and expressions between sentences to generate as many different English sentences as possible.
#Be sure to include "{product}" and "{defect}" in each question. These contain the name of the product and the name of the defect, which will be set by the user later.
#Be sure to include "{defect}" and "{product}" in each answer. Also, XX indicates the location where the defect exists, be sure to use "{position}" for this part. These parts will be set by the user later.
Create #30 questions and answers. However, please generate the sentences so that the conversation can be natural even if the questions and answers are randomly selected.
#The order of sentence generation should be as follows
##1) Generate all question sentences, ##2) Generate answer sentences
##Finally, the generated prompts are treated as input for the LLM. Therefore, please generate grammatically correct and naturally expressed English sentences.
Please generate sentences that comply with all of the items in ##Generation Requirements.