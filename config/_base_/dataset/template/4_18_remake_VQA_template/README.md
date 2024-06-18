## number of sentence 10,20,30

# VQA-AC:Generation command with gpt4:
You are an image annotation professional in the field of deep learning.
I want to create data about a visual inspection task to be trained on a Large Vision and Language Model (LVLM). Therefore, I need you to create a prompt that describes a defect in a product image.
Please consider prompts in English according to the following requirements.

Requirements
1. For a given image, come up with a question and answer about what defects are present in the image.
2. For the question sentence, generate a sentence that includes the meaning "What defects do you see in the product in this image <image>?
3. Answer sentence A: If the defect is present　
Generate a response statement that means "This image has {defects}."
4. Answer statement B: If there is no defect
Generate a response statement that means "The product in this image has no defects." 

Generation conditions
1. Create sentences of various lengths, from short to long.
2. Generate English sentences with as many different expressions as possible by changing words and expressions between sentences.
3. In each questionnaire, be sure to include "<image>" immediately after the word meaning "image".
4. Be sure to include {defect} in the response sentence when there is a defect. This part will be set by the user later as an "f" strings of python.
5. Please create sentences for each of the 10 questions, 10 answer sentences A and 10 answer sentences B. However, please generate the sentences in such a way that a natural conversation can take place even if the questions and answers are chosen at random.
6. The order of sentence generation should be as follows: 
(1) Generate all questions
(2) Generate all answers A
(3) Generate all answers B.
7. The generated prompts are treated as LLM input. Therefore, please generate grammatically correct and naturally expressed English sentences.

# refine sentence

feedback
#Please review again the following generation requirements for generating statements. 
#Next, increase the complexity of the statement and generate the statement according to the following generation requirements.
#However, in increasing the complexity of the sentence, do not add any information that is not related to the visual information obtained from the image.

Generation conditions
1. Create sentences of various lengths, from short to long.
2. Generate English sentences with as many different expressions as possible by changing words and expressions between sentences.
3. In each questionnaire, be sure to include "<image>" immediately after the word meaning "image".
4. Be sure to include {defect} in the response sentence when there is a defect. This part will be set by the user later as an "f" strings of python.
5. Please create sentences for each of the 10 questions, 10 answer sentences A and 10 answer sentences B. However, please generate the sentences in such a way that a natural conversation can take place even if the questions and answers are chosen at random.
6. The order of sentence generation should be as follows: 
(1) Generate all questions
(2) Generate all answers A
(3) Generate all answers B.
7. The generated prompts are treated as LLM input. Therefore, please generate grammatically correct and naturally expressed English sentences.



# VQA-PG:Generation command with gpt4:
You are an image annotation professional in the field of deep learning.
I want to create data about a visual inspection task to be trained on a Large Vision and Language Model (LVLM). Therefore, I need you to create a prompt that describes a what kind of product in a image.
Please consider prompts in English according to the following requirements.

Requirements
1. For a given image, come up with a question that asks about what product is shown in the image and the answer to that question.
2. For the question sentence, generate a sentence that includes the meaning "What kind of product is shown in this image <image>?" in response to the question text.
3. For the answer sentence, generate a response statement that means "This image show a {products}."

Generation conditions
1. Create sentences of various lengths, from short to long.
2. Generate English sentences with as many different expressions as possible by changing words and expressions between sentences.
3. In each questionnaire, be sure to include "<image>" immediately after the word meaning "image".
4. Be sure to include {product} in the question sentence. This part will be set by the user later as an "f" strings of python.
5. Please create sentences for each of the 20 questions, 20 answer sentences. However, please generate the sentences in such a way that a natural conversation can take place even if the questions and answers are chosen at random.
6. The order of sentence generation should be as follows: 
(1) Generate all questions
(2) Generate all answers 
7. The generated prompts are treated as LLM input. Therefore, please generate grammatically correct and naturally expressed English sentences.

# refine sentence
feedback
#Please review again the following generation requirements for generating statements. 
#Next, increase the complexity of the statement and generate the statement according to the following generation requirements.
#When complicating sentences, however, do not add information that is not relevant to the visual information obtained from the image, such as sales information or advertising.

Generation conditions
1. Create sentences of various lengths, from short to long.
2. Generate English sentences with as many different expressions as possible by changing words and expressions between sentences.
3. In each questionnaire, be sure to include "<image>" immediately after the word meaning "image".
4. Be sure to include {product} in the question sentence. This part will be set by the user later as an "f" strings of python.
5. Please create sentences for each of the 20 questions, 20 answer sentences. However, please generate the sentences in such a way that a natural conversation can take place even if the questions and answers are chosen at random.
6. The order of sentence generation should be as follows: 
(1) Generate all questions
(2) Generate all answers 
7. The generated prompts are treated as LLM input. Therefore, please generate grammatically correct and naturally expressed English sentences.


# VI_LOC
## Questions
Flicker30kのテンプレをすべて読ませたうえで出力させる．

You are an image annotation professional in the deep learning field.
You want to create data about a visual inspection task to be trained on a large visual language model (LVLM). So, please create an instruction tuning style prompt that would determine whether a product in an image is good or bad.
Please consider prompts in English that meet the following requirements.

Fisrt, read the sentences below and understand. These sentences are used for template question for training object defection in Fliker30k.

## Answers
You are an image annotation professional in the deep learning field.
You want to create data about a visual inspection task to be trained on a large visual language model (LVLM). So, please create questions and 2 types of answers which satisfy the following requirements.

Requirement.
1. for a given image, come up with a question and an answer that asks whether the product in the picture is good or defective. However, be sure to refer to the attached [Reference Text] and [Generation Conditions] for the question and answer text. 
2. the question or instruction texts:
[Reference Text]
#"If there are any defects, can you provide a description of each mentioned defect on the {product} in the image <image> and include the coordinates [x0,y0,x1,y1] for each defect?"
#"If there are any defects, please explain what kind of defects exist on the {product} in the photo <image> and give coordinates [xmin,ymin,xmax,ymax] for all the defects you reference."
#"Analyze the {product} in the picture <image> and, if there are any defects, share the positions of the mentioned defects."
3. answer A: In case that there is one defect.
[Reference Text]
#"There is a <ph_st>{defect}<ph_ed> on the {position} of this {product}."
#"A <ph_st>{defect}<ph_ed> can be seen in the {position} of the {product}."
#"This {product} has a <ph_st>{defect}<ph_ed>, which is located in the {position}."
4. answer B: In case that there are defects.
[Reference Text]
#"This {product} has multiple instances of the same <ph_st>{defect}<ph_ed> located in the {position} area."
#"Several identical <ph_st>{defect}<ph_ed> are visible in the {position} of the {product}."
#"This {product} exhibits multiple instances of a single type of <ph_st>{defect}<ph_ed>, all situated in the {position}."

[Generation Conditions]
1. Vary the length of the sentence.
2. Vary the words and expressions between sentences to generate English sentences with as many different expressions as possible.
3. Be sure to include "{product}" in each question or insrtuction. These contain the name of the product, and this part will be set by the user later as f-strings of python.
4. Be sure to include "<ph_st>{defect}<ph_ed>", "{product}" and "{position}"  in each answer. This part will be set by the user later as f-strings of python.
5. Create 20 questions, 20 answers A, and 20 answers B. Please generate responses in such a way that even if a response is randomly selected from answer A or answer B, the conversation remains natural.
6. The order of sentence generation should be as follows
##1) Generate all question or instruction sentences, ##2) Generate answer A ##2) Generate answer B
7. Finally, the generated prompts are treated as LLM input. Therefore, please generate grammatically correct and naturally expressed English sentences.
8. Please generate sentences that conform to all of the "Generation Conditions".

# refine statements
Feedback
# Please review again the following [Generation Cequirements] for generating statements. 
# Next, increase the complexity of the statement and generate the statement according to the following generation requirements.
# However, in increasing the complexity of the sentence, do not add any information that is not related to the visual information obtained from the image.

[Generation Conditions]
1. Vary the length of the sentence.
2. Vary the words and expressions between sentences to generate English sentences with as many different expressions as possible.
3. Be sure to include "{product}" in each question or insrtuction. These contain the name of the product, and this part will be set by the user later as f-strings of python.
4. Be sure to include "<ph_st>{defect}<ph_ed>", "{product}" and "{position}"  in each answer. This part will be set by the user later as f-strings of python.
5. Create 20 questions, 20 answers A, and 20 answers B. Please generate responses in such a way that even if a response is randomly selected from answer A or answer B, the conversation remains natural.
6. The order of sentence generation should be as follows.
##1) Generate all question or instruction sentences, ##2) Generate answer A ##3) Generate answer B
7. Finally, the generated prompts are treated as LLM input. Therefore, please generate grammatically correct and naturally expressed English sentences.
8. Please generate sentences that conform to all of the "Generation Conditions".


# generate additional pattern answer
## step3
Additional Requirement(1/3)

# Generate a response to the question using the example sentences for the case where two or three types of defects exist as well.
# Please generate 20 patterns of sentences for each item.　However, increase the complexity of the statement and generate the statement according to the following generation requirements.

The pattern of sentences to be generated is as follows.
1. if there are two types of defects; 
1.1 If only one defect exists for all defects;
[Reference sentence]
# "{defect1} is located in the {position1} of the {product}, and {defect2} is confirmed to be in the {position2}."
# "There is {defect1} in the {position1} and {defect2} in the {position2} of the {product}."
1.2 If there is one defect on one side and multiple defects on the other side;
[Reference sentence]
# "{defect1} is located in the {position1} of the {product}, while multiple {defects2} are found in the {position2}."
# "There is a {defect1} in the {position1}, and several {defects2} are in the {position2} of the {product}."
1.3 If there is more than one of each of the two defects;
[Reference sentence]
# "{defects1} are located in the {position1} of the {product}, and {defects2} can be seen in the {position2}."
# "There are {defects1} in the {position1} and {defects2} in the {position2} of the {product}."

[Generation Conditions]
1. Vary the length of the sentence.
2. Vary the words and expressions between sentences to generate English sentences with as many different expressions as possible.
3. Be sure to include "<ph_st>{defect}<ph_ed>", "{product}" and "{position}"  in each answer. This part will be set by the user later as f-strings of python.
4. Create 20 answers 1.1 to 1.3. 
5. The order of sentence generation should be as follows
##1) Generate answer sentence 1.1  ##2) Generate answer sentence 1.2  ##3)Generate answer sentence 1.3
6. Finally, the generated prompts are treated as LLM input. Therefore, please generate grammatically correct and naturally expressed English sentences.
7. Please generate sentences that conform to all of the "Generation Conditions".


## step4
Additional Requirement(2/3)

Generate a response to the question using the example sentences for the case where two or three types of defects exist as well.
Please generate 20 patterns of sentences for each item.　However, increase the complexity of the statement and generate the statement according to the following generation requirements.

The pattern of sentences to be generated is as follows.

2. when there are three types of defects:
2.1 When there is only one defect for all defects;
[Reference sentence]
# "{defect1} is located in the {position} of the {product}, {defect2} is in the {position2}, and {defect3} is also present, located in the {position3}."
# "There is a {defect1} in the {position1}, a {defect2} in the {position2}, and a {defect3} in the {position3} of the {product}."
2.2 When two defects are singular and there is more than one defect of the remaining type;
[Reference sentence]
# "{defect1} is located in the {position1} of the {product}, multiple {defects2} are found in the {position2}, and {defect3} is also present, located in the {position3}."
# "There is a {defect1} in the {position1}, several {defects2} in the {position2}, and a {defect3} in the {position3} of the {product}."
2.3 When one defect is singular and there are multiple defects of the other two types;
[Reference sentence]
# "Multiple {defects1} are located in the {position1} of the {product}, {defect2} is in the {position2}, and several {defects3} are also found in the {position3}."
# "There are several {defects1} in the {position1}, a {defect2} in the {position2}, and several {defects3} in the {position3} of the {product}."


[Generation Conditions]
1. Vary the length of the sentence.
2. Vary the words and expressions between sentences to generate English sentences with as many different expressions as possible.
3. Be sure to include "<ph_st>{defect}<ph_ed>", "{product}" and "{position}"  in each answer. This part will be set by the user later as f-strings of python.
4. Create 20 answers 2.1 to 2.3. 
5. The order of sentence generation should be as follows
##1) Generate answer sentence 2.1  ##2) Generate answer sentence 2.2  ##3)Generate answer sentence 2.3
6. Finally, the generated prompts are treated as LLM input. Therefore, please generate grammatically correct and naturally expressed English sentences.
7. Please generate sentences that conform to all of the "Generation [Generation Conditions]".

## step5
Additional Requirement(3/3)

Generate a response to the question using the example sentences for the case where two or three types of defects exist as well.
Please generate 20 patterns of sentences for each item.　However, increase the complexity of the statement and generate the statement according to the following generation requirements.

The pattern of sentences to be generated is as follows.

3. If there are no defects:
[Reference sentence].
# "No defects were found in this {product}."
# "No defects were detected on this {product}."


[Generation Conditions]
1. Vary the length of the sentence.
2. Vary the words and expressions between sentences to generate English sentences with as many different expressions as possible.
3. Create 20 answers 3. 
4. Finally, the generated prompts are treated as LLM input. Therefore, please generate grammatically correct and naturally expressed English sentences.
5. Please generate sentences that conform to all of the "Generation Conditions".