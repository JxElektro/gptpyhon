import openai  # importing the openai library


def open_file(filepath):  # function to open the file containing the API key
    # open the file with read permission, utf-8 encoding
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()  # return the content of the file


# set the openai api key as the content of the file
openai.api_key = open_file('openaiapikey.txt')



def gpt3_completion(prompt, engine='text-davinci-002', temp=0.7, top_p=1.0, tokens=400, freq_pen=0.0, pres_pen=0.0, stop=['&#8203;`oaicite:{"index":0,"invalid_reason":"Malformed citation <<END>>"}`&#8203;']):
    """
    function to get the gpt3 completion
    prompt : str : the prompt to give to the model
    engine : str : the model to use, default text-davinci-002
    temp : float : temperature for the model, default 0.7
    top_p : float : top p for the model, default 1.0
    tokens : int : maximum number of tokens to generate, default 400
    freq_pen : float : frequency penalty, default 0.0
    pres_pen : float : presence penalty, default 0.0
    stop : list : list of stop tokens, default ["&#8203;`oaicite:{"index":1,"invalid_reason":"Malformed citation <<END>>"}`&#8203;"]
    """
    prompt = prompt.encode(encoding='ASCII', errors='ignore').decode(
    )  # remove any non-ascii characters from the prompt
    response = openai.Completion.create(  # send a request to the openai api
        engine=engine,
        prompt=prompt,
        temperature=temp,
        max_tokens=tokens,
        top_p=top_p,
        frequency_penalty=freq_pen,
        presence_penalty=pres_pen,
        stop=stop)
    # extract the text from the response
    text = response['choices'][0]['text'].strip()
    return text


if __name__ == '__main__':
    prompt = 'Write a list of famous American actors:'
    response = gpt3_completion(prompt)  # get the response from the gpt3 model
    print(response)  # print the response
