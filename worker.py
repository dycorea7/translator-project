from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods
import requests

# Watsonx 설정
PROJECT_ID = "3b1c567e-6dd0-484d-ad2c-df7eb21d8b7f"
credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": "5GPhzfBpLa8yCy5u5XLXEJhx7_V5FlbMVOOm_-Aj9ecn"
}

model_id = "mistralai/mistral-medium-2505"

parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 1024
}

model = Model(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=PROJECT_ID
)

# STT 설정
STT_API_KEY = "LiEONhH7SxXZAElvumBCgtGEr986VevikKa3wwajjGmB"
STT_URL = "https://api.us-south.speech-to-text.watson.cloud.ibm.com/instances/2718ddee-d123-436f-bf89-e7d2735cf7f2"

# TTS 설정
TTS_API_KEY = "Dj4-il2h3i01SyqcQX9_Hjkh7cgKmYzaNO4URI6Yr4TA"
TTS_URL = "https://api.us-south.text-to-speech.watson.cloud.ibm.com/instances/d4fb4dc7-a22f-481e-879c-01f2eda5b40e"


def speech_to_text(audio_binary):

	# Set up Watson Speech-to-Text HTTP Api url
	api_url = STT_URL + '/v1/recognize'

	# Set up parameters for our HTTP reqeust
	params = {
		'model': 'en-US_Multimedia',
	}

	# Set up the body of our HTTP request
	body = audio_binary

	# Send a HTTP Post request
	response = requests.post(
    api_url,
    auth=("apikey", STT_API_KEY),
    headers={"Content-Type": "audio/webm"},  # ← webm으로 맞춰야 함
    params=params,
    data=audio_binary
    ).json()

	# Parse the response to get our transcribed text
	text = 'null'
	while bool(response.get('results')):
		print('Speech-to-Text response:', response)
		text = response.get('results').pop().get('alternatives').pop().get('transcript')
		print('recognised text: ', text)
		return text


def text_to_speech(text, voice=""):
    api_url = TTS_URL + '/v1/synthesize'
    if voice != "" and voice != "default":
        api_url += "?voice=" + voice
    headers = {
        'Accept': 'audio/wav',
        'Content-Type': 'application/json',
    }
    response = requests.post(
        api_url,
        auth=("apikey", TTS_API_KEY),
        headers=headers,
        json={'text': text}
    )
    print('TTS 응답:', response)
    return response.content


def watsonx_process_message(user_message):
    prompt = f"""
    Translate the following English sentence into Spanish. 
    Reply ONLY with the translation, no explanations, no formatting, no extra text.

    English: {user_message}
    Spanish:
    """
    response_text = model.generate_text(prompt=prompt)
    print("watsonx response:", response_text)
    return response_text.strip()