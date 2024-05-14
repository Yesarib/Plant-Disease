import google.generativeai as genai
from dotenv import load_dotenv
import os
import json

load_dotenv()
my_variable = os.getenv('GEMINI_KEY')

def recommendation(promt):
    genai.configure(api_key=my_variable)
    model = genai.GenerativeModel('gemini-pro')
    if promt == "healthy":
        response = model.generate_content("bitkinin Xception modeli üzerinde çıkan sonucu bu: "+ promt + ". Bu çıkan sonuca göre bitki için ek bir şeyler önerir misin ?")
        text = response._result.candidates[0].content.parts[0].text
        return text

    else:
        response = model.generate_content("bitkinin Xception modeli üzerinde çıkan sonucu bu: "+ promt + ". Bu bitki hastalığı için ne önerirsin ?")
        text = response._result.candidates[0].content.parts[0].text

        return text

