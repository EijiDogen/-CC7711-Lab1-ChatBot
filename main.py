from chatbot import ChatBot
myChatBot = ChatBot()
#apenas carregar um modelo pronto
myChatBot.loadModel()

#criar o modelo
#myChatBot.createModel()

print("Bem-vindo ao Chatbot FEI Intercâmbio.")

pergunta = input("Como posso ajudar?\n")
resposta, intencao = myChatBot.chatbot_response(pergunta)

#debugging print
#print(resposta + "   ["+intencao[0]['intent']+"]")

#normal print
print(resposta+"\n")

while (intencao[0]['intent']!="despedida"):
    pergunta = input("Possui outra dúvida?")
    resposta, intencao = myChatBot.chatbot_response(pergunta)

    print(resposta+"\n")

print("Foi um prazer atendê-lo")
