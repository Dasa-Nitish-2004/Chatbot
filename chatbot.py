from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS

def populateResult(que,faiss_index):
        ret = faiss_index.as_retriever()
        rdocs = ret.get_relevant_documents(que)
        print("More Suitable question")
        result = []

        

        for doc in rdocs:
            prompt = doc.page_content.split('?')[0].replace('prompt: ', '')
            response = doc.page_content.split('?')[1].replace('response: ', '')
            result.append({"question":prompt,"answer":response})
            # print(f"Question: {prompt}")
            # print(f"Answer: {response}\n")
        return result

def AskQuestion(user_input):
        
    llm = ChatGoogleGenerativeAI(
        api_key="AIzaSyDOTWIDe1OkNOJv01XC8OU5jQOB4xsA2dw",
        model="gemini-1.5-pro",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # other params...
    )

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key="AIzaSyDOTWIDe1OkNOJv01XC8OU5jQOB4xsA2dw")

    loader =CSVLoader(file_path="D:\\nitish\\python\\ml\\codebasics\\chatbot\\chatbot2\\Lib\\q and a\\JNTUH_Student_Services_FAQ.csv")
    data = loader.load()
    faiss_index = FAISS.from_documents(documents=data, embedding=embeddings)


    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that combines the following question answers to generate single question and answer {question_answer}.",
        ),
        ("human", "{actual_question}"),
    ]
)
    chain = prompt | llm
    que = user_input
    result = populateResult(que,faiss_index)
    return {
        "AI" : chain.invoke(
        {
            "question_answer": result,
            "actual_question": que,
        }
    ).content,
        "related_Doc": result
        }

