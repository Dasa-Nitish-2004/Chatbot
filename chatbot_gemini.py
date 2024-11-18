from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS


class ChatBot:
    def __init__(self, csv_path, api_key, model_name):
        # Initialize attributes
        self.csv_path = csv_path
        self.api_key = api_key
        self.model_name = model_name

        # Initialize embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )

        # Load data from CSV
        self.loader = CSVLoader(file_path=csv_path)
        self.data = self.loader.load()

        # Create FAISS index and retriever
        self.faiss_index = FAISS.from_documents(documents=self.data, embedding=self.embeddings)
        self.retriever = self.faiss_index.as_retriever()

        # Initialize the language model
        self.llm = ChatGoogleGenerativeAI(
            api_key=api_key,
            model=model_name,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

        # Define the prompt template
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant that combines the following question answers to generate a single question and answer: {question_answer}."
                ),
                ("human", "{actual_question}")
            ]
        )

        # Chain prompt with the language model
        self.chain = self.prompt | self.llm

    def populate_result(self, question):
        try:
            # Retrieve relevant documents
            relevant_docs = self.retriever.get_relevant_documents(question)

            # Process retrieved documents into a structured format
            result = []
            for doc in relevant_docs:
                prompt = doc.page_content.split(r'?\n')[0].replace('ï»¿prompt: ', '')
                response = doc.page_content.split('?')[1].replace('\nresponse: ', '')
                result.append({"question": prompt, "answer": response})
            return result
        except Exception as e:
            print(f"Error in populate_result: {e}")
            return []

    def ask_question(self, user_input):
        try:
            # Get relevant results
            result = self.populate_result(user_input)

            if not result:
                return {
                    "error": "No relevant documents found.",
                    "related_Doc": []
                }

            # Generate the AI response
            ai_response = self.chain.invoke(
                {
                    "question_answer": result,
                    "actual_question": user_input
                }
            ).content

            return {
                "AI": ai_response,
                "related_Doc": result
            }
        except Exception as e:
            print(f"Error in ask_question: {e}")
            return {"error": f"An error occurred: {str(e)}"}


# Example Usage
if __name__ == "__main__":
    chatbot = ChatBot(
        csv_path="D:\\nitish\\python\\ml\\codebasics\\chatbot\\chatbot2\\Lib\\q and a\\JNTUH_Student_Services_FAQ.csv",
        api_key="AIzaSyAjhrUX0W18VR8yIKxKsu-8gs7qCskFf3A",
        model_name="gemini-1.5-pro"
    )

    user_question = "about JNTUH college?"
    response = chatbot.ask_question(user_question)
    print(response)
