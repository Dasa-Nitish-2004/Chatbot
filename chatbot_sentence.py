from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
# from langchain.vectorstores import FAISS


class ChatBot:
    def __init__(self, csv_path):
        # Initialize attributes
        self.csv_path = csv_path
        

        embedding_model = SentenceTransformer(r'D:\nitish\python\ml\codebasics\chatbot\chatbot2\Lib\q and a\fine_tuned_model',use_auth_token="")


        # Load data from CSV
        self.loader = CSVLoader(file_path=csv_path)
        self.data = self.loader.load()

        documents = [doc.page_content for doc in self.data]  # Assuming data contains your FAQ documents
        embeddings = embedding_model.encode(documents)

        text_embeddings = list(zip(documents, embeddings))

        self.faiss_index = FAISS.from_embeddings(
        text_embeddings=text_embeddings,  # Combined text and embeddings
        embedding=embedding_model.encode  # Embedding function
        )
        # Create FAISS index and retriever
        # self.faiss_index = FAISS.from_documents(documents=self.data, embedding=self.embeddings)
        self.retriever = self.faiss_index.as_retriever()


    def populate_result(self, question):
        try:
            # Retrieve relevant documents
            relevant_docs = self.retriever.get_relevant_documents(question)
            # print(relevant_docs)

            # Process retrieved documents into a structured format
            result = []
            for doc in relevant_docs:
                
                prompt = doc.page_content.split('|')[0].replace('labels: 1\nsentence_1: ', '')
                response = doc.page_content.split('|')[1].replace('\nsentence_2: ', '')
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

            # Generate the AI
            return result
        
        except Exception as e:
            print(f"Error in ask_question: {e}")
            return {"error": f"An error occurred: {str(e)}"}



# Example Usage
# if __name__ == "_main_":
chatbot = ChatBot(
        csv_path=r"D:\nitish\python\ml\codebasics\chatbot\chatbot2\Lib\q and a\JNTUH_Student_Services_FAQ_updated.csv",
    )

user_question = "about JNTUH college?"
response = chatbot.ask_question(user_question)
print(response)