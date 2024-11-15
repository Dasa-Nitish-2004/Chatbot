from flask import Flask, request, jsonify
# from your_chatbot_module import chatbot_response  # Replace with your actual chatbot module and function
import chatbot as cb
app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    
    print("entered chat functions")
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({'error': 'No input provided'}), 400
    
    
    response = cb.AskQuestion(user_input)  # Call your chatbot function
    return jsonify(response)

if __name__ == '__main__':
    app.run()
