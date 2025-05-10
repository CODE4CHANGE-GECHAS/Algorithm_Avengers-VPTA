from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import threading

app = Flask(__name__)
CORS(app)

# Dictionary for mapping conditions to exercises (if needed)
condition_to_exercise = {
    "stroke": "Stroke Rehabilitation",
    "arthritis": "Arthritis",
    "sports": "Sports Injury Recovery",
    "surgery": "Post Surgery Rehab",
    "flexibility": "General Flexibility"
}

@app.route('/api/get_exercises', methods=['POST'])
def get_exercises():
    data = request.get_json()
    condition = data.get("condition", "").lower()

    # Decide the exercise type
    selected_exercise = condition_to_exercise.get(condition, "General Flexibility")

    # Launch the main program in a separate thread to avoid blocking
    def run_main_program():
        subprocess.run(["python", "main_prgm.py", selected_exercise])

    threading.Thread(target=run_main_program).start()

    return jsonify({"status": "started", "exercise": selected_exercise})

if __name__ == '__main__':
    app.run(debug=True)
